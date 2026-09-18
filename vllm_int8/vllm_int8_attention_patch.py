from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from src.triton_ops.int8_paged_attention import (
    int8_paged_attention,
)

from vllm_int8.cache_manager import (
    get_layer_cache,
    has_layer_cache,
)

PatchMode = Literal[
    "shadow",
    "takeover",
    "int8_only",
]

_PATCHED = False

ATTN_COMPARE_COUNT: dict[str, int] = {}

ATTN_METRICS: dict[str, list[dict]] = {}


def _relative_l2(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    x = x.float()
    y = y.float()
    return ((x - y).norm() / y.norm().clamp_min(1e-12)).item()


def _cosine(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    return F.cosine_similarity(
        x.float().reshape(-1),
        y.float().reshape(-1),
        dim=0,
    ).item()


def _mae(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    return (x.float() - y.float()).abs().mean().item()


def _is_supported_decode(
    query,
    attn_metadata,
) -> bool:
    """MVP decode only: B=1, query [1,Hq,D], valid seq_lens/block_table; else vLLM fallback."""
    if attn_metadata is None:
        return False
    if query.ndim != 3:
        return False
    if query.shape[0] != 1:
        return False
    seq_lens = getattr(
        attn_metadata,
        "seq_lens",
        None,
    )
    block_table = getattr(
        attn_metadata,
        "block_table",
        None,
    )
    if seq_lens is None:
        return False
    if block_table is None:
        return False
    if seq_lens.numel() != 1:
        return False
    if block_table.ndim != 2:
        return False
    if block_table.shape[0] != 1:
        return False
    return True


def _run_int8_attention(
    layer,
    query,
    attn_metadata,
):
    layer_name = layer.layer_name
    shadow = get_layer_cache(layer_name)
    seq_lens = attn_metadata.seq_lens.contiguous()
    block_table = attn_metadata.block_table.contiguous()
    return int8_paged_attention(
        query,
        shadow.key_cache,
        shadow.value_cache,
        block_table,
        seq_lens,
        shadow.k_scale,
        shadow.v_scale,
        quantize_q=False,
    )


def apply_int8_attention_patch(
    *,
    mode: PatchMode = "shadow",
    verbose_layer: int = 0,
    max_compare_calls: int = 8,
):
    """shadow: BF16 out + compare; takeover: INT8 out; int8_only: decode INT8, prefill vLLM."""
    global _PATCHED
    if mode not in (
        "shadow",
        "takeover",
        "int8_only",
    ):
        raise ValueError(f"Unknown mode: {mode}")
    if _PATCHED:
        raise RuntimeError("Attention has already been patched in this Python process.")
    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionImpl,
    )
    original_forward = TritonAttentionImpl.forward
    def patched_forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        output_scale=None,
        output_block_scale=None,
    ):
        layer_name = getattr(
            layer,
            "layer_name",
            None,
        )
        if (
            layer_name is None
            or not has_layer_cache(layer_name)
            or not _is_supported_decode(
                query,
                attn_metadata,
            )
        ):
            return original_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        if mode in (
            "shadow",
            "takeover",
        ):
            bf16_result = original_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
            bf16_out = (
                output.detach()
                .clone()
                .reshape(
                    query.shape[0],
                    query.shape[1],
                    query.shape[2],
                )
            )
            int8_out = _run_int8_attention(
                layer,
                query,
                attn_metadata,
            )
            cosine = _cosine(
                int8_out,
                bf16_out,
            )
            rel_l2 = _relative_l2(
                int8_out,
                bf16_out,
            )
            mae = _mae(
                int8_out,
                bf16_out,
            )
            count = ATTN_COMPARE_COUNT.get(
                layer_name,
                0,
            )
            ATTN_COMPARE_COUNT[layer_name] = count + 1
            if layer_name not in ATTN_METRICS:
                ATTN_METRICS[layer_name] = []
            ATTN_METRICS[layer_name].append(
                {
                    "cosine": cosine,
                    "rel_l2": rel_l2,
                    "mae": mae,
                    "seq_len": int(attn_metadata.seq_lens[0].item()),
                }
            )
            if f"layers.{verbose_layer}." in layer_name and count < max_compare_calls:
                print()
                print("=" * 80)
                print(f"[INT8 Attention / {mode}]")
                print("=" * 80)
                print(
                    "layer:",
                    layer_name,
                )
                print(
                    "query:",
                    tuple(query.shape),
                    query.dtype,
                )
                print(
                    "seq_lens:",
                    attn_metadata.seq_lens.tolist(),
                )
                print(
                    "block_table:",
                    tuple(attn_metadata.block_table.shape),
                )
                print(
                    "cosine      :",
                    cosine,
                )
                print(
                    "relative L2 :",
                    rel_l2,
                )
                print(
                    "MAE         :",
                    mae,
                )
                print(
                    "INT8 NaN    :",
                    torch.isnan(int8_out).any().item(),
                )
            if mode == "shadow":
                return bf16_result
            assert not torch.isnan(int8_out).any()
            assert not torch.isinf(int8_out).any()
            output.copy_(int8_out.reshape_as(output))
            return output
        elif mode == "int8_only":
            int8_out = _run_int8_attention(
                layer,
                query,
                attn_metadata,
            )
            if torch.isnan(int8_out).any():
                raise RuntimeError(f"NaN in INT8 attention: {layer_name}")
            if torch.isinf(int8_out).any():
                raise RuntimeError(f"Inf in INT8 attention: {layer_name}")
            output.copy_(int8_out.reshape_as(output))
            return output
    TritonAttentionImpl.forward = patched_forward
    _PATCHED = True
    print()
    print("=" * 80)
    print("INT8 ATTENTION PATCH ENABLED")
    print("=" * 80)
    print("mode:", mode)
    if mode == "shadow":
        print("Decode result: BF16")
    elif mode == "takeover":
        print("Decode result: INT8 (BF16 still computed for reference)")
    elif mode == "int8_only":
        print("Decode result: PURE INT8")
