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


# ============================================================
# Metrics
# ============================================================


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


# ============================================================
# 判断是不是我们目前支持的 Decode Case
# ============================================================


def _is_supported_decode(
    query,
    attn_metadata,
) -> bool:
    """
    当前 MVP 只支持：

        batch = 1
        decode query token = 1
        query = [1, Hq, D]

    Prefill、profiling 等情况全部 fallback 到原 vLLM。
    """

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


# ============================================================
# 调我们的 INT8 Kernel
# ============================================================


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


# ============================================================
# Main Patch
# ============================================================


def apply_int8_attention_patch(
    *,
    mode: PatchMode = "shadow",
    verbose_layer: int = 0,
    max_compare_calls: int = 8,
):
    """
    mode="shadow":

        BF16 forward
        INT8 forward
        compare
        返回 BF16


    mode="takeover":

        BF16 forward
        INT8 forward
        compare
        但最终给模型 INT8 output


    mode="int8_only":

        Prefill:
            原 BF16

        Decode:
            跳过原 BF16
            直接 INT8 PagedAttention
    """

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

    # ========================================================
    # Patched Forward
    # ========================================================

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

        # ----------------------------------------------------
        # Profiling / unsupported layer
        # ----------------------------------------------------

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

        # ====================================================
        # MODE 1 / 2
        #
        # shadow / takeover：
        # 先算原 BF16 reference
        # ====================================================

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

            # output 已经由 vLLM kernel 写入
            #
            # 注意 clone：
            # 后面 takeover 会覆盖 output。
            bf16_out = (
                output.detach()
                .clone()
                .reshape(
                    query.shape[0],
                    query.shape[1],
                    query.shape[2],
                )
            )

            # ================================================
            # INT8
            # ================================================

            int8_out = _run_int8_attention(
                layer,
                query,
                attn_metadata,
            )

            # ================================================
            # Metrics
            # ================================================

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

            # ================================================
            # Debug print
            # ================================================

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

            # ================================================
            # SHADOW
            #
            # 原 BF16 结果继续给模型
            # ================================================

            if mode == "shadow":
                return bf16_result

            # ================================================
            # TAKEOVER
            #
            # BF16 只是 reference，
            # 最终 output 改成 INT8。
            # ================================================

            assert not torch.isnan(int8_out).any()

            assert not torch.isinf(int8_out).any()

            output.copy_(int8_out.reshape_as(output))

            return output

        # ====================================================
        # MODE 3:
        #
        # PURE INT8 DECODE
        #
        # 不执行原 BF16 attention。
        # ====================================================

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

    # ========================================================
    # Apply patch
    # ========================================================

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
