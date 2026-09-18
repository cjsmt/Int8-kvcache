from __future__ import annotations

import torch

from src.triton_ops.int8_cache_write import (
    int8_kv_cache_write,
)

from vllm_int8.cache_manager import (
    allocate_layer_cache,
    get_layer_cache,
    has_layer_cache,
)

from vllm_int8.static_scales import (
    get_layer_scales,
)

_PATCHED = False

WRITE_CALL_COUNT = {}

NATIVE_CACHE_BYTES_BY_LAYER = {}


def apply_shadow_cache_patch(
    verbose_layer: int = 0,
):
    """Patch TritonAttentionImpl.do_kv_cache_update: keep vLLM BF16 write + mirror INT8 shadow."""
    global _PATCHED
    if _PATCHED:
        print("[INT8 Shadow] already patched")
        return
    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionImpl,
    )
    original_do_kv_cache_update = TritonAttentionImpl.do_kv_cache_update
    def patched_do_kv_cache_update(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        original_do_kv_cache_update(
            self,
            layer,
            key,
            value,
            kv_cache,
            slot_mapping,
        )
        layer_name = layer.layer_name
        if (
            layer_name
            not in NATIVE_CACHE_BYTES_BY_LAYER
        ):
            NATIVE_CACHE_BYTES_BY_LAYER[
                layer_name
            ] = (
                kv_cache.numel()
                *
                kv_cache.element_size()
            )
        assert key.ndim == 3
        assert value.ndim == 3
        (
            num_tokens,
            num_kv_heads,
            head_dim,
        ) = key.shape
        # vLLM Triton kv_cache: [num_blocks, Hkv, block_size, 2*D] (logical); block_size at dim 2
        num_blocks = kv_cache.shape[0]
        block_size = kv_cache.shape[2]
        if not has_layer_cache(layer_name):
            (
                k_scale,
                v_scale,
            ) = get_layer_scales(
                layer_name,
                key.device,
            )
            allocate_layer_cache(
                layer_name=layer_name,
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                k_scale=k_scale,
                v_scale=v_scale,
                device=key.device,
            )
            if f"layers.{verbose_layer}." in layer_name:
                print()
                print("=" * 80)
                print("[INT8 Shadow] allocate")
                print("=" * 80)
                print("layer:", layer_name)
                print(
                    "key runtime:",
                    tuple(key.shape),
                    key.dtype,
                )
                print(
                    "value runtime:",
                    tuple(value.shape),
                    value.dtype,
                )
                print(
                    "original kv cache:",
                    tuple(kv_cache.shape),
                    kv_cache.dtype,
                    kv_cache.stride(),
                )
                shadow = get_layer_cache(layer_name)
                print(
                    "INT8 K cache:",
                    tuple(shadow.key_cache.shape),
                    shadow.key_cache.dtype,
                )
                print(
                    "INT8 V cache:",
                    tuple(shadow.value_cache.shape),
                    shadow.value_cache.dtype,
                )
                print(
                    "K scale:",
                    shadow.k_scale,
                )
                print(
                    "V scale:",
                    shadow.v_scale,
                )
        shadow = get_layer_cache(layer_name)
        runtime_slots = slot_mapping.reshape(-1).contiguous()
        assert runtime_slots.numel() == num_tokens, (
            f"slot_mapping={runtime_slots.shape}, key={key.shape}"
        )
        int8_kv_cache_write(
            key,
            value,
            shadow.key_cache,
            shadow.value_cache,
            runtime_slots,
            shadow.k_scale,
            shadow.v_scale,
        )
        count = WRITE_CALL_COUNT.get(
            layer_name,
            0,
        )
        WRITE_CALL_COUNT[layer_name] = count + 1
        if f"layers.{verbose_layer}." in layer_name and count < 5:
            valid_slots = runtime_slots[runtime_slots >= 0]
            print()
            print(f"[INT8 Shadow] write layer={verbose_layer} call={count}")
            print(
                "num_tokens:",
                num_tokens,
            )
            print(
                "slot_mapping shape:",
                tuple(runtime_slots.shape),
            )
            if valid_slots.numel() > 0:
                print(
                    "first valid slots:",
                    valid_slots[: min(8, valid_slots.numel())].tolist(),
                )
    TritonAttentionImpl.do_kv_cache_update = patched_do_kv_cache_update
    _PATCHED = True
    print("[INT8 Shadow] TritonAttentionImpl.do_kv_cache_update patched.")
