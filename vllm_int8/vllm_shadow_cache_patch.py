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


# 防止重复 patch
_PATCHED = False


# 调试统计
WRITE_CALL_COUNT = {}

NATIVE_CACHE_BYTES_BY_LAYER = {}


def apply_shadow_cache_patch(
    verbose_layer: int = 0,
):
    """
    Patch:

        TritonAttentionImpl.do_kv_cache_update

    目标：

    vLLM 原 BF16 cache write 保持不变。

    同时：

        new K/V
            ↓
        我们的 int8_kv_cache_write
            ↓
        INT8 shadow cache
    """

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

        # ====================================================
        # 1. 先让原始 vLLM 正常写 BF16 Cache
        #
        # 非常重要：
        #
        # Step 3 暂时不能破坏原 vLLM。
        # ====================================================

        original_do_kv_cache_update(
            self,
            layer,
            key,
            value,
            kv_cache,
            slot_mapping,
        )

        # ====================================================
        # 2. 当前 Layer Name
        # ====================================================

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

        # ====================================================
        # 3. key/value runtime shape
        #
        # vLLM Triton backend：
        #
        # key:
        # [num_tokens, Hkv, D]
        #
        # value:
        # [num_tokens, Hkv, D]
        # ====================================================

        assert key.ndim == 3
        assert value.ndim == 3

        (
            num_tokens,
            num_kv_heads,
            head_dim,
        ) = key.shape

        # ====================================================
        # 4. 从 vLLM 原 cache 推断：
        #
        # num_blocks
        # block_size
        #
        # vLLM 0.26 Triton layout：
        #
        # logical:
        # [num_blocks, Hkv, block_size, 2D]
        #
        # 我们之前 hook 的实际：
        #
        # [3900,4,16,256]
        # ====================================================

        num_blocks = kv_cache.shape[0]

        block_size = kv_cache.shape[2]

        # ====================================================
        # 5. 首次进入当前 layer：
        #
        # 分配 shadow INT8 Cache。
        # ====================================================

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

        # ====================================================
        # 6. slot_mapping
        #
        # vLLM 有时 metadata 会含 padding slot。
        #
        # 我们自己的 kernel 已支持 slot < 0。
        # ====================================================

        runtime_slots = slot_mapping.reshape(-1).contiguous()

        # key/value 有时可能包含 padded token。
        #
        # 正常来说长度应该一致。
        assert runtime_slots.numel() == num_tokens, (
            f"slot_mapping={runtime_slots.shape}, key={key.shape}"
        )

        # ====================================================
        # 7. 写我们的 INT8 shadow cache
        # ====================================================

        int8_kv_cache_write(
            key,
            value,
            shadow.key_cache,
            shadow.value_cache,
            runtime_slots,
            shadow.k_scale,
            shadow.v_scale,
        )

        # ====================================================
        # 8. Debug 计数
        # ====================================================

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
