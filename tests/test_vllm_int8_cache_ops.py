import torch

from vllm_int8.vllm_int8_cache_ops import (
    allocate_int8_kv_cache,
    reshape_and_cache_int8,
    dequantize_int8_cache,
    print_int8_cache_memory,
)


def main():

    torch.manual_seed(0)

    device = "cuda"

    num_blocks = 32
    block_size = 16

    num_tokens = 23

    num_kv_heads = 4
    head_dim = 128

    # ========================================================
    # new K/V from Qwen attention projection
    # ========================================================

    key = torch.randn(
        num_tokens,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )

    value = torch.randn_like(key)

    # ========================================================
    # allocate cache
    # ========================================================

    (
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
    ) = allocate_int8_kv_cache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )

    # ========================================================
    # Construct non-trivial slot mapping
    #
    # deliberately cross block boundary:
    #
    # slots:
    # 7 ... 29
    # ========================================================

    slot_mapping = torch.arange(
        7,
        7 + num_tokens,
        device=device,
        dtype=torch.int64,
    )

    # ========================================================
    # INT8 cache write
    # ========================================================

    reshape_and_cache_int8(
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
    )

    torch.cuda.synchronize()

    # ========================================================
    # check every written token
    # ========================================================

    k_errors = []
    v_errors = []

    for token_idx in range(num_tokens):
        slot = int(slot_mapping[token_idx].item())

        block = slot // block_size
        offset = slot % block_size

        # --------------------------------------------
        # dequant one token
        # --------------------------------------------

        k_dequant = (
            key_cache[
                block,
                offset,
            ].float()
            * k_scale_cache[
                block,
                offset,
            ][:, None]
        )

        v_dequant = (
            value_cache[
                block,
                offset,
            ].float()
            * v_scale_cache[
                block,
                offset,
            ][:, None]
        )

        k_ref = key[token_idx].float()
        v_ref = value[token_idx].float()

        k_error = (k_dequant - k_ref).abs().mean()

        v_error = (v_dequant - v_ref).abs().mean()

        k_errors.append(k_error.item())

        v_errors.append(v_error.item())

    mean_k_error = sum(k_errors) / len(k_errors)

    mean_v_error = sum(v_errors) / len(v_errors)

    print()
    print("=" * 70)
    print("INT8 Cache Write Correctness")
    print("=" * 70)

    print("mean K error:", mean_k_error)

    print("mean V error:", mean_v_error)

    print("max K error:", max(k_errors))

    print("max V error:", max(v_errors))

    print()

    print_int8_cache_memory(
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
    )

    # INT8 per-token-head quantization should
    # normally be much better than this threshold.
    assert mean_k_error < 0.02
    assert mean_v_error < 0.02

    print()
    print("PASS")


if __name__ == "__main__":
    main()
