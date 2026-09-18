import torch
from vllm_int8.vllm_int8_kvcache_adapter import (
    convert_vllm_fused_kv_cache_to_int8,
    dequant_kv_cache,
)


def test_vllm_kv_quant():
    torch.manual_seed(0)
    device = "cuda"

    # 模拟 Qwen2.5-7B vLLM layout
    num_blocks = 128
    num_heads = 4
    block_size = 16
    head_dim = 128
    hidden_dim = 2 * head_dim
    kv_cache = torch.randn(
        num_blocks,
        num_heads,
        block_size,
        hidden_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    print()
    print("=" * 60)
    print("Original vLLM KV Cache")
    print("=" * 60)
    print("shape:", kv_cache.shape)
    print("dtype:", kv_cache.dtype)
    (
        k_int8,
        v_int8,
        k_scale,
        v_scale,
    ) = convert_vllm_fused_kv_cache_to_int8(kv_cache)
    print()
    print("=" * 60)
    print("INT8 Cache")
    print("=" * 60)
    print("K:", k_int8.shape, k_int8.dtype)
    print("V:", v_int8.shape, v_int8.dtype)
    print("K scale:", k_scale.shape)
    print("V scale:", v_scale.shape)

    # Accuracy check
    k_fp = dequant_kv_cache(k_int8, k_scale)
    v_fp = dequant_kv_cache(v_int8, v_scale)
    k_ref = kv_cache[..., :head_dim].float()
    v_ref = kv_cache[..., head_dim:].float()
    k_error = (k_fp - k_ref).abs().mean()
    v_error = (v_fp - v_ref).abs().mean()
    print()
    print("=" * 60)
    print("Quantization Error")
    print("=" * 60)
    print("K mean abs error:", k_error.item())
    print("V mean abs error:", v_error.item())
    assert k_error < 0.02
    assert v_error < 0.02
    print()
    print("PASS")
if __name__ == "__main__":
    test_vllm_kv_quant()
