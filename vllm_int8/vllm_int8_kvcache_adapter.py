"""Split vLLM fused BF16 KV [blocks,Hkv,BS,2*D] into INT8 K/V + per-head FP32 scales."""

import torch


def quantize_per_head(
    x: torch.Tensor,
):
    """Per-KV-head symmetric INT8. x [B,H,S,D] -> int8, scale [H]."""
    assert x.ndim == 4
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )
    scale = x.float().abs().amax(dim=(0, 2, 3)) / 127.0
    scale = torch.clamp(scale, min=1e-6)
    x_int8 = torch.round(x.float() / scale[None, :, None, None])
    x_int8 = torch.clamp(x_int8, -127, 127)
    return (x_int8.to(torch.int8), scale.to(torch.float32))


def convert_vllm_fused_kv_cache_to_int8(
    kv_cache: torch.Tensor,
):
    """Fused [blocks,Hkv,BS,2*D] bf16 -> K/V int8 + k_scale/v_scale [Hkv]."""
    assert kv_cache.ndim == 4
    num_blocks = kv_cache.shape[0]
    num_kv_heads = kv_cache.shape[1]
    block_size = kv_cache.shape[2]
    hidden = kv_cache.shape[3]
    assert hidden % 2 == 0, "vLLM fused KV cache last dimension must be 2*head_dim"
    head_dim = hidden // 2
    k_cache = kv_cache[..., :head_dim]
    v_cache = kv_cache[..., head_dim:]
    k_int8, k_scale = quantize_per_head(k_cache)
    v_int8, v_scale = quantize_per_head(v_cache)
    return (
        k_int8,
        v_int8,
        k_scale,
        v_scale,
    )


def dequant_kv_cache(
    x_int8,
    scale,
):
    """Debug: INT8 -> FP32 via broadcast scale."""
    return x_int8.float() * scale[None, :, None, None]
