"""
vllm_int8_kvcache_adapter.py


功能:

将 vLLM v2 attention backend 的 fused KV Cache:

BF16:

[num_blocks,
 num_kv_heads,
 block_size,
 2*head_dim]


转换为:

INT8:

K:

[num_blocks,
 num_kv_heads,
 block_size,
 head_dim]


V:

[num_blocks,
 num_kv_heads,
 block_size,
 head_dim]


同时生成:

per KV head scale


"""

import torch


# ============================================================
# Quantization
# ============================================================


def quantize_per_head(
    x: torch.Tensor,
):
    """
    Per KV Head INT8 quantization


    输入:

    x:

    [num_blocks,
     num_heads,
     block_size,
     head_dim]


    输出:

    x_int8

    scale


    scale:

    [num_heads]


    """

    assert x.ndim == 4

    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )

    # --------------------------------------------------------
    # 计算每个 KV head scale
    #
    # x:
    #
    # [B,H,S,D]
    #
    #
    # reduce:
    #
    # B,S,D
    #
    # 保留 H
    #
    # --------------------------------------------------------

    scale = x.float().abs().amax(dim=(0, 2, 3)) / 127.0

    scale = torch.clamp(scale, min=1e-6)

    # --------------------------------------------------------
    # quant
    # --------------------------------------------------------

    x_int8 = torch.round(x.float() / scale[None, :, None, None])

    x_int8 = torch.clamp(x_int8, -127, 127)

    return (x_int8.to(torch.int8), scale.to(torch.float32))


# ============================================================
# Main Adapter
# ============================================================


def convert_vllm_fused_kv_cache_to_int8(
    kv_cache: torch.Tensor,
):
    """
    将 vLLM fused KV cache 转换为 INT8。


    输入:

    kv_cache:

        [num_blocks,
         num_kv_heads,
         block_size,
         2*head_dim]


    dtype:

        bf16


    输出:


    k_cache_int8:

        [num_blocks,
         num_kv_heads,
         block_size,
         head_dim]


    v_cache_int8:

        same


    k_scale:

        [num_kv_heads]


    v_scale:

        [num_kv_heads]


    """

    assert kv_cache.ndim == 4

    num_blocks = kv_cache.shape[0]

    num_kv_heads = kv_cache.shape[1]

    block_size = kv_cache.shape[2]

    hidden = kv_cache.shape[3]

    assert hidden % 2 == 0, "vLLM fused KV cache last dimension must be 2*head_dim"

    head_dim = hidden // 2

    # --------------------------------------------------------
    # split K/V
    #
    # vLLM layout:
    #
    # [...,0:D]      K
    #
    # [...,D:2D]     V
    #
    # --------------------------------------------------------

    k_cache = kv_cache[..., :head_dim]

    v_cache = kv_cache[..., head_dim:]

    # --------------------------------------------------------
    # quant
    # --------------------------------------------------------

    k_int8, k_scale = quantize_per_head(k_cache)

    v_int8, v_scale = quantize_per_head(v_cache)

    return (
        k_int8,
        v_int8,
        k_scale,
        v_scale,
    )


# ============================================================
# Dequant helper
# ============================================================


def dequant_kv_cache(
    x_int8,
    scale,
):
    """
    Debug only


    INT8 -> FP32


    """

    return x_int8.float() * scale[None, :, None, None]
