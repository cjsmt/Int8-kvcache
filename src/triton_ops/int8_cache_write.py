import torch
import triton
import triton.language as tl


@triton.jit
def _int8_kv_cache_write_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    # key strides
    stride_kt,
    stride_kh,
    stride_kd,
    # value strides
    stride_vt,
    stride_vh,
    stride_vd,
    # key cache strides
    stride_kcb,
    stride_kcs,
    stride_kch,
    stride_kcd,
    # value cache strides
    stride_vcb,
    stride_vcs,
    stride_vch,
    stride_vcd,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
):
    """BF16/FP16 K,V -> static per-head INT8 -> paged cache write.
    One program: (token, kv_head). grid=(num_tokens, num_kv_heads).
    Inputs: key/value [T,Hkv,D]; cache [num_blocks,BS,Hkv,D]; scale [Hkv].
    """
    token_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    # slot = physical_block * BLOCK_SIZE + offset_in_block; slot<0 => skip write
    slot = tl.load(slot_mapping_ptr + token_id).to(tl.int64)
    valid_slot = slot >= 0
    safe_slot = tl.where(valid_slot, slot, 0)
    physical_block = safe_slot // BLOCK_SIZE
    offset_in_block = safe_slot % BLOCK_SIZE
    k_scale = tl.maximum(tl.load(k_scale_ptr + head_id).to(tl.float32), 1e-6)
    v_scale = tl.maximum(tl.load(v_scale_ptr + head_id).to(tl.float32), 1e-6)
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    valid_d = offs_d < HEAD_DIM
    key_offsets = token_id * stride_kt + head_id * stride_kh + offs_d * stride_kd
    value_offsets = token_id * stride_vt + head_id * stride_vh + offs_d * stride_vd
    key = tl.load(key_ptr + key_offsets, mask=valid_d, other=0.0).to(tl.float32)
    value = tl.load(value_ptr + value_offsets, mask=valid_d, other=0.0).to(tl.float32)

    # Symmetric INT8 in [-127, 127]; round-half-away-from-zero (tests allow diff<=1 vs torch.round)
    key_scaled = key / k_scale
    value_scaled = value / v_scale
    key_rounded = tl.where(key_scaled >= 0, tl.floor(key_scaled + 0.5), tl.ceil(key_scaled - 0.5))
    value_rounded = tl.where(value_scaled >= 0, tl.floor(value_scaled + 0.5), tl.ceil(value_scaled - 0.5))
    key_int8 = tl.maximum(tl.minimum(key_rounded, 127.0), -127.0).to(tl.int8)
    value_int8 = tl.maximum(tl.minimum(value_rounded, 127.0), -127.0).to(tl.int8)

    # cache[physical_block, offset, head, :] via strides
    key_cache_offsets = (
        physical_block * stride_kcb
        + offset_in_block * stride_kcs
        + head_id * stride_kch
        + offs_d * stride_kcd
    )
    value_cache_offsets = (
        physical_block * stride_vcb
        + offset_in_block * stride_vcs
        + head_id * stride_vch
        + offs_d * stride_vcd
    )
    write_mask = valid_slot & valid_d
    tl.store(key_cache_ptr + key_cache_offsets, key_int8, mask=write_mask)
    tl.store(value_cache_ptr + value_cache_offsets, value_int8, mask=write_mask)


def int8_kv_cache_write(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_warps: int = 4,
    num_stages: int = 2,
):
    """Quantize BF16/FP16 K,V with static per-head scales and write INT8 paged cache.
    key/value: [T, Hkv, D]; key_cache/value_cache: [num_blocks, BS, Hkv, D] int8;
    slot_mapping: [T] (slot=block*BS+offset; slot<0 skips); k_scale/v_scale: [Hkv].
    """
    assert key.is_cuda and value.is_cuda
    assert key_cache.is_cuda and value_cache.is_cuda
    assert slot_mapping.is_cuda and k_scale.is_cuda and v_scale.is_cuda
    assert key.ndim == 3 and value.ndim == 3
    assert key_cache.ndim == 4 and value_cache.ndim == 4
    assert key.shape == value.shape and key_cache.shape == value_cache.shape
    assert key_cache.dtype == torch.int8 and value_cache.dtype == torch.int8
    assert key.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert value.dtype in (torch.float16, torch.bfloat16, torch.float32)
    num_tokens, num_kv_heads, head_dim = key.shape
    num_blocks, block_size, cache_num_kv_heads, cache_head_dim = key_cache.shape
    assert cache_num_kv_heads == num_kv_heads and cache_head_dim == head_dim
    assert slot_mapping.ndim == 1 and slot_mapping.numel() == num_tokens
    assert k_scale.ndim == 1 and v_scale.ndim == 1
    assert k_scale.numel() == num_kv_heads and v_scale.numel() == num_kv_heads
    assert slot_mapping.dtype in (torch.int32, torch.int64)
    if k_scale.dtype != torch.float32:
        k_scale = k_scale.float().contiguous()
    if v_scale.dtype != torch.float32:
        v_scale = v_scale.float().contiguous()
    head_dim_pad = triton.next_power_of_2(head_dim)
    grid = (num_tokens, num_kv_heads)
    _int8_kv_cache_write_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        HEAD_DIM_PAD=head_dim_pad,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def torch_int8_kv_cache_write_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
):
    """Simple PyTorch reference for Triton correctness tests (not for perf)."""
    assert key.ndim == 3
    num_tokens, num_kv_heads, head_dim = key.shape
    block_size = key_cache.shape[1]
    for token_id in range(num_tokens):
        slot = int(slot_mapping[token_id].item())
        if slot < 0:
            continue
        physical_block = slot // block_size
        offset_in_block = slot % block_size
        for head_id in range(num_kv_heads):
            key_int8 = torch.clamp(
                torch.round(key[token_id, head_id].float() / k_scale[head_id].float()),
                -127,
                127,
            ).to(torch.int8)
            value_int8 = torch.clamp(
                torch.round(value[token_id, head_id].float() / v_scale[head_id].float()),
                -127,
                127,
            ).to(torch.int8)
            key_cache[physical_block, offset_in_block, head_id, :] = key_int8
            value_cache[physical_block, offset_in_block, head_id, :] = value_int8


def dequantize_paged_cache_for_debug(cache: torch.Tensor, scale: torch.Tensor):
    """Debug helper: int8 cache [B,BS,Hkv,D] * scale[Hkv] -> float32."""
    assert cache.dtype == torch.int8
    return cache.float() * scale[None, None, :, None].float()
