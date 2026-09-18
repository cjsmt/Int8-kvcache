"""INT8 KV cache write for vLLM 0.26.x: symmetric per-token-per-head scale into paged INT8."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

INT8_QMAX = tl.constexpr(127.0)
INT8_QMIN = tl.constexpr(-127.0)


@triton.jit
def _reshape_and_cache_int8_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    slot_mapping_ptr,
    # key [num_tokens, Hkv, D]
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    # cache [num_blocks, BS, Hkv, D]
    stride_kcb,
    stride_kcs,
    stride_kch,
    stride_kcd,
    stride_vcb,
    stride_vcs,
    stride_vch,
    stride_vcd,
    # scale cache [num_blocks, BS, Hkv]
    stride_ksb,
    stride_kss,
    stride_ksh,
    stride_vsb,
    stride_vss,
    stride_vsh,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """grid (num_tokens, num_kv_heads): quant K/V, map slot, write INT8 + FP32 scale."""
    token_id = tl.program_id(0)
    kv_head = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    d_mask = offs_d < HEAD_DIM
    slot = tl.load(slot_mapping_ptr + token_id).to(tl.int64)
    valid_slot = slot >= 0
    physical_block = slot // BLOCK_SIZE
    block_offset = slot % BLOCK_SIZE
    key_offsets = token_id * stride_kt + kv_head * stride_kh + offs_d * stride_kd
    k = tl.load(
        key_ptr + key_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    value_offsets = token_id * stride_vt + kv_head * stride_vh + offs_d * stride_vd
    v = tl.load(
        value_ptr + value_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    k_absmax = tl.max(tl.abs(k), axis=0)
    v_absmax = tl.max(tl.abs(v), axis=0)
    k_scale = tl.maximum(
        k_absmax / INT8_QMAX,
        1.0e-6,
    )
    v_scale = tl.maximum(
        v_absmax / INT8_QMAX,
        1.0e-6,
    )
    k_q = k / k_scale
    v_q = v / v_scale
    k_q = tl.maximum(
        tl.minimum(k_q, INT8_QMAX),
        INT8_QMIN,
    )
    v_q = tl.maximum(
        tl.minimum(v_q, INT8_QMAX),
        INT8_QMIN,
    )
    # round half away from zero (no tl.rint on Triton 3.6)
    k_q = tl.where(k_q >= 0, tl.floor(k_q + 0.5), tl.ceil(k_q - 0.5)).to(tl.int8)
    v_q = tl.where(v_q >= 0, tl.floor(v_q + 0.5), tl.ceil(v_q - 0.5)).to(tl.int8)
    k_cache_offsets = (
        physical_block * stride_kcb
        + block_offset * stride_kcs
        + kv_head * stride_kch
        + offs_d * stride_kcd
    )
    v_cache_offsets = (
        physical_block * stride_vcb
        + block_offset * stride_vcs
        + kv_head * stride_vch
        + offs_d * stride_vcd
    )
    cache_mask = d_mask & valid_slot
    tl.store(
        key_cache_ptr + k_cache_offsets,
        k_q,
        mask=cache_mask,
    )
    tl.store(
        value_cache_ptr + v_cache_offsets,
        v_q,
        mask=cache_mask,
    )
    k_scale_offset = (
        physical_block * stride_ksb + block_offset * stride_kss + kv_head * stride_ksh
    )
    v_scale_offset = (
        physical_block * stride_vsb + block_offset * stride_vss + kv_head * stride_vsh
    )
    tl.store(
        k_scale_cache_ptr + k_scale_offset,
        k_scale,
        mask=valid_slot,
    )
    tl.store(
        v_scale_cache_ptr + v_scale_offset,
        v_scale,
        mask=valid_slot,
    )


def reshape_and_cache_int8(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Dynamic INT8 quant; write K/V + per-(token,head) scales into paged cache."""
    assert key.is_cuda
    assert value.is_cuda
    assert key_cache.is_cuda
    assert value_cache.is_cuda
    assert slot_mapping.is_cuda
    assert key.ndim == 3
    assert value.ndim == 3
    assert key.shape == value.shape
    num_tokens, num_kv_heads, head_dim = key.shape
    assert key_cache.ndim == 4
    assert value_cache.ndim == 4
    assert key_cache.dtype == torch.int8
    assert value_cache.dtype == torch.int8
    assert k_scale_cache.dtype == torch.float32
    assert v_scale_cache.dtype == torch.float32
    assert slot_mapping.numel() == num_tokens
    num_blocks, block_size, cache_heads, cache_dim = key_cache.shape
    assert cache_heads == num_kv_heads
    assert cache_dim == head_dim
    assert value_cache.shape == key_cache.shape
    expected_scale_shape = (
        num_blocks,
        block_size,
        num_kv_heads,
    )
    assert tuple(k_scale_cache.shape) == expected_scale_shape
    assert tuple(v_scale_cache.shape) == expected_scale_shape
    if not slot_mapping.is_contiguous():
        slot_mapping = slot_mapping.contiguous()
    head_dim_pad = triton.next_power_of_2(head_dim)
    grid = (
        num_tokens,
        num_kv_heads,
    )
    _reshape_and_cache_int8_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
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
        k_scale_cache.stride(0),
        k_scale_cache.stride(1),
        k_scale_cache.stride(2),
        v_scale_cache.stride(0),
        v_scale_cache.stride(1),
        v_scale_cache.stride(2),
        HEAD_DIM=head_dim,
        HEAD_DIM_PAD=head_dim_pad,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


def allocate_int8_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device | str = "cuda",
):
    """Allocate INT8 K/V caches and FP32 scale caches."""
    key_cache = torch.empty(
        (
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        ),
        device=device,
        dtype=torch.int8,
    )
    value_cache = torch.empty_like(key_cache)
    k_scale_cache = torch.empty(
        (
            num_blocks,
            block_size,
            num_kv_heads,
        ),
        device=device,
        dtype=torch.float32,
    )
    v_scale_cache = torch.empty_like(k_scale_cache)
    return (
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
    )


def dequantize_int8_cache(
    cache: torch.Tensor,
    scale_cache: torch.Tensor,
) -> torch.Tensor:
    """Debug: INT8 cache * broadcast scale -> FP32."""
    assert cache.dtype == torch.int8
    return cache.float() * scale_cache[..., None]


def tensor_nbytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def print_int8_cache_memory(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
):
    kv_bytes = tensor_nbytes(key_cache) + tensor_nbytes(value_cache)
    scale_bytes = tensor_nbytes(k_scale_cache) + tensor_nbytes(v_scale_cache)
    total = kv_bytes + scale_bytes
    print("=" * 70)
    print("INT8 KV Cache Memory")
    print("=" * 70)
    print(f"K/V INT8 : {kv_bytes / 1024**2:.3f} MB")
    print(f"Scales    : {scale_bytes / 1024**2:.3f} MB")
    print(f"Total     : {total / 1024**2:.3f} MB")
from src.triton_ops.int8_cache_write import int8_kv_cache_write


def runtime_int8_cache_write(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    assert key.ndim == 3
    assert value.shape == key.shape
    assert key_cache.dtype == torch.int8
    assert value_cache.dtype == torch.int8
    assert slot_mapping.ndim == 1
    assert slot_mapping.numel() == key.shape[0]
    assert k_scale.shape == (key.shape[1],)
    assert v_scale.shape == (key.shape[1],)
    int8_kv_cache_write(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
    )
