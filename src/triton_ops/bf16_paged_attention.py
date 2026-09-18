import math

import torch

import triton
import triton.language as tl


# ============================================================
# BF16 PagedAttention Triton Kernel
# ============================================================


@triton.jit
def bf16_paged_attention_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    output_ptr,
    # ----------------------------
    # query stride
    # ----------------------------
    stride_qb,
    stride_qh,
    stride_qd,
    # ----------------------------
    # KV cache stride
    # ----------------------------
    stride_kb,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vbs,
    stride_vh,
    stride_vd,
    # ----------------------------
    # block table
    # ----------------------------
    stride_bt_b,
    stride_bt_l,
    # ----------------------------
    # output
    # ----------------------------
    stride_ob,
    stride_oh,
    stride_od,
    max_num_blocks,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_PAD: tl.constexpr,
    SM_SCALE: tl.constexpr,
):

    batch_id = tl.program_id(0)

    q_head = tl.program_id(1)

    # ----------------------------
    # GQA mapping
    # ----------------------------

    q_per_kv = NUM_Q_HEADS // NUM_KV_HEADS

    kv_head = q_head // q_per_kv

    seq_len = tl.load(seq_lens_ptr + batch_id).to(tl.int32)

    # ----------------------------
    # Load Query
    # ----------------------------

    offs_d = tl.arange(0, HEAD_DIM_PAD)

    d_mask = offs_d < HEAD_DIM

    q_ptrs = query_ptr + batch_id * stride_qb + q_head * stride_qh + offs_d * stride_qd

    q = tl.load(
        q_ptrs,
        mask=d_mask,
        other=0,
    ).to(tl.float32)

    # ----------------------------
    # Online softmax
    # ----------------------------

    m_i = -float("inf")

    l_i = 0.0

    acc = tl.zeros([HEAD_DIM_PAD], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_SIZE_PAD)

    n_mask = offs_n < BLOCK_SIZE

    # ========================================================
    # Loop over pages
    # ========================================================

    for logical_block in tl.range(0, max_num_blocks):
        block_start = logical_block * BLOCK_SIZE

        valid_page = block_start < seq_len

        physical_block = tl.load(
            block_tables_ptr + batch_id * stride_bt_b + logical_block * stride_bt_l,
            mask=valid_page,
            other=0,
        ).to(tl.int64)

        token_ids = block_start + offs_n

        token_mask = n_mask & (token_ids < seq_len)

        # ----------------------------
        # Load K
        # ----------------------------

        k_ptrs = (
            key_cache_ptr
            + physical_block * stride_kb
            + offs_n[:, None] * stride_kbs
            + kv_head * stride_kh
            + offs_d[None, :] * stride_kd
        )

        k = tl.load(
            k_ptrs,
            mask=(token_mask[:, None] & d_mask[None, :]),
            other=0,
        ).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1)

        scores *= SM_SCALE

        scores = tl.where(token_mask, scores, -float("inf"))

        tile_max = tl.max(scores, axis=0)

        m_new = tl.maximum(m_i, tile_max)

        alpha = tl.exp(m_i - m_new)

        p = tl.exp(scores - m_new)

        p = tl.where(token_mask, p, 0.0)

        l_new = l_i * alpha + tl.sum(p, axis=0)

        # ----------------------------
        # Load V
        # ----------------------------

        v_ptrs = (
            value_cache_ptr
            + physical_block * stride_vb
            + offs_n[:, None] * stride_vbs
            + kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )

        v = tl.load(
            v_ptrs,
            mask=(token_mask[:, None] & d_mask[None, :]),
            other=0,
        ).to(tl.float32)

        pv = tl.sum(p[:, None] * v, axis=0)

        acc = acc * alpha + pv

        m_i = m_new

        l_i = l_new

    # ----------------------------
    # normalize
    # ----------------------------

    out = acc / l_i

    out_ptrs = (
        output_ptr + batch_id * stride_ob + q_head * stride_oh + offs_d * stride_od
    )

    tl.store(out_ptrs, out, mask=d_mask)


# ============================================================
# Python Wrapper
# ============================================================


def bf16_paged_attention(
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
):

    B, Hq, D = query.shape

    _, BLOCK_SIZE, Hkv, _ = key_cache.shape

    output = torch.empty_like(query)

    grid = (
        B,
        Hq,
    )

    bf16_paged_attention_kernel[grid](
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_tables.shape[1],
        NUM_Q_HEADS=Hq,
        NUM_KV_HEADS=Hkv,
        HEAD_DIM=D,
        HEAD_DIM_PAD=triton.next_power_of_2(D),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_PAD=triton.next_power_of_2(BLOCK_SIZE),
        SM_SCALE=1 / math.sqrt(D),
        num_warps=4,
    )

    return output
