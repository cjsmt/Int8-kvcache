"""INT8 PagedAttention — V4 (GQA reuse + tl.dot + autotune + split-KV)."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_GQA_AUTOTUNE_CONFIGS = [
    triton.Config({"NUM_BLOCKS_PER_TILE": 1}, num_warps=2, num_stages=2),
    triton.Config({"NUM_BLOCKS_PER_TILE": 1}, num_warps=4, num_stages=2),
    triton.Config({"NUM_BLOCKS_PER_TILE": 1}, num_warps=4, num_stages=3),
    triton.Config({"NUM_BLOCKS_PER_TILE": 1}, num_warps=8, num_stages=2),
    triton.Config({"NUM_BLOCKS_PER_TILE": 2}, num_warps=4, num_stages=2),
    triton.Config({"NUM_BLOCKS_PER_TILE": 2}, num_warps=4, num_stages=3),
    triton.Config({"NUM_BLOCKS_PER_TILE": 2}, num_warps=8, num_stages=3),
    triton.Config({"NUM_BLOCKS_PER_TILE": 4}, num_warps=4, num_stages=3),
    triton.Config({"NUM_BLOCKS_PER_TILE": 4}, num_warps=8, num_stages=3),
    triton.Config({"NUM_BLOCKS_PER_TILE": 4}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=_GQA_AUTOTUNE_CONFIGS, key=["HEAD_DIM", "CACHE_BLOCK_SIZE", "GROUP_SIZE"])
@triton.jit
def _int8_paged_attention_gqa_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    k_scale_ptr,
    v_scale_ptr,
    output_ptr,
    partial_out_ptr,
    partial_m_ptr,
    partial_l_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kcb,
    stride_kcs,
    stride_kch,
    stride_kcd,
    stride_vcb,
    stride_vcs,
    stride_vch,
    stride_vcd,
    stride_btb,
    stride_btblk,
    stride_ob,
    stride_oh,
    stride_od,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pod,
    stride_pmb,
    stride_pmh,
    stride_pms,
    stride_plb,
    stride_plh,
    stride_pls,
    max_num_blocks,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS_PER_TILE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    QUANTIZE_Q: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    USE_SPLIT_KV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    split_id = tl.program_id(2)

    q_head_start = kv_head_id * GROUP_SIZE
    seq_len = tl.load(seq_lens_ptr + batch_id).to(tl.int32)

    offs_g = tl.arange(0, GROUP_PAD)
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    g_mask = offs_g < GROUP_SIZE
    d_mask = offs_d < HEAD_DIM

    q_ptrs = (
        query_ptr
        + batch_id * stride_qb
        + (q_head_start + offs_g[:, None]) * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=g_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    if QUANTIZE_Q:
        q_amax = tl.max(tl.abs(q), axis=1)
        q_scale = tl.maximum(q_amax / 127.0, 1e-6)
        q_scaled = q / q_scale[:, None]
        q_rounded = tl.where(
            q_scaled >= 0,
            tl.floor(q_scaled + 0.5),
            -tl.floor(-q_scaled + 0.5),
        )
        q_for_dot = tl.maximum(tl.minimum(q_rounded, 127.0), -127.0)
    else:
        q_scale = tl.full([GROUP_PAD], 1.0, dtype=tl.float32)
        q_for_dot = q

    k_scale = tl.maximum(tl.load(k_scale_ptr + kv_head_id).to(tl.float32), 1e-6)
    v_scale = tl.maximum(tl.load(v_scale_ptr + kv_head_id).to(tl.float32), 1e-6)

    m_i = tl.full([GROUP_PAD], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([GROUP_PAD], dtype=tl.float32)
    # Unscaled V accumulator; apply v_scale once at the end (dequant fusion)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM_PAD], dtype=tl.float32)

    BLOCK_N: tl.constexpr = NUM_BLOCKS_PER_TILE * CACHE_BLOCK_SIZE
    offs_n = tl.arange(0, BLOCK_N)
    page_idx_n = offs_n // CACHE_BLOCK_SIZE
    in_page_n = offs_n % CACHE_BLOCK_SIZE
    block_offs = tl.arange(0, NUM_BLOCKS_PER_TILE)

    if USE_SPLIT_KV:
        blocks_per_split = (max_num_blocks + NUM_SPLITS - 1) // NUM_SPLITS
        block_lo = split_id * blocks_per_split
        block_hi = tl.minimum(block_lo + blocks_per_split, max_num_blocks)
    else:
        block_lo = 0
        block_hi = max_num_blocks

    q_tc = q_for_dot.to(tl.bfloat16)

    for logical_block_base in tl.range(0, max_num_blocks, NUM_BLOCKS_PER_TILE):
        tile_in_split = (logical_block_base >= block_lo) & (logical_block_base < block_hi)
        page_start_token = logical_block_base * CACHE_BLOCK_SIZE
        page_valid = (page_start_token < seq_len) & tile_in_split

        logical_ids = logical_block_base + block_offs
        valid_block = (
            page_valid
            & (logical_ids < max_num_blocks)
            & ((logical_ids * CACHE_BLOCK_SIZE) < seq_len)
        )
        phys = tl.load(
            block_tables_ptr + batch_id * stride_btb + logical_ids * stride_btblk,
            mask=valid_block,
            other=0,
        ).to(tl.int64)

        # Map each tile row to its physical block id
        phys_n = tl.sum(
            (page_idx_n[:, None] == block_offs[None, :]).to(tl.int64) * phys[None, :],
            axis=1,
        )
        logical_token = page_start_token + offs_n
        row_block_ok = tl.sum(
            (page_idx_n[:, None] == block_offs[None, :]).to(tl.int1) & valid_block[None, :],
            axis=1,
        ) > 0
        valid_n = (offs_n < BLOCK_N) & (logical_token < seq_len) & row_block_ok & page_valid

        k_ptrs = (
            key_cache_ptr
            + phys_n[:, None] * stride_kcb
            + in_page_n[:, None] * stride_kcs
            + kv_head_id * stride_kch
            + offs_d[None, :] * stride_kcd
        )
        v_ptrs = (
            value_cache_ptr
            + phys_n[:, None] * stride_vcb
            + in_page_n[:, None] * stride_vcs
            + kv_head_id * stride_vch
            + offs_d[None, :] * stride_vcd
        )
        kv_mask = valid_n[:, None] & d_mask[None, :]

        k_tile = tl.load(k_ptrs, mask=kv_mask, other=0).to(tl.float32)
        v_tile = tl.load(v_ptrs, mask=kv_mask, other=0).to(tl.float32)

        # QK: [GROUP_PAD, D] @ [D, BLOCK_N] -> [GROUP_PAD, BLOCK_N]
        scores = tl.dot(q_tc, tl.trans(k_tile.to(tl.bfloat16))).to(tl.float32)
        if QUANTIZE_Q:
            scores = scores * (q_scale[:, None] * (k_scale * SM_SCALE))
        else:
            scores = scores * (k_scale * SM_SCALE)
        scores = tl.where(valid_n[None, :] & g_mask[:, None], scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        alpha = tl.exp(m_i - m_new)
        alpha = tl.where(m_i == -float("inf"), 0.0, alpha)
        alpha = tl.where(page_valid, alpha, 1.0)
        m_new = tl.where(page_valid, m_new, m_i)

        p = tl.exp(scores - m_new[:, None])
        p = tl.where(valid_n[None, :] & g_mask[:, None] & page_valid, p, 0.0)
        p_sum = tl.sum(p, axis=1)
        l_new = tl.where(page_valid, l_i * alpha + p_sum, l_i)

        # PV without v_scale inside the loop
        pv = tl.dot(p.to(tl.bfloat16), v_tile.to(tl.bfloat16)).to(tl.float32)
        acc = tl.where(page_valid, acc * alpha[:, None] + pv, acc)

        m_i = m_new
        l_i = l_new

    l_safe = tl.maximum(l_i, 1e-20)
    output = (acc * v_scale) / l_safe[:, None]

    if USE_SPLIT_KV:
        tl.store(
            partial_out_ptr
            + batch_id * stride_pob
            + (q_head_start + offs_g[:, None]) * stride_poh
            + split_id * stride_pos
            + offs_d[None, :] * stride_pod,
            output,
            mask=g_mask[:, None] & d_mask[None, :],
        )
        tl.store(
            partial_m_ptr
            + batch_id * stride_pmb
            + (q_head_start + offs_g) * stride_pmh
            + split_id * stride_pms,
            m_i,
            mask=g_mask,
        )
        tl.store(
            partial_l_ptr
            + batch_id * stride_plb
            + (q_head_start + offs_g) * stride_plh
            + split_id * stride_pls,
            l_i,
            mask=g_mask,
        )
    else:
        tl.store(
            output_ptr
            + batch_id * stride_ob
            + (q_head_start + offs_g[:, None]) * stride_oh
            + offs_d[None, :] * stride_od,
            output,
            mask=g_mask[:, None] & d_mask[None, :],
        )


@triton.jit
def _merge_split_kv_kernel(
    partial_out_ptr,
    partial_m_ptr,
    partial_l_ptr,
    output_ptr,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pod,
    stride_pmb,
    stride_pmh,
    stride_pms,
    stride_plb,
    stride_plh,
    stride_pls,
    stride_ob,
    stride_oh,
    stride_od,
    NUM_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
):
    batch_id = tl.program_id(0)
    q_head_id = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM_PAD)
    d_mask = offs_d < HEAD_DIM

    m = -float("inf")
    l = 0.0
    acc = tl.zeros([HEAD_DIM_PAD], dtype=tl.float32)

    for s in tl.static_range(0, NUM_SPLITS):
        m_s = tl.load(partial_m_ptr + batch_id * stride_pmb + q_head_id * stride_pmh + s * stride_pms)
        l_s = tl.load(partial_l_ptr + batch_id * stride_plb + q_head_id * stride_plh + s * stride_pls)
        o_s = tl.load(
            partial_out_ptr
            + batch_id * stride_pob
            + q_head_id * stride_poh
            + s * stride_pos
            + offs_d * stride_pod,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)

        # partial_out = (acc_unscaled * v_scale) / l_s  → numerator = o_s * l_s
        num_s = tl.where(l_s > 0, o_s * l_s, 0.0)

        m_new = tl.maximum(m, m_s)
        alpha = tl.exp(m - m_new)
        alpha_s = tl.exp(m_s - m_new)
        alpha = tl.where(m == -float("inf"), 0.0, alpha)
        alpha_s = tl.where(m_s == -float("inf"), 0.0, alpha_s)

        acc = acc * alpha + num_s * alpha_s
        l = l * alpha + l_s * alpha_s
        m = m_new

    out = acc / tl.maximum(l, 1e-20)
    tl.store(
        output_ptr + batch_id * stride_ob + q_head_id * stride_oh + offs_d * stride_od,
        out,
        mask=d_mask,
    )


# ---------------------------------------------------------------------------
# Legacy V3 per-head kernel (A/B microbench)
# ---------------------------------------------------------------------------


@triton.jit
def _int8_paged_attention_kernel_v3(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    k_scale_ptr,
    v_scale_ptr,
    output_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kcb,
    stride_kcs,
    stride_kch,
    stride_kcd,
    stride_vcb,
    stride_vcs,
    stride_vch,
    stride_vcd,
    stride_btb,
    stride_btblk,
    stride_ob,
    stride_oh,
    stride_od,
    max_num_blocks,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    CACHE_BLOCK_SIZE_PAD: tl.constexpr,
    SM_SCALE: tl.constexpr,
    QUANTIZE_Q: tl.constexpr,
):
    batch_id = tl.program_id(0)
    query_head_id = tl.program_id(1)
    kv_head_id = query_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)
    seq_len = tl.load(seq_lens_ptr + batch_id).to(tl.int32)

    offs_d = tl.arange(0, HEAD_DIM_PAD)
    d_mask = offs_d < HEAD_DIM
    q = tl.load(
        query_ptr + batch_id * stride_qb + query_head_id * stride_qh + offs_d * stride_qd,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    if QUANTIZE_Q:
        q_amax = tl.max(tl.abs(q), axis=0)
        q_scale = tl.maximum(q_amax / 127.0, 1e-6)
        q_scaled = q / q_scale
        q_rounded = tl.where(q_scaled >= 0, tl.floor(q_scaled + 0.5), -tl.floor(-q_scaled + 0.5))
        q_for_dot = tl.maximum(tl.minimum(q_rounded, 127.0), -127.0).to(tl.float32)
    else:
        q_scale = 1.0
        q_for_dot = q

    k_scale = tl.maximum(tl.load(k_scale_ptr + kv_head_id).to(tl.float32), 1e-6)
    v_scale = tl.maximum(tl.load(v_scale_ptr + kv_head_id).to(tl.float32), 1e-6)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM_PAD], dtype=tl.float32)
    offs_n = tl.arange(0, CACHE_BLOCK_SIZE_PAD)
    valid_cache_offset = offs_n < CACHE_BLOCK_SIZE

    for logical_block_id in tl.range(0, max_num_blocks):
        page_start_token = logical_block_id * CACHE_BLOCK_SIZE
        page_valid = page_start_token < seq_len
        physical_block = tl.load(
            block_tables_ptr + batch_id * stride_btb + logical_block_id * stride_btblk,
            mask=page_valid,
            other=0,
        ).to(tl.int64)
        logical_token = page_start_token + offs_n
        valid_n = valid_cache_offset & (logical_token < seq_len)
        kv_mask = valid_n[:, None] & d_mask[None, :]

        k_int = tl.load(
            key_cache_ptr
            + physical_block * stride_kcb
            + offs_n[:, None] * stride_kcs
            + kv_head_id * stride_kch
            + offs_d[None, :] * stride_kcd,
            mask=kv_mask,
            other=0,
        ).to(tl.float32)
        scores = tl.sum(k_int * q_for_dot[None, :], axis=1)
        if QUANTIZE_Q:
            scores = scores * q_scale * k_scale * SM_SCALE
        else:
            scores = scores * k_scale * SM_SCALE
        scores = tl.where(valid_n, scores, -float("inf"))

        tile_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, tile_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.where(valid_n, tl.exp(scores - m_new), 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=0)

        v_int = tl.load(
            value_cache_ptr
            + physical_block * stride_vcb
            + offs_n[:, None] * stride_vcs
            + kv_head_id * stride_vch
            + offs_d[None, :] * stride_vcd,
            mask=kv_mask,
            other=0,
        ).to(tl.float32)
        pv = tl.sum(p[:, None] * v_int, axis=0) * v_scale
        acc = acc * alpha + pv
        m_i = m_new
        l_i = l_new

    tl.store(
        output_ptr + batch_id * stride_ob + query_head_id * stride_oh + offs_d * stride_od,
        acc / tl.maximum(l_i, 1e-20),
        mask=d_mask,
    )


def _choose_num_splits(batch_size: int, num_kv_heads: int, max_num_blocks: int) -> int:
    programs = batch_size * num_kv_heads
    target = 128
    if programs >= target or max_num_blocks < 8:
        return 1
    ideal = (target + programs - 1) // programs
    max_by_blocks = max(1, max_num_blocks // 4)
    return int(min(max(ideal, 1), max_by_blocks, 16))


def _launch_v3(
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    k_scale,
    v_scale,
    output,
    *,
    quantize_q: bool,
    num_warps: int,
    max_num_blocks: int,
    Hq: int,
    Hkv: int,
    D: int,
    cache_block_size: int,
    sm_scale: float,
):
    B = query.shape[0]
    _int8_paged_attention_kernel_v3[(B, Hq)](
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
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
        max_num_blocks,
        NUM_Q_HEADS=Hq,
        NUM_KV_HEADS=Hkv,
        HEAD_DIM=D,
        HEAD_DIM_PAD=triton.next_power_of_2(D),
        CACHE_BLOCK_SIZE=cache_block_size,
        CACHE_BLOCK_SIZE_PAD=triton.next_power_of_2(cache_block_size),
        SM_SCALE=sm_scale,
        QUANTIZE_Q=quantize_q,
        num_warps=num_warps,
        num_stages=2,
    )


def _launch_v4(
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    k_scale,
    v_scale,
    output,
    *,
    quantize_q: bool,
    max_num_blocks: int,
    Hq: int,
    Hkv: int,
    D: int,
    cache_block_size: int,
    sm_scale: float,
    num_splits: int | None,
):
    B = query.shape[0]
    group_size = Hq // Hkv
    group_pad = triton.next_power_of_2(group_size)
    head_dim_pad = triton.next_power_of_2(D)

    if num_splits is None:
        num_splits = _choose_num_splits(B, Hkv, max_num_blocks)
    num_splits = max(1, int(num_splits))
    use_split = num_splits > 1

    if use_split:
        partial_out = torch.empty((B, Hq, num_splits, D), device=query.device, dtype=torch.float32)
        partial_m = torch.empty((B, Hq, num_splits), device=query.device, dtype=torch.float32)
        partial_l = torch.empty((B, Hq, num_splits), device=query.device, dtype=torch.float32)
        spo, spm, spl = partial_out.stride(), partial_m.stride(), partial_l.stride()
    else:
        partial_out = torch.empty(1, device=query.device, dtype=torch.float32)
        partial_m = torch.empty(1, device=query.device, dtype=torch.float32)
        partial_l = torch.empty(1, device=query.device, dtype=torch.float32)
        spo, spm, spl = (0, 0, 0, 0), (0, 0, 0), (0, 0, 0)

    grid = (B, Hkv, num_splits)
    _int8_paged_attention_gqa_kernel[grid](
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        output,
        partial_out,
        partial_m,
        partial_l,
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
        spo[0],
        spo[1],
        spo[2],
        spo[3] if use_split else 0,
        spm[0],
        spm[1],
        spm[2] if use_split else 0,
        spl[0],
        spl[1],
        spl[2] if use_split else 0,
        max_num_blocks,
        NUM_Q_HEADS=Hq,
        NUM_KV_HEADS=Hkv,
        GROUP_SIZE=group_size,
        GROUP_PAD=group_pad,
        HEAD_DIM=D,
        HEAD_DIM_PAD=head_dim_pad,
        CACHE_BLOCK_SIZE=cache_block_size,
        SM_SCALE=sm_scale,
        QUANTIZE_Q=quantize_q,
        NUM_SPLITS=num_splits,
        USE_SPLIT_KV=use_split,
    )

    if use_split:
        _merge_split_kv_kernel[(B, Hq)](
            partial_out,
            partial_m,
            partial_l,
            output,
            partial_out.stride(0),
            partial_out.stride(1),
            partial_out.stride(2),
            partial_out.stride(3),
            partial_m.stride(0),
            partial_m.stride(1),
            partial_m.stride(2),
            partial_l.stride(0),
            partial_l.stride(1),
            partial_l.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            NUM_SPLITS=num_splits,
            HEAD_DIM=D,
            HEAD_DIM_PAD=head_dim_pad,
            num_warps=4,
            num_stages=2,
        )


def int8_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    quantize_q: bool = False,
    num_warps: int = 4,
    impl: str = "v4",
    num_splits: int | None = None,
):
    """
    Int8 Paged Decode Attention.

    impl:
      "v4" — GQA KV reuse + tl.dot + autotune (+ auto split-KV)
      "v3" — legacy per-query-head baseline
    """
    assert query.is_cuda
    assert key_cache.is_cuda and value_cache.is_cuda
    assert block_tables.is_cuda and seq_lens.is_cuda
    assert k_scale.is_cuda and v_scale.is_cuda
    assert query.ndim == 3 and key_cache.ndim == 4 and value_cache.ndim == 4
    assert block_tables.ndim == 2 and seq_lens.ndim == 1
    assert query.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert key_cache.dtype == torch.int8 and value_cache.dtype == torch.int8
    assert block_tables.dtype in (torch.int32, torch.int64)
    assert seq_lens.dtype in (torch.int32, torch.int64)

    B, Hq, D = query.shape
    _, cache_block_size, Hkv, cache_D = key_cache.shape
    assert value_cache.shape == key_cache.shape
    assert cache_D == D and Hq % Hkv == 0
    assert block_tables.shape[0] == B and seq_lens.numel() == B
    assert k_scale.ndim == 1 and v_scale.ndim == 1
    assert k_scale.numel() == Hkv and v_scale.numel() == Hkv

    if k_scale.dtype != torch.float32:
        k_scale = k_scale.float().contiguous()
    if v_scale.dtype != torch.float32:
        v_scale = v_scale.float().contiguous()

    assert key_cache.stride(-1) == 1 and value_cache.stride(-1) == 1
    assert int(seq_lens.min().item()) > 0

    max_seq_len = int(seq_lens.max().item())
    required_max_blocks = (max_seq_len + cache_block_size - 1) // cache_block_size
    assert block_tables.shape[1] >= required_max_blocks

    output = torch.empty((B, Hq, D), device=query.device, dtype=query.dtype)
    sm_scale = 1.0 / math.sqrt(D)

    if impl == "v3":
        _launch_v3(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            k_scale,
            v_scale,
            output,
            quantize_q=quantize_q,
            num_warps=num_warps,
            max_num_blocks=required_max_blocks,
            Hq=Hq,
            Hkv=Hkv,
            D=D,
            cache_block_size=cache_block_size,
            sm_scale=sm_scale,
        )
    elif impl == "v4":
        _launch_v4(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            k_scale,
            v_scale,
            output,
            quantize_q=quantize_q,
            max_num_blocks=required_max_blocks,
            Hq=Hq,
            Hkv=Hkv,
            D=D,
            cache_block_size=cache_block_size,
            sm_scale=sm_scale,
            num_splits=num_splits,
        )
    else:
        raise ValueError(f"Unknown impl={impl!r}; expected 'v3' or 'v4'")

    return output


def _round_half_away_from_zero_torch(x: torch.Tensor):
    return torch.where(x >= 0, torch.floor(x + 0.5), -torch.floor(-x + 0.5))


def torch_int8_paged_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    quantize_q: bool = False,
):
    B, Hq, D = query.shape
    (_, block_size, Hkv, _) = key_cache.shape
    assert Hq % Hkv == 0
    q_per_kv = Hq // Hkv
    outputs = []

    for b in range(B):
        T = int(seq_lens[b].item())
        batch_outputs = []
        for qh in range(Hq):
            kvh = qh // q_per_kv
            q = query[b, qh].float()
            if quantize_q:
                q_amax = q.abs().max()
                q_scale = torch.clamp(q_amax / 127.0, min=1e-6)
                q8 = torch.clamp(_round_half_away_from_zero_torch(q / q_scale), -127, 127)
                q = q8 * q_scale

            k_list, v_list = [], []
            for lb in range((T + block_size - 1) // block_size):
                physical_block = int(block_tables[b, lb].item())
                k_list.append(key_cache[physical_block, :, kvh, :].float())
                v_list.append(value_cache[physical_block, :, kvh, :].float())

            k = torch.cat(k_list, dim=0)[:T] * k_scale[kvh].float()
            v = torch.cat(v_list, dim=0)[:T] * v_scale[kvh].float()
            score = torch.sum(k * q[None, :], dim=-1) / math.sqrt(D)
            p = torch.softmax(score, dim=-1)
            batch_outputs.append(torch.sum(p[:, None] * v, dim=0))
        outputs.append(torch.stack(batch_outputs, dim=0))

    return torch.stack(outputs, dim=0).to(query.dtype)
