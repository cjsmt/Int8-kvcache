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
    """
    Triton kernel:
        BF16 / FP16 K,V
            ↓
        static per-head INT8 quant
            ↓
        paged KV cache write

    一个 Triton program 负责：
        一个 token
        一个 KV head

    grid:
        (num_tokens, num_kv_heads)

    Input:
        key/value:
            [num_tokens, num_kv_heads, head_dim]

        Cache:
            [num_blocks, block_size, num_kv_heads, head_dim]

        Scale:
            [num_kv_heads]
    """

    # ---------------------------------------------------------
    # 1. 当前 program 负责哪个 token / KV head
    # ---------------------------------------------------------

    token_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    # ---------------------------------------------------------
    # 2. 读取当前 token 对应的 physical slot
    #
    # slot 的含义：
    #
    #     slot =
    #         physical_block * BLOCK_SIZE
    #         + offset_in_block
    #
    # 例如：
    #
    #     BLOCK_SIZE = 16
    #     slot = 37
    #
    # 则：
    #
    #     physical_block = 2
    #     offset_in_block = 5
    #
    # 最终写入：
    #
    #     cache[2, 5, head, :]
    # ---------------------------------------------------------

    slot = tl.load(slot_mapping_ptr + token_id).to(tl.int64)

    # ---------------------------------------------------------
    # 3. slot < 0 时不写 cache
    #
    # 真实框架里经常用负 slot 表示：
    # padding / ignored token
    #
    # Triton 中尽量不用 Python 风格 return 控制流，
    # 所以后面通过 valid_slot 做 mask。
    # ---------------------------------------------------------

    valid_slot = slot >= 0

    safe_slot = tl.where(valid_slot, slot, 0)

    physical_block = safe_slot // BLOCK_SIZE

    offset_in_block = safe_slot % BLOCK_SIZE

    # ---------------------------------------------------------
    # 4. 读取 static per-head scale
    #
    # k_scale:
    #     [Hkv]
    #
    # v_scale:
    #     [Hkv]
    # ---------------------------------------------------------

    k_scale = tl.load(k_scale_ptr + head_id).to(tl.float32)

    v_scale = tl.load(v_scale_ptr + head_id).to(tl.float32)

    # 避免极端情况下 scale == 0
    k_scale = tl.maximum(k_scale, 1e-6)

    v_scale = tl.maximum(v_scale, 1e-6)

    # ---------------------------------------------------------
    # 5. 一个 program 读取当前 head 的整个 head_dim
    #
    # Qwen2.5-7B:
    #
    #     HEAD_DIM = 128
    #
    # 如果以后 HEAD_DIM 不是 2 的幂，
    # HEAD_DIM_PAD 会自动 pad。
    # ---------------------------------------------------------

    offs_d = tl.arange(0, HEAD_DIM_PAD)

    valid_d = offs_d < HEAD_DIM

    # input key:
    #
    # [token, head, dim]

    key_offsets = token_id * stride_kt + head_id * stride_kh + offs_d * stride_kd

    value_offsets = token_id * stride_vt + head_id * stride_vh + offs_d * stride_vd

    key = tl.load(key_ptr + key_offsets, mask=valid_d, other=0.0).to(tl.float32)

    value = tl.load(value_ptr + value_offsets, mask=valid_d, other=0.0).to(tl.float32)

    # ---------------------------------------------------------
    # 6. Static symmetric INT8 quantization
    #
    #     q = round(x / scale)
    #
    #     q = clamp(q, -127, 127)
    #
    # 为什么不是 -128：
    #
    #     为了保持严格对称量化，
    #     使用 [-127, 127]
    # ---------------------------------------------------------

    key_scaled = key / k_scale

    value_scaled = value / v_scale

    # ---------------------------------------------------------
    # 7. Round
    #
    # 不直接依赖 Python round。
    #
    # 当前使用：
    #
    # x >= 0:
    #     floor(x + 0.5)
    #
    # x < 0:
    #     ceil(x - 0.5)
    #
    # 相当于 round-half-away-from-zero。
    #
    # 和 torch.round 的 tie-to-even 在刚好 *.5 时可能
    # 有 1 个整数级差异，所以单测允许 diff <= 1。
    # ---------------------------------------------------------

    key_rounded = tl.where(key_scaled >= 0, tl.floor(key_scaled + 0.5), tl.ceil(key_scaled - 0.5))

    value_rounded = tl.where(value_scaled >= 0, tl.floor(value_scaled + 0.5), tl.ceil(value_scaled - 0.5))

    # ---------------------------------------------------------
    # 8. Clamp
    # ---------------------------------------------------------

    key_clamped = tl.maximum(tl.minimum(key_rounded, 127.0), -127.0)

    value_clamped = tl.maximum(tl.minimum(value_rounded, 127.0), -127.0)

    # ---------------------------------------------------------
    # 9. Cast INT8
    # ---------------------------------------------------------

    key_int8 = key_clamped.to(tl.int8)

    value_int8 = value_clamped.to(tl.int8)

    # ---------------------------------------------------------
    # 10. 计算 Paged Cache 地址
    #
    # cache layout:
    #
    # [physical_block,
    #  slot_in_block,
    #  kv_head,
    #  head_dim]
    #
    # 注意这里完全使用 tensor stride，
    # 所以不要求 cache 必须是某个手写连续 layout。
    # ---------------------------------------------------------

    key_cache_offsets = physical_block * stride_kcb + offset_in_block * stride_kcs + head_id * stride_kch + offs_d * stride_kcd

    value_cache_offsets = physical_block * stride_vcb + offset_in_block * stride_vcs + head_id * stride_vch + offs_d * stride_vcd

    # ---------------------------------------------------------
    # 11. 最终写 cache
    #
    # valid_slot=False 时不写。
    # ---------------------------------------------------------

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
    """
    将 BF16 / FP16 K,V 使用 Static Per-Head Scale
    量化为 INT8，并直接写入 Paged KV Cache。

    Parameters
    ----------
    key:
        shape:
            [num_tokens, num_kv_heads, head_dim]

        dtype:
            torch.float16 / torch.bfloat16 / torch.float32

    value:
        与 key shape 相同。

    key_cache:
        shape:
            [num_blocks,
             block_size,
             num_kv_heads,
             head_dim]

        dtype:
            torch.int8

    value_cache:
        与 key_cache shape 相同。

    slot_mapping:
        shape:
            [num_tokens]

        每个 token 对应一个 physical slot：

            slot =
                physical_block * block_size
                + offset_in_block

        slot < 0:
            不写入 cache。

    k_scale:
        shape:
            [num_kv_heads]

        Static Per-Head K Scale。

    v_scale:
        shape:
            [num_kv_heads]

        Static Per-Head V Scale。

    Returns
    -------
    None

    直接原地修改：

        key_cache
        value_cache
    """

    # ---------------------------------------------------------
    # 1. 基础检查
    # ---------------------------------------------------------

    assert key.is_cuda
    assert value.is_cuda

    assert key_cache.is_cuda
    assert value_cache.is_cuda

    assert slot_mapping.is_cuda

    assert k_scale.is_cuda
    assert v_scale.is_cuda

    assert key.ndim == 3
    assert value.ndim == 3

    assert key_cache.ndim == 4
    assert value_cache.ndim == 4

    assert key.shape == value.shape

    assert key_cache.shape == value_cache.shape

    assert key_cache.dtype == torch.int8

    assert value_cache.dtype == torch.int8

    assert key.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )

    assert value.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )

    # ---------------------------------------------------------
    # 2. Shape
    # ---------------------------------------------------------

    (
        num_tokens,
        num_kv_heads,
        head_dim,
    ) = key.shape

    (
        num_blocks,
        block_size,
        cache_num_kv_heads,
        cache_head_dim,
    ) = key_cache.shape

    assert cache_num_kv_heads == num_kv_heads

    assert cache_head_dim == head_dim

    assert slot_mapping.ndim == 1

    assert slot_mapping.numel() == num_tokens

    assert k_scale.ndim == 1
    assert v_scale.ndim == 1

    assert k_scale.numel() == num_kv_heads

    assert v_scale.numel() == num_kv_heads

    # ---------------------------------------------------------
    # 3. Scale dtype
    #
    # 推荐 float32。
    # ---------------------------------------------------------

    if k_scale.dtype != torch.float32:
        k_scale = k_scale.float().contiguous()

    if v_scale.dtype != torch.float32:
        v_scale = v_scale.float().contiguous()

    # ---------------------------------------------------------
    # 4. slot_mapping dtype
    #
    # int32 / int64 都可以。
    # ---------------------------------------------------------

    assert slot_mapping.dtype in (
        torch.int32,
        torch.int64,
    )

    # ---------------------------------------------------------
    # 5. Head Dim Pad
    #
    # Triton tl.arange 要求 block size 通常是 2 的幂。
    #
    # Qwen D=128：
    #
    # HEAD_DIM_PAD = 128
    # ---------------------------------------------------------

    head_dim_pad = triton.next_power_of_2(head_dim)

    # ---------------------------------------------------------
    # 6. Grid
    #
    # 一个 program：
    #
    #     一个 token
    #     一个 kv head
    #
    # 所以：
    #
    #     num_programs =
    #         num_tokens * num_kv_heads
    # ---------------------------------------------------------

    grid = (
        num_tokens,
        num_kv_heads,
    )

    # ---------------------------------------------------------
    # 7. Launch
    # ---------------------------------------------------------

    _int8_kv_cache_write_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
        # key strides
        key.stride(0),
        key.stride(1),
        key.stride(2),
        # value strides
        value.stride(0),
        value.stride(1),
        value.stride(2),
        # key cache strides
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        # value cache strides
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
    """
    PyTorch correctness reference。

    这个函数故意写得简单，
    不考虑性能。

    用途：

        Triton kernel correctness test。
    """

    assert key.ndim == 3

    (
        num_tokens,
        num_kv_heads,
        head_dim,
    ) = key.shape

    block_size = key_cache.shape[1]

    for token_id in range(num_tokens):
        slot = int(slot_mapping[token_id].item())

        if slot < 0:
            continue

        physical_block = slot // block_size

        offset_in_block = slot % block_size

        for head_id in range(num_kv_heads):
            # -------------------------
            # K
            # -------------------------

            key_int8 = torch.round(
                key[
                    token_id,
                    head_id,
                ].float()
                / k_scale[head_id].float()
            )

            key_int8 = torch.clamp(
                key_int8,
                -127,
                127,
            ).to(torch.int8)

            # -------------------------
            # V
            # -------------------------

            value_int8 = torch.round(
                value[
                    token_id,
                    head_id,
                ].float()
                / v_scale[head_id].float()
            )

            value_int8 = torch.clamp(
                value_int8,
                -127,
                127,
            ).to(torch.int8)

            # -------------------------
            # write paged cache
            # -------------------------

            key_cache[physical_block, offset_in_block, head_id, :] = key_int8

            value_cache[physical_block, offset_in_block, head_id, :] = value_int8


def dequantize_paged_cache_for_debug(
    cache: torch.Tensor,
    scale: torch.Tensor,
):
    """
    Debug 辅助函数。

    cache:
        [num_blocks,
         block_size,
         Hkv,
         D]

    scale:
        [Hkv]

    return:
        float32 cache
    """

    assert cache.dtype == torch.int8

    return cache.float() * scale[None, None, :, None].float()
