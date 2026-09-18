import math
import random

import torch
import torch.nn.functional as F
import pytest


from src.triton_ops.int8_cache_write import (
    int8_kv_cache_write,
    torch_int8_kv_cache_write_reference,
)


from src.triton_ops.int8_paged_attention import (
    int8_paged_attention,
    torch_int8_paged_attention_reference,
)


# ============================================================
# 工具函数1：
# 创建随机 physical block table
# ============================================================


def create_random_block_tables(
    batch_size,
    seq_lens,
    block_size,
    num_blocks,
    device,
):
    """
    创建真正 PagedAttention block table。

    不连续映射：

    logical block

        0
        1
        2

    可能对应：

        13
        2
        7


    返回:

        block_tables

    shape:

        [B, max_blocks]
    """

    max_blocks = max([(x + block_size - 1) // block_size for x in seq_lens])

    block_tables = torch.zeros(
        batch_size,
        max_blocks,
        device=device,
        dtype=torch.int32,
    )

    available = list(range(num_blocks))

    random.shuffle(available)

    ptr = 0

    for b in range(batch_size):
        needed = (seq_lens[b] + block_size - 1) // block_size

        selected = available[ptr : ptr + needed]

        ptr += needed

        block_tables[b, :needed] = torch.tensor(
            selected,
            device=device,
            dtype=torch.int32,
        )

    return block_tables


# ============================================================
# 工具函数2：
# logical token -> physical slot
#
# 给 Cache Write 使用
# ============================================================


def build_slot_mapping(
    block_tables,
    seq_lens,
    block_size,
):
    """
    根据 block_table 生成 slot_mapping。


    例如：

    block_size=16


    logical token:

        token 20


    logical block:

        20//16=1


    offset:

        20%16=4


    block_table:

        block_tables[0][1]=7


    physical slot:

        7*16+4


    """

    device = block_tables.device

    slots = []

    batch_size = len(seq_lens)

    for b in range(batch_size):
        T = seq_lens[b]

        for token in range(T):
            logical_block = token // block_size

            offset = token % block_size

            physical_block = int(block_tables[b, logical_block].item())

            slot = physical_block * block_size + offset

            slots.append(slot)

    return torch.tensor(
        slots,
        device=device,
        dtype=torch.int64,
    )


# ============================================================
# 工具函数3：
# BF16 原始 Attention Reference
# ============================================================


def torch_bf16_attention_reference(
    query,
    keys,
    values,
):
    """
    原始 BF16 Attention。


    query:

        [B,Hq,D]


    keys:

        list

        每个:

        [T,Hkv,D]


    values:

        list

        [T,Hkv,D]


    """

    B, Hq, D = query.shape

    Hkv = keys[0].shape[1]

    group = Hq // Hkv

    outputs = []

    for b in range(B):
        q_batch = []

        q = query[b].float()

        k = keys[b].float().permute(1, 0, 2)

        v = values[b].float().permute(1, 0, 2)

        #
        # GQA:
        #
        # Hkv -> Hq
        #

        k = k.repeat_interleave(group, dim=0)

        v = v.repeat_interleave(group, dim=0)

        score = torch.einsum(
            "hd,htd->ht",
            q,
            k,
        )

        score /= math.sqrt(D)

        prob = torch.softmax(
            score,
            dim=-1,
        )

        out = torch.einsum(
            "ht,htd->hd",
            prob,
            v,
        )

        q_batch.append(out)

        outputs.append(torch.stack(q_batch))

    return torch.stack(outputs).to(query.dtype)


# ============================================================
# 主测试
# ============================================================


@pytest.mark.cuda
def test_int8_paged_attention_correctness():

    torch.manual_seed(0)

    random.seed(0)

    device = "cuda"

    # --------------------------------------------------------
    # 模拟 Qwen2.5-7B
    #
    # Hq=28
    # Hkv=4
    # D=128
    # --------------------------------------------------------

    B = 2

    Hq = 28

    Hkv = 4

    D = 128

    block_size = 16

    seq_lens_list = [
        127,
        301,
    ]

    # --------------------------------------------------------
    # 创建 Query
    # --------------------------------------------------------

    query = torch.randn(
        B,
        Hq,
        D,
        device=device,
        dtype=torch.bfloat16,
    )

    # --------------------------------------------------------
    # 创建真实 K/V
    #
    # 每个 batch 一个 sequence
    # --------------------------------------------------------

    keys = []

    values = []

    for T in seq_lens_list:
        keys.append(
            torch.randn(
                T,
                Hkv,
                D,
                device=device,
                dtype=torch.bfloat16,
            )
        )

        values.append(torch.randn_like(keys[-1]))

    # --------------------------------------------------------
    # Static Per-Head Scale
    #
    # 模拟 calibration 后得到
    #
    # shape:
    #
    # [Hkv]
    #
    # --------------------------------------------------------

    all_k = torch.cat(
        keys,
        dim=0,
    ).float()

    all_v = torch.cat(
        values,
        dim=0,
    ).float()

    k_scale = (all_k.abs().amax(dim=(0, 2)) / 127.0).clamp_min(1e-6)

    v_scale = (all_v.abs().amax(dim=(0, 2)) / 127.0).clamp_min(1e-6)

    # --------------------------------------------------------
    # 创建随机 Page Table
    # --------------------------------------------------------

    required_blocks = sum([(x + block_size - 1) // block_size for x in seq_lens_list])

    num_blocks = required_blocks + 20

    block_tables = create_random_block_tables(
        B,
        seq_lens_list,
        block_size,
        num_blocks,
        device,
    )

    seq_lens = torch.tensor(
        seq_lens_list,
        device=device,
        dtype=torch.int32,
    )

    # --------------------------------------------------------
    # 生成 slot mapping
    # --------------------------------------------------------

    slot_mapping = build_slot_mapping(
        block_tables,
        seq_lens_list,
        block_size,
    )

    # --------------------------------------------------------
    # 创建 INT8 Cache
    # --------------------------------------------------------

    key_cache_ref = torch.zeros(
        num_blocks,
        block_size,
        Hkv,
        D,
        device=device,
        dtype=torch.int8,
    )

    value_cache_ref = torch.zeros_like(key_cache_ref)

    key_cache_tri = torch.zeros_like(key_cache_ref)

    value_cache_tri = torch.zeros_like(key_cache_ref)

    # --------------------------------------------------------
    # Flatten K/V
    #
    # Cache Write 输入:
    #
    # [num_tokens,Hkv,D]
    #
    # --------------------------------------------------------

    key_flat = torch.cat(keys, dim=0)

    value_flat = torch.cat(values, dim=0)

    # ========================================================
    # Test 1:
    #
    # Cache Write
    # ========================================================

    torch_int8_kv_cache_write_reference(
        key_flat,
        value_flat,
        key_cache_ref,
        value_cache_ref,
        slot_mapping,
        k_scale,
        v_scale,
    )

    int8_kv_cache_write(
        key_flat,
        value_flat,
        key_cache_tri,
        value_cache_tri,
        slot_mapping,
        k_scale,
        v_scale,
    )

    torch.cuda.synchronize()

    diff_k = (key_cache_ref.to(torch.int16) - key_cache_tri.to(torch.int16)).abs().max()

    diff_v = (
        (value_cache_ref.to(torch.int16) - value_cache_tri.to(torch.int16)).abs().max()
    )

    print("\n===== Cache Write =====")

    print("K int diff:", diff_k.item())

    print("V int diff:", diff_v.item())

    assert diff_k <= 1

    assert diff_v <= 1

    # ========================================================
    # Test 2:
    #
    # Int8 Paged Attention
    # ========================================================

    out_tri = int8_paged_attention(
        query,
        key_cache_tri,
        value_cache_tri,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        quantize_q=False,
    )

    torch.cuda.synchronize()

    out_ref = int8_paged_attention(
        query,
        key_cache_ref,
        value_cache_ref,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        quantize_q=False,
    )

    torch.cuda.synchronize()

    kernel_diff = out_tri.float() - out_ref.float()

    kernel_rel_l2 = (kernel_diff.norm() / out_ref.norm()).item()

    kernel_cos = F.cosine_similarity(
        out_tri.reshape(-1).float(),
        out_ref.reshape(-1).float(),
        dim=0,
    ).item()

    print("\n===== Triton vs INT8 Reference =====")

    print("relative L2:", kernel_rel_l2)

    print("cosine:", kernel_cos)

    assert kernel_cos > 0.999

    assert kernel_rel_l2 < 0.02

    # ========================================================
    # Test 3:
    #
    # INT8 Attention
    #
    # vs
    #
    # BF16 Attention
    #
    # ========================================================

    bf16_out = torch_bf16_attention_reference(
        query,
        keys,
        values,
    )

    int8_out = torch_int8_paged_attention_reference(
        query,
        key_cache_tri,
        value_cache_tri,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    )

    quant_cos = F.cosine_similarity(
        int8_out.reshape(-1).float(),
        bf16_out.reshape(-1).float(),
        dim=0,
    ).item()

    quant_rel_l2 = (
        (int8_out.float() - bf16_out.float()).norm() / bf16_out.float().norm()
    ).item()

    print("\n===== Quantization Error =====")

    print("INT8 vs BF16 cosine:", quant_cos)

    print("INT8 vs BF16 relative L2:", quant_rel_l2)

    assert quant_cos > 0.98
