import math
import random

import torch
import torch.nn.functional as F


from vllm_int8.vllm_int8_cache_ops import (
    runtime_int8_cache_write,
)

from src.triton_ops.int8_paged_attention import (
    int8_paged_attention,
)


# ============================================================
# 1. 创建随机 Block Table
# ============================================================


def create_random_block_tables(
    batch_size,
    seq_lens,
    block_size,
    num_physical_blocks,
    device,
):
    """
    创建真正随机的 logical block -> physical block 映射。

    例如：

        logical blocks:
            0, 1, 2, 3

        physical blocks:
            7, 2, 11, 5

    这样可以真正验证 PagedAttention 的地址寻址，
    避免连续 [0,1,2,3] 映射掩盖 bug。
    """

    max_num_logical_blocks = max(
        (seq_len + block_size - 1) // block_size for seq_len in seq_lens
    )

    block_tables = torch.zeros(
        batch_size,
        max_num_logical_blocks,
        device=device,
        dtype=torch.int32,
    )

    available_blocks = list(range(num_physical_blocks))

    random.shuffle(available_blocks)

    ptr = 0

    for batch_id, seq_len in enumerate(seq_lens):
        needed_blocks = (seq_len + block_size - 1) // block_size

        selected = available_blocks[ptr : ptr + needed_blocks]

        assert len(selected) == needed_blocks

        ptr += needed_blocks

        block_tables[batch_id, :needed_blocks] = torch.tensor(
            selected,
            device=device,
            dtype=torch.int32,
        )

    return block_tables


# ============================================================
# 2. 根据 Block Table 构造 Runtime slot_mapping
# ============================================================


def build_slot_mapping(
    block_tables,
    seq_lens,
    block_size,
):
    """
    将：

        batch + logical token

    转成：

        physical slot

    slot 公式：

        slot =
            physical_block * block_size
            + offset_in_block

    这是 runtime_int8_cache_write 所需要的格式。
    """

    device = block_tables.device

    slots = []

    for batch_id, seq_len in enumerate(seq_lens):
        for token_id in range(seq_len):
            logical_block = token_id // block_size

            offset_in_block = token_id % block_size

            physical_block = int(block_tables[batch_id, logical_block].item())

            physical_slot = physical_block * block_size + offset_in_block

            slots.append(physical_slot)

    return torch.tensor(
        slots,
        device=device,
        dtype=torch.int64,
    )


# ============================================================
# 3. Static Per-Head INT8 Quant Reference
# ============================================================


def quantize_static_per_head(
    x,
    scale,
):
    """
    x:
        [T, Hkv, D]

    scale:
        [Hkv]

    返回：
        INT8 tensor
    """

    q = x.float() / scale[None, :, None]

    q = torch.round(q)

    q = torch.clamp(
        q,
        -127,
        127,
    )

    return q.to(torch.int8)


# ============================================================
# 4. BF16 / FP32 Attention Reference
# ============================================================


def torch_attention_reference(
    query,
    keys,
    values,
):
    """
    Decode Attention Reference。

    query:
        [B, Hq, D]

    keys:
        list of [T, Hkv, D]

    values:
        list of [T, Hkv, D]

    Qwen2.5-7B:
        Hq  = 28
        Hkv = 4

    GQA:
        每 7 个 Q head 共用一个 KV head。
    """

    batch_size, num_q_heads, head_dim = query.shape

    num_kv_heads = keys[0].shape[1]

    assert num_q_heads % num_kv_heads == 0

    group_size = num_q_heads // num_kv_heads

    outputs = []

    for batch_id in range(batch_size):
        # query:
        # [Hq, D]
        q = query[batch_id].float()

        # key:
        # [T, Hkv, D]
        #
        # ->
        #
        # [Hkv, T, D]

        k = (
            keys[batch_id]
            .float()
            .permute(
                1,
                0,
                2,
            )
        )

        v = (
            values[batch_id]
            .float()
            .permute(
                1,
                0,
                2,
            )
        )

        # ----------------------------------------------------
        # GQA expand:
        #
        # [Hkv,T,D]
        #
        # ->
        #
        # [Hq,T,D]
        # ----------------------------------------------------

        k = k.repeat_interleave(
            group_size,
            dim=0,
        )

        v = v.repeat_interleave(
            group_size,
            dim=0,
        )

        # ----------------------------------------------------
        # QK
        #
        # q:
        # [Hq,D]
        #
        # k:
        # [Hq,T,D]
        #
        # score:
        # [Hq,T]
        # ----------------------------------------------------

        scores = torch.einsum(
            "hd,htd->ht",
            q,
            k,
        )

        scores = scores / math.sqrt(head_dim)

        probs = torch.softmax(
            scores,
            dim=-1,
        )

        # ----------------------------------------------------
        # PV
        #
        # probs:
        # [Hq,T]
        #
        # v:
        # [Hq,T,D]
        #
        # output:
        # [Hq,D]
        # ----------------------------------------------------

        output = torch.einsum(
            "ht,htd->hd",
            probs,
            v,
        )

        outputs.append(output)

    return torch.stack(
        outputs,
        dim=0,
    )


# ============================================================
# 5. 从真实 INT8 Paged Cache gather 回连续 K/V
#
# 仅用于 Reference，不用于实际 Kernel。
# ============================================================


def gather_dequantized_paged_cache(
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    k_scale,
    v_scale,
):
    """
    将 Paged INT8 Cache 按 logical sequence 顺序 gather 回来。

    返回：

        keys:
            list [T,Hkv,D]

        values:
            list [T,Hkv,D]

    这是测试 reference，用来验证：
        Cache Write + Paged Address 是否正确。
    """

    block_size = key_cache.shape[1]

    keys = []
    values = []

    for batch_id, seq_len in enumerate(seq_lens):
        k_tokens = []
        v_tokens = []

        for token_id in range(seq_len):
            logical_block = token_id // block_size

            offset = token_id % block_size

            physical_block = int(block_tables[batch_id, logical_block].item())

            # --------------------------------------------
            # [Hkv,D]
            # --------------------------------------------

            k_int8 = key_cache[
                physical_block,
                offset,
            ].float()

            v_int8 = value_cache[
                physical_block,
                offset,
            ].float()

            # Static Per-Head dequant

            k_fp = k_int8 * k_scale[:, None]

            v_fp = v_int8 * v_scale[:, None]

            k_tokens.append(k_fp)

            v_tokens.append(v_fp)

        keys.append(
            torch.stack(
                k_tokens,
                dim=0,
            )
        )

        values.append(
            torch.stack(
                v_tokens,
                dim=0,
            )
        )

    return (
        keys,
        values,
    )


# ============================================================
# 6. Metric helpers
# ============================================================


def relative_l2(
    x,
    y,
):
    return ((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12)).item()


def cosine_similarity_flat(
    x,
    y,
):
    return F.cosine_similarity(
        x.float().reshape(-1),
        y.float().reshape(-1),
        dim=0,
    ).item()


def mean_absolute_error(
    x,
    y,
):
    return (x.float() - y.float()).abs().mean().item()


# ============================================================
# 7. 主测试
# ============================================================


def test_runtime_write_and_attention():

    # --------------------------------------------------------
    # 固定随机数
    # --------------------------------------------------------

    random.seed(0)
    torch.manual_seed(0)

    device = "cuda"

    # --------------------------------------------------------
    # Qwen2.5-7B attention config
    # --------------------------------------------------------

    B = 2

    Hq = 28
    Hkv = 4

    D = 128

    block_size = 16

    # --------------------------------------------------------
    # 故意使用非 16 整除长度
    #
    # 这样最后一个 page 一定是 partial page。
    # --------------------------------------------------------

    seq_lens_list = [
        127,
        301,
    ]

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    query = torch.randn(
        B,
        Hq,
        D,
        device=device,
        dtype=torch.bfloat16,
    )

    # --------------------------------------------------------
    # 每个 request 独立 K/V
    # --------------------------------------------------------

    keys = []
    values = []

    for seq_len in seq_lens_list:
        k = torch.randn(
            seq_len,
            Hkv,
            D,
            device=device,
            dtype=torch.bfloat16,
        )

        v = torch.randn_like(k)

        keys.append(k)
        values.append(v)

    # ========================================================
    # 8. 构造 Static Per-Head Scale
    #
    # 实际项目中这个 scale 来自 calibration。
    #
    # 单测中使用当前测试数据 absmax，
    # 只为了隔离 Kernel correctness。
    # ========================================================

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

    print()
    print("=" * 80)
    print("Static Per-Head Scale")
    print("=" * 80)

    print("K scale:", k_scale)

    print("V scale:", v_scale)

    # ========================================================
    # 9. 创建随机 Physical Page Layout
    # ========================================================

    required_blocks = sum(
        (seq_len + block_size - 1) // block_size for seq_len in seq_lens_list
    )

    # 故意多申请一些 physical blocks
    num_physical_blocks = required_blocks + 32

    block_tables = create_random_block_tables(
        B,
        seq_lens_list,
        block_size,
        num_physical_blocks,
        device,
    )

    print()
    print("=" * 80)
    print("Random Block Tables")
    print("=" * 80)

    print(block_tables)

    # ========================================================
    # 10. slot_mapping
    # ========================================================

    slot_mapping = build_slot_mapping(
        block_tables,
        seq_lens_list,
        block_size,
    )

    print()
    print("slot_mapping shape:", slot_mapping.shape)

    print("slot first 20:", slot_mapping[:20])

    # ========================================================
    # 11. Flatten runtime new K/V
    #
    # runtime cache write 接口：
    #
    # [num_tokens,Hkv,D]
    # ========================================================

    key_flat = torch.cat(
        keys,
        dim=0,
    ).contiguous()

    value_flat = torch.cat(
        values,
        dim=0,
    ).contiguous()

    total_tokens = key_flat.shape[0]

    print()
    print("total runtime tokens:", total_tokens)

    # ========================================================
    # 12. Allocate INT8 Paged Cache
    # ========================================================

    key_cache = torch.zeros(
        num_physical_blocks,
        block_size,
        Hkv,
        D,
        device=device,
        dtype=torch.int8,
    )

    value_cache = torch.zeros_like(key_cache)

    # ========================================================
    # STEP A
    #
    # Runtime INT8 Cache Write
    # ========================================================

    runtime_int8_cache_write(
        key_flat,
        value_flat,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
    )

    torch.cuda.synchronize()

    print()
    print("=" * 80)
    print("Runtime Cache Write")
    print("=" * 80)

    print("PASS: cache write kernel finished")

    # ========================================================
    # STEP B
    #
    # Gather INT8 Cache -> dequant
    #
    # 验证：
    #
    # runtime write 后的内容是否和
    # 直接 static quant/dequant 一致。
    # ========================================================

    (
        cache_keys_deq,
        cache_values_deq,
    ) = gather_dequantized_paged_cache(
        key_cache,
        value_cache,
        block_tables,
        seq_lens_list,
        k_scale,
        v_scale,
    )

    # 直接数学 quant/dequant

    direct_keys_deq = []
    direct_values_deq = []

    for k, v in zip(
        keys,
        values,
    ):
        k8 = quantize_static_per_head(
            k,
            k_scale,
        )

        v8 = quantize_static_per_head(
            v,
            v_scale,
        )

        kd = k8.float() * k_scale[None, :, None]

        vd = v8.float() * v_scale[None, :, None]

        direct_keys_deq.append(kd)

        direct_values_deq.append(vd)

    # --------------------------------------------------------
    # Cache Write 数值误差
    # --------------------------------------------------------

    cache_write_k_errors = []
    cache_write_v_errors = []

    for i in range(B):
        k_err = relative_l2(
            cache_keys_deq[i],
            direct_keys_deq[i],
        )

        v_err = relative_l2(
            cache_values_deq[i],
            direct_values_deq[i],
        )

        cache_write_k_errors.append(k_err)

        cache_write_v_errors.append(v_err)

    print()
    print("=" * 80)
    print("Cache Write -> Gather Correctness")
    print("=" * 80)

    print("K relative L2:", cache_write_k_errors)

    print("V relative L2:", cache_write_v_errors)

    # rounding 允许很小的误差
    assert max(cache_write_k_errors) < 0.02

    assert max(cache_write_v_errors) < 0.02

    # ========================================================
    # STEP C
    #
    # INT8 Mathematical Reference
    #
    # 注意：
    #
    # 这里使用 Cache 中真正写进去并 dequant 后的 K/V，
    # 而不是重新 quant 一份。
    #
    # 这样 reference 和 Triton kernel 消费的是同一份数据。
    # ========================================================

    int8_math_reference = torch_attention_reference(
        query,
        cache_keys_deq,
        cache_values_deq,
    )

    # ========================================================
    # STEP D
    #
    # Triton INT8 PagedAttention
    # ========================================================

    seq_lens_tensor = torch.tensor(
        seq_lens_list,
        device=device,
        dtype=torch.int32,
    )

    int8_triton_output = int8_paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens_tensor,
        k_scale,
        v_scale,
        quantize_q=False,
    )

    torch.cuda.synchronize()

    # ========================================================
    # STEP E
    #
    # Kernel Correctness
    #
    # Triton PagedAttention
    #
    # vs
    #
    # 同一份 INT8 cache mathematical reference
    # ========================================================

    kernel_cos = cosine_similarity_flat(
        int8_triton_output,
        int8_math_reference,
    )

    kernel_rel_l2 = relative_l2(
        int8_triton_output,
        int8_math_reference,
    )

    kernel_mae = mean_absolute_error(
        int8_triton_output,
        int8_math_reference,
    )

    print()
    print("=" * 80)
    print("INT8 Triton Kernel Correctness")
    print("=" * 80)

    print("cosine      :", kernel_cos)

    print("relative L2 :", kernel_rel_l2)

    print("MAE         :", kernel_mae)

    # ========================================================
    # STEP F
    #
    # BF16 Reference
    # ========================================================

    bf16_reference = torch_attention_reference(
        query,
        keys,
        values,
    )

    # ========================================================
    # STEP G
    #
    # Quantization Error
    #
    # INT8 mathematical reference
    #
    # vs
    #
    # original BF16 attention
    # ========================================================

    quant_cos = cosine_similarity_flat(
        int8_math_reference,
        bf16_reference,
    )

    quant_rel_l2 = relative_l2(
        int8_math_reference,
        bf16_reference,
    )

    quant_mae = mean_absolute_error(
        int8_math_reference,
        bf16_reference,
    )

    print()
    print("=" * 80)
    print("INT8 Quantization Error")
    print("=" * 80)

    print("cosine      :", quant_cos)

    print("relative L2 :", quant_rel_l2)

    print("MAE         :", quant_mae)

    # ========================================================
    # STEP H
    #
    # End-to-End:
    #
    # Triton INT8
    # vs
    # BF16 reference
    # ========================================================

    e2e_cos = cosine_similarity_flat(
        int8_triton_output,
        bf16_reference,
    )

    e2e_rel_l2 = relative_l2(
        int8_triton_output,
        bf16_reference,
    )

    print()
    print("=" * 80)
    print("End-to-End INT8 vs BF16")
    print("=" * 80)

    print("cosine      :", e2e_cos)

    print("relative L2 :", e2e_rel_l2)

    # ========================================================
    # Assertions
    # ========================================================

    # -------------------------
    # Kernel 本身
    # 应该和同一 INT8 数学高度一致
    # -------------------------

    assert kernel_cos > 0.999, "PagedAttention kernel cosine too low"

    assert kernel_rel_l2 < 0.02, "PagedAttention kernel relative L2 too high"

    # -------------------------
    # INT8 量化精度
    #
    # 当前随机数据先用较宽松阈值。
    # 后续真实 Qwen KV 再重新设。
    # -------------------------

    assert quant_cos > 0.98, "INT8 quantization cosine too low"

    # -------------------------
    # 最终结果必须有限
    # -------------------------

    assert not torch.isnan(int8_triton_output).any()

    assert not torch.isinf(int8_triton_output).any()

    print()
    print("=" * 80)
    print("STEP 2 PASS")
    print("=" * 80)


# ============================================================
# main
# ============================================================


if __name__ == "__main__":
    test_runtime_write_and_attention()
