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


def create_random_block_tables(
    batch_size,
    seq_lens,
    block_size,
    num_physical_blocks,
    device,
):
    """Random logical->physical block map (non-contiguous) for paged addressing tests."""
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


def build_slot_mapping(
    block_tables,
    seq_lens,
    block_size,
):
    """Token -> physical slot = physical_block * block_size + offset (runtime write format)."""
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


def quantize_static_per_head(
    x,
    scale,
):
    """Static per-head INT8: x [T,Hkv,D], scale [Hkv]."""
    q = x.float() / scale[None, :, None]
    q = torch.round(q)
    q = torch.clamp(
        q,
        -127,
        127,
    )
    return q.to(torch.int8)


def torch_attention_reference(
    query,
    keys,
    values,
):
    """Decode ref: query [B,Hq,D], keys/values list of [T,Hkv,D], GQA repeat_interleave."""
    batch_size, num_q_heads, head_dim = query.shape
    num_kv_heads = keys[0].shape[1]
    assert num_q_heads % num_kv_heads == 0
    group_size = num_q_heads // num_kv_heads
    outputs = []
    for batch_id in range(batch_size):
        q = query[batch_id].float()
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
        k = k.repeat_interleave(
            group_size,
            dim=0,
        )
        v = v.repeat_interleave(
            group_size,
            dim=0,
        )
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


def gather_dequantized_paged_cache(
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    k_scale,
    v_scale,
):
    """Gather paged INT8 by logical seq order -> dequant K/V lists (reference only)."""
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
            k_int8 = key_cache[
                physical_block,
                offset,
            ].float()
            v_int8 = value_cache[
                physical_block,
                offset,
            ].float()
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


def test_runtime_write_and_attention():
    random.seed(0)
    torch.manual_seed(0)
    device = "cuda"
    B = 2
    Hq = 28
    Hkv = 4
    D = 128
    block_size = 16
    seq_lens_list = [
        127,
        301,
    ]
    query = torch.randn(
        B,
        Hq,
        D,
        device=device,
        dtype=torch.bfloat16,
    )
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
    required_blocks = sum(
        (seq_len + block_size - 1) // block_size for seq_len in seq_lens_list
    )
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
    slot_mapping = build_slot_mapping(
        block_tables,
        seq_lens_list,
        block_size,
    )
    print()
    print("slot_mapping shape:", slot_mapping.shape)
    print("slot first 20:", slot_mapping[:20])
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
    key_cache = torch.zeros(
        num_physical_blocks,
        block_size,
        Hkv,
        D,
        device=device,
        dtype=torch.int8,
    )
    value_cache = torch.zeros_like(key_cache)
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
    assert max(cache_write_k_errors) < 0.02
    assert max(cache_write_v_errors) < 0.02
    int8_math_reference = torch_attention_reference(
        query,
        cache_keys_deq,
        cache_values_deq,
    )
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
    bf16_reference = torch_attention_reference(
        query,
        keys,
        values,
    )
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
    assert kernel_cos > 0.999, "PagedAttention kernel cosine too low"
    assert kernel_rel_l2 < 0.02, "PagedAttention kernel relative L2 too high"
    assert quant_cos > 0.98, "INT8 quantization cosine too low"
    assert not torch.isnan(int8_triton_output).any()
    assert not torch.isinf(int8_triton_output).any()
    print()
    print("=" * 80)
    print("STEP 2 PASS")
    print("=" * 80)
if __name__ == "__main__":
    test_runtime_write_and_attention()
