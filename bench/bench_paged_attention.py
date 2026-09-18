import math
import random
import torch

from src.triton_ops.int8_cache_write import int8_kv_cache_write
from src.triton_ops.int8_paged_attention import int8_paged_attention
from src.triton_ops.bf16_paged_attention import bf16_paged_attention


# ============================================================
# CUDA Timer
# ============================================================


def benchmark_cuda(
    func,
    warmup=20,
    repeat=100,
):

    for _ in range(warmup):
        func()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)

    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(repeat):
        func()

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeat


# ============================================================
# Build random paged layout
# ============================================================


def create_block_tables(
    batch_size,
    seq_len,
    block_size,
    num_blocks,
    device,
):

    blocks_per_seq = (seq_len + block_size - 1) // block_size

    table = torch.empty(
        batch_size,
        blocks_per_seq,
        device=device,
        dtype=torch.int32,
    )

    available = list(range(num_blocks))

    random.shuffle(available)

    ptr = 0

    for b in range(batch_size):
        selected = available[ptr : ptr + blocks_per_seq]

        table[b] = torch.tensor(
            selected,
            device=device,
            dtype=torch.int32,
        )

        ptr += blocks_per_seq

    return table


# ============================================================
# Create BF16 + INT8 Cache
# ============================================================


def build_cache_pair(
    batch_size,
    seq_len,
    hkv,
    head_dim,
    block_size,
    device,
):

    blocks_per_seq = (seq_len + block_size - 1) // block_size

    num_blocks = batch_size * blocks_per_seq * 2

    block_tables = create_block_tables(
        batch_size,
        seq_len,
        block_size,
        num_blocks,
        device,
    )

    # -------------------------
    # Original KV
    # -------------------------

    key = torch.randn(
        batch_size * seq_len,
        hkv,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )

    value = torch.randn_like(key)

    # -------------------------
    # scale calibration
    # -------------------------

    k_scale = (key.float().abs().amax(dim=(0, 2)) / 127).clamp_min(1e-6)

    v_scale = (value.float().abs().amax(dim=(0, 2)) / 127).clamp_min(1e-6)

    # -------------------------
    # BF16 cache
    # -------------------------

    bf16_k_cache = torch.zeros(
        num_blocks,
        block_size,
        hkv,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )

    bf16_v_cache = torch.zeros_like(bf16_k_cache)

    # -------------------------
    # INT8 cache
    # -------------------------

    int8_k_cache = torch.zeros(
        num_blocks,
        block_size,
        hkv,
        head_dim,
        device=device,
        dtype=torch.int8,
    )

    int8_v_cache = torch.zeros_like(int8_k_cache)

    # -------------------------
    # write cache
    # -------------------------

    slot_mapping = []

    for b in range(batch_size):
        for token in range(seq_len):
            logical_block = token // block_size

            offset = token % block_size

            physical_block = int(block_tables[b, logical_block])

            slot_mapping.append(physical_block * block_size + offset)

            bf16_k_cache[physical_block, offset] = key[b * seq_len + token]

            bf16_v_cache[physical_block, offset] = value[b * seq_len + token]

    slot_mapping = torch.tensor(
        slot_mapping,
        device=device,
        dtype=torch.int64,
    )

    int8_kv_cache_write(
        key,
        value,
        int8_k_cache,
        int8_v_cache,
        slot_mapping,
        k_scale,
        v_scale,
    )

    seq_lens = torch.full(
        (batch_size,),
        seq_len,
        device=device,
        dtype=torch.int32,
    )

    return (
        bf16_k_cache,
        bf16_v_cache,
        int8_k_cache,
        int8_v_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    )


# ============================================================
# Memory calculation
# ============================================================


def kv_memory_MB(
    seq_len,
    hkv,
    dim,
    dtype_bytes,
):

    return seq_len * hkv * dim * dtype_bytes / 1024 / 1024


# ============================================================
# Main benchmark
# ============================================================


def run_benchmark():

    device = "cuda"

    random.seed(0)

    torch.manual_seed(0)

    # Qwen2.5-7B config

    Hq = 28

    Hkv = 4

    D = 128

    block_size = 16

    configs = [
        (1, 512),
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (4, 2048),
        (8, 4096),
    ]

    print()

    print("=" * 90)

    print("INT8 KVCache PagedAttention Benchmark")

    print("=" * 90)

    print(
        f"{'Batch':>8}"
        f"{'Seq':>8}"
        f"{'BF16(ms)':>15}"
        f"{'INT8(ms)':>15}"
        f"{'Speedup':>12}"
        f"{'BF16 MB':>12}"
        f"{'INT8 MB':>12}"
    )

    for batch, seq_len in configs:
        print()

        query = torch.randn(batch, Hq, D, device=device, dtype=torch.bfloat16)

        (
            bf16_k,
            bf16_v,
            int8_k,
            int8_v,
            block_tables,
            seq_lens,
            k_scale,
            v_scale,
        ) = build_cache_pair(
            batch,
            seq_len,
            Hkv,
            D,
            block_size,
            device,
        )

        # -------------------------
        # BF16
        # -------------------------
        bf16_ms = benchmark_cuda(
            lambda: bf16_paged_attention(query, bf16_k, bf16_v, block_tables, seq_lens)
        )

        # -------------------------
        # INT8
        # -------------------------

        int8_ms = benchmark_cuda(
            lambda: int8_paged_attention(
                query, int8_k, int8_v, block_tables, seq_lens, k_scale, v_scale
            )
        )

        speedup = bf16_ms / int8_ms

        bf16_mem = kv_memory_MB(
            seq_len,
            Hkv,
            D,
            2,
        )

        int8_mem = kv_memory_MB(
            seq_len,
            Hkv,
            D,
            1,
        )

        print(
            f"{batch:8d}"
            f"{seq_len:8d}"
            f"{bf16_ms:15.4f}"
            f"{int8_ms:15.4f}"
            f"{speedup:12.3f}x"
            f"{bf16_mem:12.2f}"
            f"{int8_mem:12.2f}"
        )


if __name__ == "__main__":
    run_benchmark()
