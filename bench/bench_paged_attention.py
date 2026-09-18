"""V4 microbench: BF16 vs INT8-v3 vs INT8-v4 (+ optional quantize_q / split-KV). Writes timing + rough roofline hints to outputs/v4_baseline/."""
from __future__ import annotations
import csv
import json
import math
import os
import random
import time
from pathlib import Path
import torch
from src.triton_ops.bf16_paged_attention import bf16_paged_attention
from src.triton_ops.int8_cache_write import int8_kv_cache_write
from src.triton_ops.int8_paged_attention import int8_paged_attention
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "v4_baseline"


def benchmark_cuda(func, warmup=25, repeat=100):
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


def create_block_tables(batch_size, seq_len, block_size, num_blocks, device):
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    table = torch.empty(batch_size, blocks_per_seq, device=device, dtype=torch.int32)
    available = list(range(num_blocks))
    random.shuffle(available)
    ptr = 0
    for b in range(batch_size):
        selected = available[ptr : ptr + blocks_per_seq]
        table[b] = torch.tensor(selected, device=device, dtype=torch.int32)
        ptr += blocks_per_seq
    return table


def build_cache_pair(batch_size, seq_len, hkv, head_dim, block_size, device):
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_seq * 2
    block_tables = create_block_tables(batch_size, seq_len, block_size, num_blocks, device)
    key = torch.randn(batch_size * seq_len, hkv, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    k_scale = (key.float().abs().amax(dim=(0, 2)) / 127).clamp_min(1e-6)
    v_scale = (value.float().abs().amax(dim=(0, 2)) / 127).clamp_min(1e-6)
    bf16_k = torch.zeros(num_blocks, block_size, hkv, head_dim, device=device, dtype=torch.bfloat16)
    bf16_v = torch.zeros_like(bf16_k)
    int8_k = torch.zeros(num_blocks, block_size, hkv, head_dim, device=device, dtype=torch.int8)
    int8_v = torch.zeros_like(int8_k)
    slots = []
    for b in range(batch_size):
        for token in range(seq_len):
            logical_block = token // block_size
            offset = token % block_size
            physical_block = int(block_tables[b, logical_block])
            slots.append(physical_block * block_size + offset)
            bf16_k[physical_block, offset] = key[b * seq_len + token]
            bf16_v[physical_block, offset] = value[b * seq_len + token]
    slot_mapping = torch.tensor(slots, device=device, dtype=torch.int64)
    int8_kv_cache_write(key, value, int8_k, int8_v, slot_mapping, k_scale, v_scale)
    seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    return bf16_k, bf16_v, int8_k, int8_v, block_tables, seq_lens, k_scale, v_scale


def estimate_kv_bytes(batch, seq_len, hkv, dim, dtype_bytes, gqa_reload=1):
    """Bytes read for K+V once per attention call (per logical load factor)."""
    return batch * seq_len * hkv * dim * dtype_bytes * 2 * gqa_reload


def roofline_hint(ms, bytes_moved, peak_bw_GBs=1000.0):
    """Rough bound check vs ~1 TB/s HBM (4090-ish)."""
    if ms <= 0:
        return {"gbps": 0.0, "bound": "unknown"}
    gbps = (bytes_moved / 1e9) / (ms / 1e3)
    # If achieved BW close to peak → memory-bound; if far below → compute/latency-bound
    ratio = gbps / peak_bw_GBs
    if ratio > 0.35:
        bound = "memory-leaning"
    elif ratio < 0.10:
        bound = "compute-or-launch-leaning"
    else:
        bound = "mixed"
    return {"gbps": gbps, "bw_ratio_vs_1TBs": ratio, "bound": bound}


def run_benchmark(save=True):
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    random.seed(0)
    torch.manual_seed(0)
    Hq, Hkv, D, block_size = 28, 4, 128, 16
    configs = [
        (1, 512),
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (2, 2048),
        (4, 2048),
        (8, 2048),
        (8, 4096),
    ]
    rows = []
    print("=" * 110)
    print("V4 INT8 PagedAttention Microbench (Qwen2.5-7B shape)")
    print("=" * 110)
    header = (
        f"{'B':>4}{'Seq':>8}{'BF16':>10}{'V3':>10}{'V4':>10}"
        f"{'Auto':>10}{'V4q':>10}{'V4s':>10}{'V3/V4':>10}{'bound':>22}"
    )
    print(header)
    for batch, seq_len in configs:
        query = torch.randn(batch, Hq, D, device=device, dtype=torch.bfloat16)
        bf16_k, bf16_v, int8_k, int8_v, block_tables, seq_lens, k_scale, v_scale = build_cache_pair(
            batch, seq_len, Hkv, D, block_size, device
        )
        bf16_ms = benchmark_cuda(
            lambda: bf16_paged_attention(query, bf16_k, bf16_v, block_tables, seq_lens)
        )
        v3_ms = benchmark_cuda(
            lambda: int8_paged_attention(
                query, int8_k, int8_v, block_tables, seq_lens, k_scale, v_scale, impl="v3"
            )
        )
        v4_ms = benchmark_cuda(
            lambda: int8_paged_attention(
                query, int8_k, int8_v, block_tables, seq_lens, k_scale, v_scale, impl="v4"
            )
        )
        auto_ms = benchmark_cuda(
            lambda: int8_paged_attention(
                query, int8_k, int8_v, block_tables, seq_lens, k_scale, v_scale, impl="auto"
            )
        )
        v4q_ms = benchmark_cuda(
            lambda: int8_paged_attention(
                query,
                int8_k,
                int8_v,
                block_tables,
                seq_lens,
                k_scale,
                v_scale,
                impl="v4",
                quantize_q=True,
            )
        )
        # Force split-KV for long ctx / small batch
        forced_splits = 8 if batch == 1 and seq_len >= 2048 else 1
        v4s_ms = benchmark_cuda(
            lambda: int8_paged_attention(
                query,
                int8_k,
                int8_v,
                block_tables,
                seq_lens,
                k_scale,
                v_scale,
                impl="v4",
                num_splits=forced_splits,
            )
        )

        # V3 reloads KV ~q_per_kv times; V4 once
        bytes_v3 = estimate_kv_bytes(batch, seq_len, Hkv, D, 1, gqa_reload=Hq // Hkv)
        hint = roofline_hint(v3_ms, bytes_v3)
        row = {
            "batch": batch,
            "seq_len": seq_len,
            "bf16_ms": bf16_ms,
            "int8_v3_ms": v3_ms,
            "int8_v4_ms": v4_ms,
            "int8_auto_ms": auto_ms,
            "int8_v4_q_ms": v4q_ms,
            "int8_v4_split_ms": v4s_ms,
            "forced_splits": forced_splits,
            "speedup_v3_over_v4": (v3_ms / v4_ms) if v4_ms > 0 else None,
            "speedup_bf16_over_v4": (bf16_ms / v4_ms) if v4_ms > 0 else None,
            "v3_est_kv_bytes": bytes_v3,
            "v4_est_kv_bytes": estimate_kv_bytes(batch, seq_len, Hkv, D, 1, gqa_reload=1),
            "v3_roofline": hint,
            "device": torch.cuda.get_device_name(0),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        rows.append(row)
        print(
            f"{batch:4d}{seq_len:8d}{bf16_ms:10.4f}{v3_ms:10.4f}{v4_ms:10.4f}"
            f"{auto_ms:10.4f}{v4q_ms:10.4f}{v4s_ms:10.4f}{(v3_ms / v4_ms):10.3f}x"
            f"{hint['bound']:>22}"
        )

    # Decision summary for Phase routing
    long = [r for r in rows if r["seq_len"] >= 2048 and r["batch"] == 1]
    short = [r for r in rows if r["seq_len"] <= 512 and r["batch"] == 1]
    decision = {
        "long_ctx_v3_bound": long[0]["v3_roofline"]["bound"] if long else None,
        "short_ctx_v3_bound": short[0]["v3_roofline"]["bound"] if short else None,
        "recommendation": (
            "Prioritize GQA reuse (Phase1) — long-ctx memory-leaning"
            if long and long[0]["v3_roofline"]["bound"] == "memory-leaning"
            else "Push tl.dot / compute path early — short-ctx compute-leaning or mixed"
        ),
    }
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        json_path = OUT_DIR / "microbench.json"
        csv_path = OUT_DIR / "microbench.csv"
        with open(json_path, "w") as f:
            json.dump({"rows": rows, "decision": decision}, f, indent=2)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "batch",
                    "seq_len",
                    "bf16_ms",
                    "int8_v3_ms",
                    "int8_v4_ms",
                    "int8_auto_ms",
                    "int8_v4_q_ms",
                    "int8_v4_split_ms",
                    "forced_splits",
                    "speedup_v3_over_v4",
                    "speedup_bf16_over_v4",
                ],
            )
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        with open(OUT_DIR / "decision.md", "w") as f:
            f.write("# V4 Phase0 Baseline Decision\n\n")
            f.write(f"- Device: `{rows[0]['device']}`\n")
            f.write(f"- Long-ctx V3 bound: **{decision['long_ctx_v3_bound']}**\n")
            f.write(f"- Short-ctx V3 bound: **{decision['short_ctx_v3_bound']}**\n")
            f.write(f"- Recommendation: {decision['recommendation']}\n")
            f.write("\n## Notes\n\n")
            f.write(
                "- Bandwidth estimate assumes V3 reloads KV `Hq/Hkv` times; V4 once.\n"
                "- Full Nsight Compute (`ncu`) is optional; run "
                "`scripts/10_ncu_profile.sh` when the toolkit is installed.\n"
            )
        print(f"\nSaved: {json_path}")
        print(f"Decision: {decision['recommendation']}")
    return rows, decision
if __name__ == "__main__":
    run_benchmark(save=True)
