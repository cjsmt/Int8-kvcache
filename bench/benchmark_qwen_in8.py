import json
import os
import subprocess
import sys


CONTEXTS = [
    512,
    1024,
    2048,
    4096,
]


NEW_TOKENS = 128


OUTPUT_DIR = "outputs/final_benchmark"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


def run_case(
    mode,
    context_len,
):

    path = f"{OUTPUT_DIR}/{mode}_{context_len}.json"

    cmd = [
        sys.executable,
        "bench/run_qwen_case.py",
        "--mode",
        mode,
        "--context-len",
        str(context_len),
        "--new-tokens",
        str(NEW_TOKENS),
        "--output",
        path,
    ]

    print()
    print("=" * 100)

    print("RUN:", " ".join(cmd))

    print("=" * 100)

    subprocess.run(
        cmd,
        check=True,
    )

    with open(
        path,
        encoding="utf-8",
    ) as f:
        return json.load(f)


results = []


for context in CONTEXTS:
    bf16 = run_case(
        "bf16",
        context,
    )

    int8 = run_case(
        "int8",
        context,
    )

    speedup = bf16["ms_per_token"] / int8["ms_per_token"]

    memory_ratio = None
    memory_reduction = None

    if int8["bf16_cache_mb"] is not None and int8["int8_cache_mb"] is not None:
        bf16_kv = int8["bf16_cache_mb"]

        int8_kv = int8["int8_cache_mb"] + int8["scale_mb"]

        memory_ratio = bf16_kv / int8_kv

        memory_reduction = (1.0 - int8_kv / bf16_kv) * 100

    results.append(
        {
            "context": context,
            "bf16_ms_per_token": bf16["ms_per_token"],
            "int8_ms_per_token": int8["ms_per_token"],
            "speedup": speedup,
            "bf16_tok_s": bf16["tokens_per_second"],
            "int8_tok_s": int8["tokens_per_second"],
            "bf16_kv_mb": int8["bf16_cache_mb"],
            "int8_kv_mb": (
                None
                if int8["int8_cache_mb"] is None
                else int8["int8_cache_mb"] + int8["scale_mb"]
            ),
            "kv_compression": memory_ratio,
            "kv_reduction_percent": memory_reduction,
        }
    )


# ============================================================
# print
# ============================================================

print()
print("=" * 130)
print("FINAL QWEN2.5-7B INT8 KVCACHE BENCHMARK")
print("=" * 130)


header = (
    f"{'Context':>8}"
    f"{'BF16 ms/tok':>15}"
    f"{'INT8 ms/tok':>15}"
    f"{'Speedup':>12}"
    f"{'BF16 tok/s':>14}"
    f"{'INT8 tok/s':>14}"
    f"{'BF16 KV MB':>14}"
    f"{'INT8 KV MB':>14}"
    f"{'Reduction':>12}"
)

print(header)


for r in results:
    reduction = (
        "N/A"
        if r["kv_reduction_percent"] is None
        else f"{r['kv_reduction_percent']:.2f}%"
    )

    print(
        f"{r['context']:8d}"
        f"{r['bf16_ms_per_token']:15.3f}"
        f"{r['int8_ms_per_token']:15.3f}"
        f"{r['speedup']:12.3f}x"
        f"{r['bf16_tok_s']:14.2f}"
        f"{r['int8_tok_s']:14.2f}"
        f"{r['bf16_kv_mb']:14.2f}"
        f"{r['int8_kv_mb']:14.2f}"
        f"{reduction:>12}"
    )


# ============================================================
# save summary
# ============================================================

with open(
    f"{OUTPUT_DIR}/summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        results,
        f,
        indent=2,
    )
