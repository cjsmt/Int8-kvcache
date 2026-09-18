import json
import os
import subprocess
import sys
OUTPUT_DIR = "outputs/batch_sweep"
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)
BATCHES = [
    1,
    2,
    4,
    8,
]
MODES = [
    "bf16",
    "int8",
]
CONTEXT = 2048
NEW_TOKENS = 128
rows = []
for batch in BATCHES:
    for mode in MODES:
        output = f"{OUTPUT_DIR}/{mode}_b{batch}.json"
        cmd = [
            sys.executable,
            "bench/run_qwen_case_final.py",
            "--mode",
            mode,
            "--context-len",
            str(CONTEXT),
            "--new-tokens",
            str(NEW_TOKENS),
            "--batch-size",
            str(batch),
            "--output",
            output,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
            )
        except subprocess.CalledProcessError:
            print(f"FAILED: mode={mode}, batch={batch}")
            continue
        with open(
            output,
            encoding="utf-8",
        ) as f:
            result = json.load(f)
        rows.append(
            {
                "mode": mode,
                "batch": batch,
                "elapsed_s": result["elapsed_s"],
                "throughput_tok_s": result["overall_tokens_per_second"],
                "peak_allocated_mb": result["peak_allocated_mb"],
            }
        )
print()
print("=" * 90)
print("BATCH SIZE SWEEP")
print("=" * 90)
print(f"{'Mode':>8}{'Batch':>10}{'Time(s)':>14}{'Tok/s':>16}{'Peak MB':>14}")
for r in rows:
    print(
        f"{r['mode']:>8}"
        f"{r['batch']:10d}"
        f"{r['elapsed_s']:14.3f}"
        f"{r['throughput_tok_s']:16.2f}"
        f"{r['peak_allocated_mb']:14.2f}"
    )
with open(
    f"{OUTPUT_DIR}/summary.json",
    "w",
) as f:
    json.dump(
        rows,
        f,
        indent=2,
    )
