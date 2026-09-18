import json
import os
import subprocess
import sys


OUTPUT_DIR = "outputs/decode_tpot"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


CONTEXTS = [
    512,
    1024,
    2048,
    4096,
]


MODES = [
    "bf16",
    "int8",
]


def run(
    mode,
    context,
    new_tokens,
):

    output = f"{OUTPUT_DIR}/{mode}_ctx{context}_n{new_tokens}.json"

    cmd = [
        sys.executable,
        "bench/run_qwen_case_final.py",
        "--mode",
        mode,
        "--context-len",
        str(context),
        "--new-tokens",
        str(new_tokens),
        "--batch-size",
        "1",
        "--output",
        output,
    ]

    subprocess.run(
        cmd,
        check=True,
    )

    with open(
        output,
        encoding="utf-8",
    ) as f:
        return json.load(f)


rows = []


for context in CONTEXTS:
    for mode in MODES:
        r1 = run(
            mode,
            context,
            1,
        )

        r128 = run(
            mode,
            context,
            128,
        )

        t1 = r1["elapsed_s"]

        t128 = r128["elapsed_s"]

        decode_seconds = t128 - t1

        tpot_ms = decode_seconds / 127 * 1000

        decode_tok_s = 1000.0 / tpot_ms

        rows.append(
            {
                "mode": mode,
                "context": context,
                "t1_ms": t1 * 1000,
                "t128_ms": t128 * 1000,
                "decode_tpot_ms": tpot_ms,
                "decode_tok_s": decode_tok_s,
            }
        )


print()
print("=" * 90)
print("APPROXIMATE DECODE TPOT")
print("=" * 90)

print(
    f"{'Mode':>8}"
    f"{'Context':>10}"
    f"{'T1(ms)':>14}"
    f"{'T128(ms)':>14}"
    f"{'TPOT(ms)':>14}"
    f"{'Decode tok/s':>16}"
)


for r in rows:
    print(
        f"{r['mode']:>8}"
        f"{r['context']:10d}"
        f"{r['t1_ms']:14.3f}"
        f"{r['t128_ms']:14.3f}"
        f"{r['decode_tpot_ms']:14.3f}"
        f"{r['decode_tok_s']:16.2f}"
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
