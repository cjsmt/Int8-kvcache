#!/usr/bin/env bash
# Optional Nsight Compute profile for V4 Phase0.
# Requires: ncu on PATH, CUDA GPU available.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/outputs/v4_baseline"
mkdir -p "${OUT}"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found; skip hardware counter profile."
  echo "Microbench roofline hints are in ${OUT}/decision.md"
  exit 0
fi

source "${ROOT}/.venv/bin/activate"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

ncu --set full -o "${OUT}/int8_paged_attention" --force-overwrite \
  python - <<'PY'
import torch
from bench.bench_paged_attention import build_cache_pair, benchmark_cuda
from src.triton_ops.int8_paged_attention import int8_paged_attention

device = "cuda"
B, Hq, Hkv, D, seq, block = 1, 28, 4, 128, 2048, 16
q = torch.randn(B, Hq, D, device=device, dtype=torch.bfloat16)
_, _, ik, iv, bt, sl, ks, vs = build_cache_pair(B, seq, Hkv, D, block, device)
# Warmup compile
for _ in range(5):
    int8_paged_attention(q, ik, iv, bt, sl, ks, vs, impl="v4")
torch.cuda.synchronize()
for _ in range(20):
    int8_paged_attention(q, ik, iv, bt, sl, ks, vs, impl="v4")
torch.cuda.synchronize()
print("done")
PY

echo "Wrote ${OUT}/int8_paged_attention.ncu-rep"
