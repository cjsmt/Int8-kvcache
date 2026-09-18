# V4 Phase0 Baseline Decision

- Device: `NVIDIA GeForce RTX 4090`
- Data source: **kernel microbench** (`bench/bench_paged_attention.py`) — calls `int8_paged_attention` **directly**, not via vLLM patch
- Therefore **B>1 rows are valid** for the Triton operator (unaffected by the old `_is_supported_decode` B=1 gate)

## Highlights (kernel ms, re-run 2026-09-19)

| Case | BF16 | V3 | V4 | Auto |
|------|-----:|---:|---:|-----:|
| B1 / 512 | 0.056 | 0.136 | 0.226 | 0.128 |
| B1 / 4096 | 0.342 | 0.414 | 0.270 | **0.270** |
| B8 / 4096 | 0.464 | 0.519 | 0.266 | **0.261** |

## Caveat on e2e Batch Sweep

Old vLLM `int8_only` path gated INT8 attention to `batch==1`. E2E Batch Sweep INT8 numbers for B>1 (README §2.4, V3 era) did **not** exercise custom INT8 PagedAttention and must not be read as operator results. Gate fixed in `vllm_int8_attention_patch.py`; e2e re-measure still pending (torch/vLLM CUDA env).

## Notes

- Bandwidth estimate assumes V3 reloads KV `Hq/Hkv` times; V4 once.
- Optional: `bash scripts/10_ncu_profile.sh` when `ncu` is installed.
