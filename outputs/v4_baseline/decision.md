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

Old vLLM `int8_only` path gated INT8 attention to `batch==1`. That V3-era table is archived in README §2.4.

V4 e2e re-measure (2026-09-19): `torch 2.9.1+cu128` + `vLLM 0.16.0` on CUDA 12.8. Every INT8 row has `int8_decode_hits=3556` and `int8_last_batch==B`. See `outputs/batch_sweep/` and README §2.4. E2E INT8 is still slower than native BF16 because of shadow dual-write + Python patch overhead; kernel wins stay in the microbench table above.

## Notes

- Bandwidth estimate assumes V3 reloads KV `Hq/Hkv` times; V4 once.
- Optional: `bash scripts/10_ncu_profile.sh` when `ncu` is installed.
