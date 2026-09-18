# V4 Phase0 Baseline Decision

- Device: `NVIDIA GeForce RTX 4090`
- Short-ctx (B=1, seq≤1024): occupancy / launch dominated → prefer **V3** (or `auto`)
- Long-ctx (B=1, seq≥4096) / large batch: KV traffic dominated → prefer **V4**
- Recommendation: default `impl="auto"`; force `impl="v4"` for long-context / throughput sweeps

## Highlights (kernel ms)

| Case | BF16 | V3 | V4 | Auto |
|------|-----:|---:|---:|-----:|
| B1 / 512 | 0.055 | 0.132 | 0.296 | 0.125 |
| B1 / 4096 | 0.312 | 0.400 | 0.266 | **0.266** |
| B8 / 4096 | 0.463 | 0.525 | 0.266 | **0.258** |

## Notes

- Bandwidth estimate assumes V3 reloads KV `Hq/Hkv` times; V4 once.
- Full Nsight Compute (`ncu`) is optional; run `scripts/10_ncu_profile.sh` when the toolkit is installed.
- Project `.venv` ships torch cu130 which needs a newer driver; microbench was run with conda `torch 2.7.0+cu128`.
