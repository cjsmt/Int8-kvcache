import time
import torch
from src.attention_ref import (
    decode_attention_ref,
    decode_attention_int8_dynamic,
)


def bench(fn, *args, warmup=20, iters=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000 / iters
for T in [256, 512, 1024, 2048, 4096]:
    B, Hq, Hkv, D = 1, 28, 4, 128
    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.bfloat16)
    t_ref = bench(decode_attention_ref, q, k, v)
    t_dyn = bench(decode_attention_int8_dynamic, q, k, v)
    print(
        f"T={T:5d}  "
        f"bf16={t_ref:8.4f} ms  "
        f"dynamic_int8={t_dyn:8.4f} ms"
    )
