import torch
from src.paged_cache import make_paged_cache, gather_from_paged_cache

k = torch.randn(2, 4, 127, 128, device="cuda")
v = torch.randn_like(k)

kc, vc, bt, sl = make_paged_cache(k, v, 16)
k2, v2 = gather_from_paged_cache(kc, vc, bt, sl)

print((k-k2).abs().max())
print((v-v2).abs().max())

assert torch.equal(k, k2)
assert torch.equal(v, v2)
