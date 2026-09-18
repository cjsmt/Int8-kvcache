import torch
from src.triton_ops.int8_cache_write import (
    int8_kv_cache_write,
    torch_int8_kv_cache_write_reference,
)
torch.manual_seed(41)
device = "cuda"
T = 37
H = 4
D = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 8
key = torch.randn(
    T,
    H,
    D,
    device=device,
    dtype=torch.bfloat16,
)
value = torch.randn_like(key)
k_scale = torch.tensor(
    [0.03, 0.025, 0.04, 0.035],
    device=device,
    dtype=torch.float32,
)
v_scale = torch.tensor(
    [0.035, 0.03, 0.025, 0.04],
    device=device,
    dtype=torch.float32,
)
# 故意做非连续 physical slots
slot_mapping = torch.tensor(
    list(range(16, 32))
    + list(range(64, 80))
    + list(range(96, 101)),
    device=device,
    dtype=torch.int64,
)[:T]
kc_ref = torch.zeros(
    NUM_BLOCKS,
    BLOCK_SIZE,
    H,
    D,
    device=device,
    dtype=torch.int8,
)
vc_ref = torch.zeros_like(kc_ref)
kc_tri = torch.zeros_like(kc_ref)
vc_tri = torch.zeros_like(vc_ref)
torch_int8_kv_cache_write_reference(
    key,
    value,
    kc_ref,
    vc_ref,
    slot_mapping,
    k_scale,
    v_scale,
)
int8_kv_cache_write(
    key,
    value,
    kc_tri,
    vc_tri,
    slot_mapping,
    k_scale,
    v_scale,
)
torch.cuda.synchronize()
k_diff = (
    kc_ref.to(torch.int16)
    -
    kc_tri.to(torch.int16)
).abs().max()
v_diff = (
    vc_ref.to(torch.int16)
    -
    vc_tri.to(torch.int16)
).abs().max()
print("K max diff =", k_diff.item())
print("V max diff =", v_diff.item())
