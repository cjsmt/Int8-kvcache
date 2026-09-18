import torch
from src.quant import (
    quant_per_tensor,
    dequant_per_tensor,
    quant_kv_per_head,
    dequant_kv_per_head,
)


def test_per_tensor():
    x = torch.randn(4, 128, device="cuda", dtype=torch.float32)
    q, s = quant_per_tensor(x)
    x2 = dequant_per_tensor(q, s)
    mae = (x - x2).abs().mean().item()
    print("per tensor MAE:", mae)
    assert mae < 0.05


def test_per_head():
    x = torch.randn(2, 4, 256, 128, device="cuda")
    q, s = quant_kv_per_head(x)
    x2 = dequant_kv_per_head(q, s)
    mae = (x - x2).abs().mean().item()
    print("per head MAE:", mae)
    assert mae < 0.05
