import torch
import torch.nn.functional as F

from src.attention_ref import (
    decode_attention_ref,
    decode_attention_int8_dynamic,
)


def metrics(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)

    cos = F.cosine_similarity(a, b, dim=0).item()
    mae = (a - b).abs().mean().item()
    max_err = (a - b).abs().max().item()
    rel_l2 = ((a - b).norm() / a.norm().clamp_min(1e-8)).item()
    return cos, mae, max_err, rel_l2


def test_int8_attention():
    torch.manual_seed(0)

    B = 2
    Hq = 28
    Hkv = 4
    T = 1024
    D = 128

    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.bfloat16)

    ref = decode_attention_ref(q, k, v)
    out = decode_attention_int8_dynamic(q, k, v)

    cos, mae, max_err, rel_l2 = metrics(ref, out)

    print("cosine =", cos)
    print("mae =", mae)
    print("max error =", max_err)
    print("relative l2 =", rel_l2)

    assert cos > 0.99
    assert rel_l2 < 0.10
