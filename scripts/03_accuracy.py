import torch
import torch.nn.functional as F
from src.attention_ref import (
    decode_attention_ref,
    decode_attention_int8_dynamic,
)


def calc(ref, out):
    a = ref.float().flatten()
    b = out.float().flatten()
    return {
        "cos": F.cosine_similarity(a, b, dim=0).item(),
        "mae": (a - b).abs().mean().item(),
        "rel_l2": (a - b).norm().div(a.norm().clamp_min(1e-8)).item(),
    }
data = torch.load("outputs/kv_sample.pt", map_location="cpu")
for layer, kv in data.items():
    k = kv["k"][:1].cuda().to(torch.bfloat16)
    v = kv["v"][:1].cuda().to(torch.bfloat16)
    B, Hkv, T, D = k.shape
    Hq = 28

    # 这里先用随机 q 做算子误差验证。；它不是模型端到端 accuracy，但可以验证真实 KV 分布下的数值误差。
    torch.manual_seed(layer)
    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.bfloat16)
    ref = decode_attention_ref(q, k, v)
    out = decode_attention_int8_dynamic(q, k, v)
    print("layer", layer, calc(ref, out))
