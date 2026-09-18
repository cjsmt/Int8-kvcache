import torch

EPS = 1e-6


def quant_per_tensor(x: torch.Tensor):
    """整个 tensor 使用一个 scale。"""
    amax = x.abs().amax().clamp_min(EPS)
    scale = amax / 127.0
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequant_per_tensor(q: torch.Tensor, scale: torch.Tensor, dtype=torch.float32):
    return q.to(dtype) * scale.to(dtype)


def quant_q_per_head(q: torch.Tensor):
    """
    q: [B, Hq, D]
    scale: [B, Hq, 1]
    """
    amax = q.abs().amax(dim=-1, keepdim=True).clamp_min(EPS)
    scale = amax / 127.0
    q8 = torch.round(q / scale).clamp(-127, 127).to(torch.int8)
    return q8, scale


def quant_kv_per_head(x: torch.Tensor):
    """
    x: [B, Hkv, T, D]
    每个 batch、每个 KV head 一个 scale。
    scale: [B, Hkv, 1, 1]
    """
    amax = x.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    scale = amax / 127.0
    q8 = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q8, scale


def dequant_kv_per_head(q: torch.Tensor, scale: torch.Tensor, dtype=torch.float32):
    return q.to(dtype) * scale.to(dtype)


def quant_kv_static_per_head(x: torch.Tensor, scale: torch.Tensor):
    """
    静态量化。
    x:     [B, Hkv, T, D]
    scale: [1, Hkv, 1, 1] 或可 broadcast 的 shape
    """
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q
