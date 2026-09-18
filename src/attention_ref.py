import math
import torch
import torch.nn.functional as F

from src.quant import (
    quant_q_per_head,
    quant_kv_per_head,
    dequant_kv_per_head,
)


def expand_kv_for_gqa(x: torch.Tensor, num_query_heads: int):
    """
    x: [B, Hkv, T, D]
    return: [B, Hq, T, D]
    """
    hkv = x.shape[1]
    assert num_query_heads % hkv == 0
    repeat = num_query_heads // hkv
    return x.repeat_interleave(repeat, dim=1)


def decode_attention_ref(q, k, v):
    """
    q: [B, Hq, D]
    k: [B, Hkv, T, D]
    v: [B, Hkv, T, D]
    output: [B, Hq, D]
    """
    B, Hq, D = q.shape

    k = expand_kv_for_gqa(k, Hq)
    v = expand_kv_for_gqa(v, Hq)

    scores = torch.einsum("bhd,bhtd->bht", q.float(), k.float())
    scores = scores / math.sqrt(D)

    p = F.softmax(scores, dim=-1)
    out = torch.einsum("bht,bhtd->bhd", p, v.float())
    return out


def decode_attention_int8_dynamic(q, k, v):
    """
    这是“正确性 reference”，不是最终高性能 kernel。

    做法：
    1. q per-head 动态量化
    2. k/v per-head 动态量化
    3. 反量化后走 reference attention

    用来证明量化误差是否可接受。
    """
    q8, qs = quant_q_per_head(q)
    k8, ks = quant_kv_per_head(k)
    v8, vs = quant_kv_per_head(v)

    q_hat = q8.float() * qs.float()
    k_hat = dequant_kv_per_head(k8, ks)
    v_hat = dequant_kv_per_head(v8, vs)

    return decode_attention_ref(q_hat, k_hat, v_hat)