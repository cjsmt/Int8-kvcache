from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerInt8Cache:
    key_cache: torch.Tensor
    value_cache: torch.Tensor

    k_scale: torch.Tensor
    v_scale: torch.Tensor


# layer_name -> cache
INT8_CACHE_POOL: dict[str, LayerInt8Cache] = {}


def has_layer_cache(
    layer_name: str,
) -> bool:
    return layer_name in INT8_CACHE_POOL


def get_layer_cache(
    layer_name: str,
) -> LayerInt8Cache:
    return INT8_CACHE_POOL[layer_name]


def allocate_layer_cache(
    *,
    layer_name: str,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    device: torch.device,
):
    """
    我们自己的 shadow INT8 layout：

        [num_blocks,
         block_size,
         Hkv,
         D]

    这正好和前面写好的 int8_cache_write.py /
    int8_paged_attention.py 对齐。
    """

    key_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.int8,
        device=device,
    )

    value_cache = torch.zeros_like(key_cache)

    INT8_CACHE_POOL[layer_name] = LayerInt8Cache(
        key_cache=key_cache,
        value_cache=value_cache,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    return INT8_CACHE_POOL[layer_name]


def clear_cache_pool():
    INT8_CACHE_POOL.clear()
