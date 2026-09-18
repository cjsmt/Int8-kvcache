"""vLLM 0.26 attention backend: INT8 paged KV + Triton int8_paged_attention."""

import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
)

from src.triton_ops.int8_paged_attention import (
    int8_paged_attention,
)

from vllm_int8.vllm_int8_patch import (
    INT8_CACHE_POOL,
)


class INT8AttentionMetadata:
    """Decode metadata: block_tables, seq_lens."""
    def __init__(
        self,
        block_tables,
        seq_lens,
    ):
        self.block_tables = block_tables
        self.seq_lens = seq_lens


class INT8PagedAttentionImpl(AttentionImpl):
    def __init__(
        self,
        layer_name,
        num_heads,
        head_size,
        num_kv_heads,
        scale,
        **kwargs,
    ):
        self.layer_name = layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        print("[INT8 Impl Init]", layer_name)
    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
    ):
        """query [num_tokens, heads, dim] (vLLM 0.26)."""
        print("[INT8 Attention Forward]")
        cache = INT8_CACHE_POOL[self.layer_name]
        k_cache = cache["k"]
        v_cache = cache["v"]
        k_scale = cache["k_scale"]
        v_scale = cache["v_scale"]
        if query.ndim == 2:
            query = query.unsqueeze(0)
        block_tables = attn_metadata.block_tables
        seq_lens = attn_metadata.seq_lens
        out = int8_paged_attention(
            query,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            k_scale,
            v_scale,
        )
        out = out.squeeze(0)
        output.copy_(out)
        return output


class INT8PagedAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name():
        return "INT8_PAGED_ATTENTION"
    @staticmethod
    def get_impl_cls():
        return INT8PagedAttentionImpl
