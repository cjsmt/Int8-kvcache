"""
vllm_int8_patch.py


通过 monkey patch 方式接入 vLLM。


功能：

1. Hook KV Cache 创建
2. 将 BF16 cache 旁路量化为 INT8 并保存 scale（不替换原 cache）
3. Hook Attention forward（当前阶段只打印）


兼容 vLLM 0.26 V1 engine + V2 Model Runner：

- init_kv_cache: vllm.v1.worker.gpu.attn_utils
- Attention:     vllm.model_executor.layers.attention.Attention

"""

import os

# EngineCore 默认多进程，主进程 monkeypatch 无效
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

from vllm_int8.vllm_int8_kvcache_adapter import quantize_per_head


INT8_CACHE_POOL = {}


def _split_fused_kv(cache: torch.Tensor):
    """
    FlashAttn 逻辑 fused layout:
      [num_blocks, num_kv_heads, block_size, 2*head_dim]
    """
    if cache.ndim != 4:
        raise ValueError(f"unexpected kv cache ndim={cache.ndim}, shape={cache.shape}")

    hidden = cache.shape[-1]
    if hidden % 2 != 0:
        raise ValueError(f"last dim must be 2*head_dim, got {hidden}")

    head_dim = hidden // 2
    return cache[..., :head_dim], cache[..., head_dim:]


def patch_kv_cache_allocation():
    """Patch vLLM KV cache 初始化，不直接改源码。"""

    try:
        import vllm.v1.worker.gpu.attn_utils as attn_utils
    except Exception as e:
        raise RuntimeError(
            "Cannot import vllm.v1.worker.gpu.attn_utils "
            "(need vLLM V1/V2 worker, e.g. 0.26+)"
        ) from e

    original_init = attn_utils.init_kv_cache

    def int8_init_kv_cache(*args, **kwargs):
        print("[INT8 KV] intercept init_kv_cache")

        result = original_init(*args, **kwargs)
        kv_caches = result
        print("[INT8 KV] layers:", len(kv_caches))

        int8_caches = {}
        for layer_name, cache in kv_caches.items():
            print("convert layer:", layer_name)

            if isinstance(cache, (list, tuple)):
                print("skip non-tensor cache:", type(cache))
                continue
            if not torch.is_tensor(cache):
                print("skip unknown cache:", type(cache))
                continue

            print("original:", tuple(cache.shape), cache.dtype)

            # 刚分配时常为全 0，仍可校验 shape；scale 近似 1e-6
            try:
                k_cache, v_cache = _split_fused_kv(cache)
                k_int8, k_scale = quantize_per_head(k_cache)
                v_int8, v_scale = quantize_per_head(v_cache)
            except (ValueError, AssertionError) as exc:
                print("skip layout:", exc)
                continue

            int8_caches[layer_name] = {
                "k": k_int8,
                "v": v_int8,
                "k_scale": k_scale,
                "v_scale": v_scale,
            }
            print("INT8:", k_int8.shape, k_int8.dtype)

        INT8_CACHE_POOL.update(int8_caches)
        # 第一阶段：不替换 vLLM 原 cache
        return result

    attn_utils.init_kv_cache = int8_init_kv_cache

    # V2 model_runner 使用 from-import，必须同步替换绑定名
    try:
        import vllm.v1.worker.gpu.model_runner as model_runner_v2

        if getattr(model_runner_v2, "init_kv_cache", None) is original_init:
            model_runner_v2.init_kv_cache = int8_init_kv_cache
    except Exception:
        pass

    # 兼容 V1 runner
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        original_v1 = GPUModelRunner.initialize_kv_cache_tensors

        def int8_initialize_kv_cache_tensors(self, *args, **kwargs):
            print("[INT8 KV] intercept initialize_kv_cache_tensors (V1 runner)")
            result = original_v1(self, *args, **kwargs)
            print("[INT8 KV] V1 layers:", len(result))
            return result

        GPUModelRunner.initialize_kv_cache_tensors = int8_initialize_kv_cache_tensors
    except Exception:
        pass

    print("[INT8 KV] init_kv_cache patched")


def patch_attention_forward():
    """第一版只 hook Attention.forward，不改变计算。"""

    try:
        from vllm.model_executor.layers.attention import Attention
    except Exception as e:
        raise RuntimeError(
            "Cannot import Attention "
            "(expected vllm.model_executor.layers.attention.Attention)"
        ) from e

    original_forward = Attention.forward

    def int8_attention_forward(self, *args, **kwargs):
        print("[INT8 Attention] forward called")
        return original_forward(self, *args, **kwargs)

    Attention.forward = int8_attention_forward
    print("[INT8 Attention] patched")


def apply_vllm_int8_patch():
    patch_kv_cache_allocation()
    patch_attention_forward()

    print(
        """
========================================
vLLM INT8 KVCache patch enabled

Current stage:
[OK] KV cache interception
[OK] INT8 calibration (side pool)

TODO:
[ ] replace attention kernel
========================================
"""
    )
