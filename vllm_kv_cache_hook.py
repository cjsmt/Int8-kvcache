"""
vllm_kv_cache_hook.py


功能：

1. 启动 Qwen2.5-7B vLLM

2. Hook KV cache 分配入口，打印 layout

3. 用于 INT8 KVCache 接入前分析


说明（vLLM 0.26 / V1 engine）：

- 默认 EngineCore 跑在子进程，主进程 monkeypatch 无效
  → 必须 VLLM_ENABLE_V1_MULTIPROCESSING=0

- 当前默认是 V2 Model Runner：
  vllm.v1.worker.gpu.attn_utils.init_kv_cache

- 旧 V1 runner 则是：
  GPUModelRunner.initialize_kv_cache_tensors

"""

import os

# 必须在 import vllm 之前设置，否则 hook 打不到 EngineCore
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch

from vllm import LLM
from vllm.sampling_params import SamplingParams


def _print_tensor(name: str, t: torch.Tensor) -> None:
    print(f"{name}:")
    print("  shape:", tuple(t.shape))
    print("  dtype:", t.dtype)
    print("  device:", t.device)
    print("  stride:", t.stride())
    print("  numel:", t.numel())
    print("  nbytes:", t.numel() * t.element_size())


def _dump_kv_caches(kv_caches, extra: dict | None = None) -> None:
    print("\n")
    print("=" * 80)
    print("========= vLLM KV CACHE DEBUG =========")
    print("=" * 80)

    if extra:
        for k, v in extra.items():
            print(f"{k}:", v)

    print("kv_caches type:", type(kv_caches))
    print("num layers / keys:", len(kv_caches))

    for i, (layer_name, layer_cache) in enumerate(kv_caches.items()):
        print("\n")
        print("-" * 60)
        print("Layer key:", layer_name)
        print("cache object:", type(layer_cache))

        if isinstance(layer_cache, (tuple, list)):
            print("list/tuple length:", len(layer_cache))
            for j, item in enumerate(layer_cache):
                if torch.is_tensor(item):
                    _print_tensor(f"item {j}", item)
                else:
                    print(f"item {j}:", type(item))
        elif torch.is_tensor(layer_cache):
            _print_tensor("tensor", layer_cache)
        else:
            print("Unknown cache format:", layer_cache)

        # 详细看前 2 个 key 即可
        if i >= 1:
            break

    print("\n")
    print("=" * 80)
    print("========= END KV DEBUG =========")
    print("=" * 80)


def hook_cache_engine():
    """Hook V2 + V1 两条分配路径，只打印，不改逻辑。"""

    # ---- V2 Model Runner（当前默认）----
    import vllm.v1.worker.gpu.attn_utils as attn_utils

    original_init_kv_cache = attn_utils.init_kv_cache

    def new_init_kv_cache(*args, **kwargs):
        kv_caches = original_init_kv_cache(*args, **kwargs)
        cfg = kwargs.get("kv_cache_config")
        if cfg is None and len(args) >= 3:
            cfg = args[2]
        cache_dtype = kwargs.get("cache_dtype")
        if cache_dtype is None and len(args) >= 6:
            cache_dtype = args[5]
        _dump_kv_caches(
            kv_caches,
            extra={
                "path": "v2 attn_utils.init_kv_cache",
                "cache_dtype": cache_dtype,
                "num_blocks": getattr(cfg, "num_blocks", None),
                "num kv_cache_groups": len(getattr(cfg, "kv_cache_groups", []) or []),
                "num kv_cache_tensors": len(
                    getattr(cfg, "kv_cache_tensors", []) or []
                ),
            },
        )
        return kv_caches

    attn_utils.init_kv_cache = new_init_kv_cache

    # V2 model_runner 可能已 from-import init_kv_cache，再补一层
    import vllm.v1.worker.gpu.model_runner as model_runner_v2

    if getattr(model_runner_v2, "init_kv_cache", None) is original_init_kv_cache:
        model_runner_v2.init_kv_cache = new_init_kv_cache

    # ---- V1 Model Runner（兼容）----
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_v1 = GPUModelRunner.initialize_kv_cache_tensors

    def new_initialize_kv_cache_tensors(self, *args, **kwargs):
        kv_caches = original_v1(self, *args, **kwargs)
        cfg = getattr(self, "kv_cache_config", None)
        cache_cfg = getattr(self, "cache_config", None)
        _dump_kv_caches(
            kv_caches,
            extra={
                "path": "v1 GPUModelRunner.initialize_kv_cache_tensors",
                "cache_dtype": getattr(cache_cfg, "cache_dtype", None),
                "block_size": getattr(cache_cfg, "block_size", None),
                "num_blocks": getattr(cfg, "num_blocks", None),
            },
        )
        return kv_caches

    GPUModelRunner.initialize_kv_cache_tensors = new_initialize_kv_cache_tensors
    print("[hook] installed (multiprocessing disabled, V1+V2 paths)")


def run_qwen():
    model_path = "Qwen/Qwen2.5-7B-Instruct"

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
    )

    prompts = ["Explain transformer architecture."]

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    for out in outputs:
        print(out.outputs[0].text)


if __name__ == "__main__":
    hook_cache_engine()
    run_qwen()
