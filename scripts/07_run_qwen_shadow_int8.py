import os

# ============================================================
# 重要：
# monkey patch 调试阶段禁用 V1 multiprocessing
# ============================================================

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


from vllm_int8.static_scales import (
    load_static_scales,
)

from vllm_int8.vllm_shadow_cache_patch import (
    apply_shadow_cache_patch,
)


# ============================================================
# 1. load calibration scales
# ============================================================

load_static_scales("outputs/static_per_head_scales.pt")


# ============================================================
# 2. patch BEFORE creating LLM
# ============================================================

apply_shadow_cache_patch(
    verbose_layer=0,
)


# ============================================================
# 3. only now import/use vLLM
# ============================================================

from vllm import (
    LLM,
    SamplingParams,
)


llm = LLM(
    model=("Qwen/Qwen2.5-7B-Instruct"),
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.75,
    enforce_eager=True,
    attention_config={
        "backend": "TRITON_ATTN",
    },
)


params = SamplingParams(
    temperature=0.0,
    max_tokens=8,
)


prompts = ["Explain KV Cache in one paragraph."]


outputs = llm.generate(
    prompts,
    params,
)


print()
print("=" * 80)
print("MODEL OUTPUT")
print("=" * 80)

for output in outputs:
    print(output.outputs[0].text)


# ============================================================
# Debug cache summary
# ============================================================

from vllm_int8.cache_manager import (
    INT8_CACHE_POOL,
)


print()
print("=" * 80)
print("INT8 SHADOW CACHE SUMMARY")
print("=" * 80)

print("num shadow layers:", len(INT8_CACHE_POOL))

for idx, (
    layer_name,
    cache,
) in enumerate(INT8_CACHE_POOL.items()):
    print(
        layer_name,
        tuple(cache.key_cache.shape),
        cache.key_cache.dtype,
    )

    if idx >= 2:
        break
