import os
# Monkey patch 调试时关闭 V1 multiprocessing
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# 1. Load Static Per-Head Scale
from vllm_int8.static_scales import (
    load_static_scales,
)
load_static_scales("outputs/static_per_head_scales.pt")
# 2. Step 3:；Shadow Cache Write
from vllm_int8.vllm_shadow_cache_patch import (
    apply_shadow_cache_patch,
)
apply_shadow_cache_patch(
    verbose_layer=0,
)
# 3. Step 4:；Shadow INT8 Attention Read；必须在 LLM 创建前 patch。
from vllm_int8.vllm_int8_attention_patch import (
    apply_shadow_attention_patch,
)
apply_shadow_attention_patch(
    verbose_layer=0,
    max_compare_calls=8,
)
# 4. 启动 vLLM
from vllm import (
    LLM,
    SamplingParams,
)
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.70,
    enforce_eager=True,
    attention_config={
        "backend": "TRITON_ATTN",
    },
)
# 5. 第一轮只生成 4 token；足够观察连续 decode。
params = SamplingParams(
    temperature=0.0,
    max_tokens=4,
)
prompts = ["Explain KV Cache briefly."]
outputs = llm.generate(
    prompts,
    params,
)
print()
print("=" * 80)
print("MODEL OUTPUT")
print("=" * 80)
for item in outputs:
    print(item.outputs[0].text)

# 6. Summary
from vllm_int8.cache_manager import (
    INT8_CACHE_POOL,
)
from vllm_int8.vllm_int8_attention_patch import (
    ATTN_COMPARE_COUNT,
)
print()
print("=" * 80)
print("STEP 4 SUMMARY")
print("=" * 80)
print(
    "num shadow cache layers:",
    len(INT8_CACHE_POOL),
)
print(
    "num attention compare layers:",
    len(ATTN_COMPARE_COUNT),
)
if "model.layers.0.self_attn.attn" in ATTN_COMPARE_COUNT:
    print(
        "layer0 compare calls:",
        ATTN_COMPARE_COUNT["model.layers.0.self_attn.attn"],
    )
