import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# 1. Static Scale
from vllm_int8.static_scales import (
    load_static_scales,
)
load_static_scales("outputs/static_per_head_scales.pt")
# 2. INT8 Cache Write
from vllm_int8.vllm_shadow_cache_patch import (
    apply_shadow_cache_patch,
)
apply_shadow_cache_patch(
    verbose_layer=0,
)
# 3. INT8 Attention TAKEOVER
from vllm_int8.vllm_int8_attention_patch import (
    apply_int8_attention_patch,
)
apply_int8_attention_patch(
    mode="int8_only",
    verbose_layer=0,
    max_compare_calls=8,
)
# 4. vLLM
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
# 5. 第一轮只生成一个 token
params = SamplingParams(
    temperature=0.0,
    max_tokens=32,
)
outputs = llm.generate(
    ["Explain KV Cache briefly."],
    params,
)
print()
print("=" * 80)
print("INT8 TAKEOVER OUTPUT")
print("=" * 80)
for item in outputs:
    print(repr(item.outputs[0].text))
from vllm_int8.vllm_int8_attention_patch import (
    ATTN_METRICS,
)
print()
print("=" * 80)
print("PER-LAYER ATTENTION ERROR")
print("=" * 80)
for layer_name, records in sorted(
    ATTN_METRICS.items()
):
    if not records:
        continue
    avg_cos = sum(
        x["cosine"]
        for x in records
    ) / len(records)
    avg_l2 = sum(
        x["rel_l2"]
        for x in records
    ) / len(records)
    avg_mae = sum(
        x["mae"]
        for x in records
    ) / len(records)
    print(
        f"{layer_name:40s}"
        f" cos={avg_cos:.6f}"
        f" rel_l2={avg_l2:.6f}"
        f" mae={avg_mae:.6f}"
    )
