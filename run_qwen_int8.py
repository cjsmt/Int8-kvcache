import os

# 必须在 import vllm / apply patch 之前关闭多进程，否则 hook 打不到 EngineCore
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm_int8.vllm_int8_patch import apply_vllm_int8_patch

# 必须先 patch
apply_vllm_int8_patch()

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
)

params = SamplingParams(
    temperature=0,
    max_tokens=32,
)

outputs = llm.generate(
    ["Explain attention mechanism."],
    params,
)

for x in outputs:
    print(x.outputs[0].text)
