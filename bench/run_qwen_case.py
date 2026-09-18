import argparse
import json
import os
import time

import torch


# ============================================================
# 参数
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=[
        "bf16",
        "int8",
    ],
    required=True,
)

parser.add_argument(
    "--context-len",
    type=int,
    required=True,
)

parser.add_argument(
    "--new-tokens",
    type=int,
    default=128,
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
)

args = parser.parse_args()


# ============================================================
# 环境
# ============================================================

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


# ============================================================
# INT8 patch
# ============================================================

if args.mode == "int8":
    from vllm_int8.static_scales import (
        load_static_scales,
    )

    load_static_scales("outputs/static_per_head_scales.pt")

    # Cache Write
    from vllm_int8.vllm_shadow_cache_patch import (
        apply_shadow_cache_patch,
    )

    apply_shadow_cache_patch(
        verbose_layer=-1,
    )

    # INT8-only Attention
    from vllm_int8.vllm_int8_attention_patch import (
        apply_int8_attention_patch,
    )

    apply_int8_attention_patch(
        mode="int8_only",
        verbose_layer=-1,
        max_compare_calls=0,
    )


# ============================================================
# import vLLM
# ============================================================

from vllm import (
    LLM,
    SamplingParams,
)

from transformers import AutoTokenizer


MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL)


# ============================================================
# 构造指定 context length 的 prompt
# ============================================================


def build_prompt(
    target_tokens,
):
    """
    使用重复技术文本构造长 prompt。

    最终 token 数尽可能接近 target_tokens。
    """

    base = (
        "KV Cache stores key and value tensors "
        "during autoregressive large language "
        "model inference. PagedAttention manages "
        "the cache using logical and physical "
        "memory blocks. "
    )

    text = base

    while True:
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(ids) >= target_tokens:
            ids = ids[:target_tokens]

            return tokenizer.decode(ids)

        text += base


prompt = build_prompt(args.context_len)


actual_context_len = len(
    tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )
)


print("mode:", args.mode)

print("context:", actual_context_len)


# ============================================================
# LLM
# ============================================================

llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=max(
        args.context_len + args.new_tokens + 128,
        4096,
    ),
    gpu_memory_utilization=0.70,
    enforce_eager=True,
    attention_config={
        "backend": "TRITON_ATTN",
    },
)


params = SamplingParams(
    temperature=0.0,
    max_tokens=args.new_tokens,
    ignore_eos=True,
)


# ============================================================
# Warmup
#
# 很重要：
# 消掉 Triton JIT。
# ============================================================

warmup_prompt = "Explain KV cache briefly."

warmup_params = SamplingParams(
    temperature=0.0,
    max_tokens=8,
    ignore_eos=True,
)


print()
print("===== WARMUP =====")


_ = llm.generate(
    [warmup_prompt],
    warmup_params,
)


torch.cuda.synchronize()


# ============================================================
# Reset peak memory AFTER warmup
# ============================================================

torch.cuda.reset_peak_memory_stats()


# ============================================================
# 正式 Benchmark
# ============================================================

print()
print("===== BENCHMARK =====")


torch.cuda.synchronize()

start = time.perf_counter()


outputs = llm.generate(
    [prompt],
    params,
)


torch.cuda.synchronize()

end = time.perf_counter()


elapsed = end - start


# ============================================================
# token count
# ============================================================

result = outputs[0]


generated_token_ids = result.outputs[0].token_ids


num_generated = len(generated_token_ids)


tokens_per_second = num_generated / elapsed


ms_per_token = (
    elapsed
    / max(
        num_generated,
        1,
    )
    * 1000
)


# ============================================================
# Peak memory
#
# 注意：
#
# INT8 Shadow 模式仍包含 BF16 cache，
# 所以这个值不能直接作为最终 KV 节约。
# 这里只用于记录。
# ============================================================

peak_allocated = torch.cuda.max_memory_allocated()

peak_reserved = torch.cuda.max_memory_reserved()


# ============================================================
# KV Cache Memory
# ============================================================

bf16_cache_bytes = None
int8_cache_bytes = None
scale_bytes = None


if args.mode == "int8":
    from vllm_int8.cache_manager import (
        INT8_CACHE_POOL,
    )

    from vllm_int8.vllm_shadow_cache_patch import (
        NATIVE_CACHE_BYTES_BY_LAYER,
    )

    # --------------------------------------------------------
    # 原 vLLM BF16 cache 实际 bytes
    # --------------------------------------------------------

    bf16_cache_bytes = sum(NATIVE_CACHE_BYTES_BY_LAYER.values())

    # --------------------------------------------------------
    # 我们 INT8 shadow cache
    # --------------------------------------------------------

    int8_cache_bytes = 0
    scale_bytes = 0

    for cache in INT8_CACHE_POOL.values():
        int8_cache_bytes += cache.key_cache.numel() * cache.key_cache.element_size()

        int8_cache_bytes += cache.value_cache.numel() * cache.value_cache.element_size()

        scale_bytes += cache.k_scale.numel() * cache.k_scale.element_size()

        scale_bytes += cache.v_scale.numel() * cache.v_scale.element_size()


# ============================================================
# save
# ============================================================

data = {
    "mode": args.mode,
    "context_len": actual_context_len,
    "generated_tokens": num_generated,
    "elapsed_s": elapsed,
    "tokens_per_second": tokens_per_second,
    "ms_per_token": ms_per_token,
    "peak_allocated_mb": peak_allocated / 1024**2,
    "peak_reserved_mb": peak_reserved / 1024**2,
    "bf16_cache_mb": None if bf16_cache_bytes is None else bf16_cache_bytes / 1024**2,
    "int8_cache_mb": None if int8_cache_bytes is None else int8_cache_bytes / 1024**2,
    "scale_mb": None if scale_bytes is None else scale_bytes / 1024**2,
    "output_text": result.outputs[0].text,
}


with open(
    args.output,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        data,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print("=" * 80)
print("RESULT")
print("=" * 80)

for k, v in data.items():
    if k != "output_text":
        print(f"{k}: {v}")
