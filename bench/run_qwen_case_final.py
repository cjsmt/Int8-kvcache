import argparse
import json
import os
import time
import torch
# Args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["bf16", "int8"],
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
    required=True,
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
)
args = parser.parse_args()
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
# INT8 patch
if args.mode == "int8":
    from vllm_int8.static_scales import (
        load_static_scales,
    )
    load_static_scales("outputs/static_per_head_scales.pt")
    from vllm_int8.vllm_shadow_cache_patch import (
        apply_shadow_cache_patch,
    )
    apply_shadow_cache_patch(
        verbose_layer=-1,
    )
    from vllm_int8.vllm_int8_attention_patch import (
        apply_int8_attention_patch,
    )
    apply_int8_attention_patch(
        mode="int8_only",
        verbose_layer=-1,
        max_compare_calls=0,
    )
from transformers import AutoTokenizer
from vllm import (
    LLM,
    SamplingParams,
)
# Prompt
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def build_prompt(
    target_tokens,
):
    base = (
        "KV cache stores key and value tensors "
        "during autoregressive transformer inference. "
        "PagedAttention manages logical blocks and "
        "physical memory blocks efficiently. "
    )
    ids = []
    base_ids = tokenizer.encode(
        base,
        add_special_tokens=False,
    )
    while len(ids) < target_tokens:
        ids.extend(base_ids)
    ids = ids[:target_tokens]
    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
    )
single_prompt = build_prompt(args.context_len)
prompts = [single_prompt for _ in range(args.batch_size)]
actual_context_len = len(
    tokenizer.encode(
        single_prompt,
        add_special_tokens=False,
    )
)
# Engine
max_model_len = args.context_len + args.new_tokens + 64
llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=max_model_len,
    gpu_memory_utilization=0.80,
    max_num_seqs=16,
    enforce_eager=True,
    enable_prefix_caching=False,
    attention_config={
        "backend": "TRITON_ATTN",
    },
)
try:
    backend = llm.llm_engine.vllm_config.attention_config.backend
    print("[bench] attention backend:", backend)
except Exception as exc:
    print("[bench] could not read attention backend:", exc)
# Warmup (same batch/context so Triton autotune is not billed to the timed run)
warmup_params = SamplingParams(
    temperature=0.0,
    max_tokens=8,
    ignore_eos=True,
)
_ = llm.generate(
    prompts,
    warmup_params,
)
torch.cuda.synchronize()
if args.mode == "int8":
    from vllm_int8.vllm_int8_attention_patch import (
        ATTN_INT8_HITS,
        ATTN_INT8_STATS,
    )
    from vllm_int8.vllm_shadow_cache_patch import (
        WRITE_CALL_COUNT,
    )
    ATTN_INT8_HITS.clear()
    ATTN_INT8_STATS["last_batch"] = 0
    ATTN_INT8_STATS["max_batch"] = 0
    WRITE_CALL_COUNT.clear()
# Benchmark
params = SamplingParams(
    temperature=0.0,
    max_tokens=args.new_tokens,
    ignore_eos=True,
)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
outputs = llm.generate(
    prompts,
    params,
)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
# Tokens
generated_per_request = [len(item.outputs[0].token_ids) for item in outputs]
total_generated = sum(generated_per_request)
overall_tok_s = total_generated / elapsed
request_tok_s = args.new_tokens / elapsed
avg_ms_per_generated_token = (
    elapsed
    / max(
        total_generated,
        1,
    )
    * 1000
)
# GPU Memory
peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
# KV payload
bf16_cache_mb = None
int8_cache_mb = None
scale_mb = None
int8_decode_hits = None
int8_write_calls = None
int8_decode_layers = None
int8_last_batch = None
if args.mode == "int8":
    from vllm_int8.cache_manager import (
        INT8_CACHE_POOL,
    )
    from vllm_int8.vllm_shadow_cache_patch import (
        NATIVE_CACHE_BYTES_BY_LAYER,
    )
    native_bytes = sum(NATIVE_CACHE_BYTES_BY_LAYER.values())
    int8_bytes = 0
    scale_bytes = 0
    for cache in INT8_CACHE_POOL.values():
        int8_bytes += cache.key_cache.numel() * cache.key_cache.element_size()
        int8_bytes += cache.value_cache.numel() * cache.value_cache.element_size()
        scale_bytes += cache.k_scale.numel() * cache.k_scale.element_size()
        scale_bytes += cache.v_scale.numel() * cache.v_scale.element_size()
    bf16_cache_mb = native_bytes / 1024**2
    int8_cache_mb = int8_bytes / 1024**2
    scale_mb = scale_bytes / 1024**2
    from vllm_int8.vllm_int8_attention_patch import (
        ATTN_INT8_HITS,
        ATTN_INT8_STATS,
    )
    from vllm_int8.vllm_shadow_cache_patch import (
        WRITE_CALL_COUNT,
    )
    int8_decode_hits = int(sum(ATTN_INT8_HITS.values()))
    int8_write_calls = int(sum(WRITE_CALL_COUNT.values()))
    int8_decode_layers = int(len(ATTN_INT8_HITS))
    int8_last_batch = int(ATTN_INT8_STATS.get("max_batch", 0) or ATTN_INT8_STATS.get("last_batch", 0))
    if int8_decode_hits <= 0:
        raise RuntimeError(
            "INT8 decode path was never hit; TritonAttentionImpl.forward "
            "did not run int8_paged_attention. Check attention backend."
        )
    if int8_last_batch != args.batch_size:
        raise RuntimeError(
            f"INT8 last decode batch={int8_last_batch} != "
            f"requested batch={args.batch_size}"
        )

# Output
result = {
    "mode": args.mode,
    "batch_size": args.batch_size,
    "context_len": actual_context_len,
    "new_tokens": args.new_tokens,
    "total_generated_tokens": total_generated,
    "elapsed_s": elapsed,
    "overall_tokens_per_second": overall_tok_s,
    "avg_ms_per_generated_token": avg_ms_per_generated_token,
    "peak_allocated_mb": peak_allocated_mb,
    "peak_reserved_mb": peak_reserved_mb,
    "bf16_cache_mb": bf16_cache_mb,
    "int8_cache_mb": int8_cache_mb,
    "scale_mb": scale_mb,
    "int8_decode_hits": int8_decode_hits,
    "int8_write_calls": int8_write_calls,
    "int8_decode_layers": int8_decode_layers,
    "int8_last_batch": int8_last_batch,
    "token_ids": [list(item.outputs[0].token_ids) for item in outputs],
    "texts": [item.outputs[0].text for item in outputs],
}
os.makedirs(
    os.path.dirname(args.output),
    exist_ok=True,
)
with open(
    args.output,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False,
    )
print(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )
)
