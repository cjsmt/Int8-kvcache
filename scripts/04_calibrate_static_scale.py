import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"

PROMPTS = [
    "请解释Transformer中的KV Cache。",
    "请介绍FlashAttention的核心思想。",
    "为什么decode阶段容易受到显存带宽限制？",
    "请比较GQA和MHA。",
    "解释PagedAttention。",
    "什么是模型量化？",
    "Explain how attention works.",
    "Explain GPU memory hierarchy.",
    "Write a short introduction to LLM inference.",
    "What is the difference between prefill and decode?",
] * 4

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

L = model.config.num_hidden_layers
Hkv = model.config.num_key_value_heads

k_amax = torch.zeros(L, Hkv, dtype=torch.float32)
v_amax = torch.zeros_like(k_amax)

for i, text in enumerate(PROMPTS):
    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to("cuda")

    with torch.inference_mode():
        out = model(**batch, use_cache=True)

    pkv = out.past_key_values
    # 兼容 DynamicCache / legacy tuple
    if hasattr(pkv, "layers"):
        layer_items = [(layer.keys, layer.values) for layer in pkv.layers]
    elif hasattr(pkv, "to_legacy_cache"):
        layer_items = pkv.to_legacy_cache()
    else:
        layer_items = pkv

    for l, layer_kv in enumerate(layer_items):
        if isinstance(layer_kv, (tuple, list)):
            k, v = layer_kv[0], layer_kv[1]
        else:
            k, v = layer_kv.keys, layer_kv.values
        # [B,Hkv,T,D] -> [Hkv]
        ka = k.float().abs().amax(dim=(0, 2, 3)).cpu()
        va = v.float().abs().amax(dim=(0, 2, 3)).cpu()

        k_amax[l] = torch.maximum(k_amax[l], ka)
        v_amax[l] = torch.maximum(v_amax[l], va)

    print(f"[{i+1}/{len(PROMPTS)}] done")

k_scale = (k_amax / 127.0).clamp_min(1e-6)
v_scale = (v_amax / 127.0).clamp_min(1e-6)

torch.save(
    {
        "k_scale": k_scale,
        "v_scale": v_scale,
        "k_amax": k_amax,
        "v_amax": v_amax,
    },
    "outputs/static_per_head_scales.pt",
)

print("K:", k_scale.shape)
print("V:", v_scale.shape)
print("Layer0 K:", k_scale[0])
print("Layer0 V:", v_scale[0])