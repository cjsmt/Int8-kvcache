import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT = "outputs/kv_sample.pt"
os.makedirs("outputs", exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()
texts = [
    "请介绍一下Transformer中的注意力机制。",
    "为什么大语言模型推理时需要KV Cache？",
    "Explain paged attention in simple words.",
    "Write a short paragraph about GPU memory bandwidth.",
]
batch = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
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
saved = {}
for layer_idx, layer_kv in enumerate(layer_items):
    if isinstance(layer_kv, (tuple, list)):
        k, v = layer_kv[0], layer_kv[1]
    else:
        k, v = layer_kv.keys, layer_kv.values

    # 常见 shape:；k/v: [batch, num_kv_heads, seq_len, head_dim]
    print(
        f"layer={layer_idx:02d}",
        "K", tuple(k.shape),
        "V", tuple(v.shape),
        "K range", float(k.min()), float(k.max()),
        "V range", float(v.min()), float(v.max()),
    )

    # 为节省磁盘，仅保存几个有代表性的 layer
    if layer_idx in [0, 7, 14, 21, 27]:
        saved[layer_idx] = {
            "k": k.detach().cpu().to(torch.float16),
            "v": v.detach().cpu().to(torch.float16),
        }
torch.save(saved, OUT)
print("saved to", OUT)
