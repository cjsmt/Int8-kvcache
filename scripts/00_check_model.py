import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

messages = [
    {"role": "user", "content": "用一句话解释什么是KV Cache。"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=32)

print(tokenizer.decode(output[0], skip_special_tokens=True))