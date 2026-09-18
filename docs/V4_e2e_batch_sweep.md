# V4 端到端 Batch Sweep 实验报告

> 项目：`int8-kvcache`  
> 模型：Qwen2.5-7B-Instruct（`Hq=28, Hkv=4, D=128, block_size=16`）  
> 硬件：NVIDIA GeForce RTX 4090  
> 文档日期：2026-09-19  

本文记录 **V4 Kernel 接入 vLLM 后的端到端 Batch Sweep**：实验动机、attention patch 修改、sweep 设计、重跑原因，以及最终数字。Kernel 层 microbench 见 [`V4_experiment_report.md`](V4_experiment_report.md)；本节只覆盖 e2e。

---

## 1. 为什么要重做 e2e

V3 时期 README §2.4 的 Batch Sweep 里，INT8×B>1 的吞吐看起来甚至偶尔高于 BF16。事后核对发现：

`vllm_int8/vllm_int8_attention_patch.py` 里的 `_is_supported_decode` **曾经强制 batch=1**。  
因此 B>1 的 decode **不会进入** 自定义 `int8_paged_attention`，而是回退到 vLLM BF16 Attention；INT8 侧当时主要只做 shadow cache write。

V4 kernel microbench 是直接调 Triton wrapper，**不受该门控影响**，B>1 的算子结论仍然成立。但要把「V4 接到真实 vLLM decode」说清楚，必须在放宽门控后重跑 e2e。

本轮目标：

1. 让任意 batch 的纯 decode 走到 V4 / Auto INT8 kernel
2. 用计数器证明命中，而不是只看 wall-clock
3. 补齐 ctx≈2048、生成 128 tokens、B∈{1,2,4,8} 的 BF16 vs INT8 表

---

## 2. 运行环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 24GB |
| 驱动 | 570.124.04（CUDA 12.8） |
| Python | 3.12（项目 `.venv`） |
| PyTorch | 2.9.1+cu128 |
| vLLM | **0.16.0** |
| Attention | `TRITON_ATTN`，`enforce_eager=True` |
| 模型 | 本地 `Qwen/Qwen2.5-7B-Instruct` |
| Scale | `outputs/static_per_head_scales.pt` |

本机驱动是 CUDA 12.8。vLLM 0.26.0 的预编译扩展依赖 CUDA 13，无法在此驱动上 import，因此 e2e 使用与 torch 2.9.1 匹配的 **vLLM 0.16.0**。KV 原生 layout 为 `[num_blocks, 2, block_size, Hkv, D]`。

完整环境快照：[`outputs/batch_sweep/env.txt`](../outputs/batch_sweep/env.txt)。

---

## 3. Attention patch 修改（`vllm_int8_attention_patch.py`）

文件：[`vllm_int8/vllm_int8_attention_patch.py`](../vllm_int8/vllm_int8_attention_patch.py)。  
这是 vLLM Triton Attention 的 monkey patch（用户口中的 INT8 paged-attn patch），decode 命中后调用 `int8_paged_attention(..., impl="auto")`。

### 3.1 放宽 `_is_supported_decode`（上一轮，本轮 e2e 的前提）

**旧逻辑**：`query / seq_lens / block_table` 的 batch 维必须全部等于 1。  
**新逻辑**：任意 B 的纯 decode 均可进入 INT8：

- `seq_lens.numel() == B`，且 `block_table.shape[0] == B`
- 本步 query token 数等于 B（一个请求一个新 token）
- 若 metadata 带 `max_query_len`，则必须 `== 1`（排除 prefill / chunked-prefill）

Prefill 仍走 vLLM 原生 BF16；decode 才 takeover。

### 3.2 兼容 vLLM 0.16 metadata

| 改动 | 原因 |
|---|---|
| `_get_block_table` 同时认 `block_table` / `block_tables` | 不同版本字段名不一致 |
| 用 `num_actual_tokens` 判断 decode，而不是裸 `query.shape[0]` | 0.16 可能 pad query；pad 后 `query.shape[0] != B` 会误判为非 decode |
| `_run_int8_attention` 对 query 做 `query[:num_actual_tokens]` | kernel 吃真实 decode token，不吃 padding |
| `output[:n].copy_(...)` | 只写回有效 token，避免和 padded output 对不齐 |

### 3.3 命中计数（证明 B>1 真的跑进算子）

新增：

- `ATTN_INT8_HITS[layer_name]`：每层调用 `int8_paged_attention` 的次数
- `ATTN_INT8_STATS["last_batch"]` / `["max_batch"]`：观测到的 decode batch

`bench/run_qwen_case_final.py` 在 INT8 跑完后写入 JSON，并 **硬失败**：

- `int8_decode_hits <= 0` → 没打上 Triton patch / 没走 INT8
- `max_batch != 请求 batch` → 调度切碎了 batch，或门控仍在挡 B>1

最终每条 INT8 行：`hits=3556`（28 层 × 约 127 步），`int8_last_batch` 等于请求 B。

### 3.4 `int8_only` 热路径去掉强制 NaN/Inf 检查

原先每层每步 `torch.isnan(int8_out).any()` / `isinf` 会 **强制 CUDA sync**。  
sweep 热路径改为仅在 `verbose_layer >= 0` 时检查（bench 传 `verbose_layer=-1`）。  
正确性仍由 kernel 单测和 takeover/shadow compare 模式覆盖。

---

## 4. 其它为 e2e 做的代码改动

### 4.1 Shadow cache layout 推断

[`vllm_int8/vllm_shadow_cache_patch.py`](../vllm_int8/vllm_shadow_cache_patch.py) 增加 `_infer_native_page_dims`，不再写死 `kv_cache.shape[2]` 为 `block_size`：

| vLLM | 原生 KV shape |
|---|---|
| 0.26 fused | `[num_blocks, Hkv, block_size, 2*D]` |
| **0.16 split（本次）** | `[num_blocks, 2, block_size, Hkv, D]` |
| 更旧 split | `[2, num_blocks, block_size, Hkv, D]` |

写入路径仍是：vLLM 写 BF16 native cache + 镜像一份 INT8 shadow。

### 4.2 Bench harness

[`bench/run_qwen_case_final.py`](../bench/run_qwen_case_final.py) / [`bench/benchmark_batch_sweep.py`](../bench/benchmark_batch_sweep.py)：

- `attention_config={"backend": "TRITON_ATTN"}`，并打印实际 backend
- `max_model_len = context + new_tokens + 64`（不再强行 4096）
- `gpu_memory_utilization=0.80`，`max_num_seqs=16`
- `enable_prefix_caching=False`（避免 warmup 把 prefill 从计时里「偷走」）
- warmup 使用 **与正式测量相同的 prompts / batch**，warmup 后清零 INT8 计数
- summary 增加 `int8_decode_hits`、`int8_last_batch`

---

## 5. 实验方案

命令：

```bash
cd /root/autodl-tmp/int8-kvcache
source .venv/bin/activate
export PYTHONPATH=.
python bench/benchmark_batch_sweep.py
```

固定参数：

| 参数 | 值 |
|---|---|
| Context 目标 | 2048（tokenizer 往返实际 **1973**） |
| 生成 tokens | 128，`temperature=0`，`ignore_eos=True` |
| Batch | 1, 2, 4, 8 |
| Mode | `bf16`（无 patch）vs `int8`（shadow write + `int8_only` decode） |
| INT8 kernel | `int8_paged_attention(..., impl="auto")` |
| 计时 | warmup 8 tokens 之后的 `generate` wall-clock（含该次请求的 prefill） |

每个 `(mode, batch)` 是独立进程，避免 patch 全局状态串扰。原始 JSON 在 [`outputs/batch_sweep/`](../outputs/batch_sweep/)。

冒烟：INT8、ctx=128、8 tokens、B=2 → backend=`TRITON_ATTN`，`int8_last_batch=2`，hits>0。随后进入全量 sweep。

---

## 6. 意外发现与重跑

正式表之前 sweep 跑了多轮。下面只记 **改变测量口径或使数字无效** 的发现。

### 6.1 INT8 在 sampler warmup 时 OOM（第 1 轮作废）

先把 `gpu_memory_utilization` 提到 0.90，想给 B=8 留足 KV。BF16 B=1 能跑；INT8 在 vLLM dummy sampler warmup（默认按很大的 `max_num_seqs`）时 OOM。

原因：0.90 会分配约 5.4 GiB **原生 BF16 KV**；INT8 路径还要再分配同等块数的 shadow（约一半容量）。模型约 14.3 GiB，再加上 warmup logits，24GB 卡顶满。

处理：`gpu_memory_utilization=0.80` + `max_num_seqs=16`。可用 KV 约 3.4 GiB，对 B=8 × 2240 足够，且给 shadow 留出空间。该轮数字全部丢弃。

### 6.2 Warmup 形状与计时形状不一致，autotune 打进正式时间（第 2 轮部分作废）

第 2 轮 INT8 已能跑通，且 hits 证明 B>1 命中算子，但数字异常：

- INT8 B=1：约 4.1 s（相对 BF16 合理）
- INT8 B=2/4/8：约 **23–24 s**（逐步时延约 180 ms，是 B=1 的 5–6 倍）

Kernel microbench 里 B=8 / seq=2048 的 Auto 只有 ~0.23 ms/层，不可能单独解释 e2e 的 10×。

原因：warmup 用的是 **B=1 短句** `"Explain KV cache briefly."`，正式测量才是 B∈{2,4,8}、ctx≈1973。Triton autotune 按 `max_num_blocks` 等 key 编译/选 config，**第一次真正的长上下文大 batch launch 发生在计时区间内**。B>1 要搜索的 config 更贵，于是 23 s 里混进了大量 compile。

处理：warmup 改为 `llm.generate(prompts, ...)`，与正式请求同 batch、同 context；warmup 结束后再清零 hits。同时关掉 prefix cache，避免 warmup 把 1973 token 的 prefill 缓存掉、让正式 generate 变成「纯 decode」、和历史口径不一致。

### 6.3 热路径 NaN/Inf 检查（第 3 轮后的微调）

怀疑 `isnan/isinf` 的 GPU reduction 在 B>1 上放大同步开销。关掉热路径检查后再跑一轮：

- INT8 B=1：3.62 s → **3.20 s**（有改善）
- INT8 B=2/4/8：仍约 11–12 s（相对第 2 轮已去掉 autotune，但 B>1 仍明显慢于 BF16）

结论：NaN 检查不是 B>1 主因；主因是 **shadow 双写 + Python patch 每层每步开销**，相对 vLLM 已融合的 Triton unified attention。第 3 轮口径（同形状 warmup、关 prefix cache）与最终表一致，最终表采用关掉热路径 NaN 检查后的第 4 轮。

---

## 7. 最终结果

日期：2026-09-19。Context 实际 1973，生成 128 tokens。

| Mode | Batch | Time (s) | Throughput (tok/s) | Peak MB | INT8 hits | INT8 B |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1 | 2.447 | 52.31 | 18459.63 | — | — |
| INT8 | 1 | 3.204 | 39.95 | 20251.66 | 3556 | 1 |
| BF16 | 2 | 2.444 | 104.74 | 18727.58 | — | — |
| INT8 | 2 | 11.316 | 22.62 | 20519.61 | 3556 | 2 |
| BF16 | 4 | 2.851 | 179.60 | 19263.43 | — | — |
| INT8 | 4 | 11.805 | 43.37 | 21055.45 | 3556 | 4 |
| BF16 | 8 | 3.731 | 274.47 | 19303.11 | — | — |
| INT8 | 8 | 12.395 | 82.61 | 21095.13 | 3556 | 8 |

相对 vLLM 原生 BF16 的 wall-clock：

| B | INT8 / BF16 time | INT8 tok/s | BF16 tok/s |
|--:|------------------:|-----------:|-----------:|
| 1 | 1.31× | 39.95 | 52.31 |
| 2 | 4.63× | 22.62 | 104.74 |
| 4 | 4.14× | 43.37 | 179.60 |
| 8 | 3.32× | 82.61 | 274.47 |

INT8 侧 cache 体积（shadow 统计，各 batch 相同，因为按 engine 的 full KV pool 分配）：

- Native BF16 KV：3525.4 MB
- INT8 payload：1762.7 MB（约一半）
- Scale：可忽略

Peak allocated 仍是 INT8 更高（约 +1.8 GB），因为 **native BF16 没有拆掉**。

---

## 8. 解读

1. **口径已经正确。** B=1/2/4/8 的 INT8 decode 都进入了 `int8_paged_attention`（`impl="auto"`）。旧 V3 表里 B>1 INT8「更快」是门控回退造成的假象，不能再用。
2. **Kernel 赢 ≠ e2e 赢。** V4 microbench 在长 seq / 大 B 已快于自研 BF16，甚至相对 V3 接近 2×。e2e 对比的是 vLLM 高度优化的 Triton unified attention，再加上每步 BF16 native write + INT8 shadow write，以及 28 层 Python `forward` patch。
3. **B=1 差距温和（1.31×），B≥2 拉大到 3–5×。** 逐步时间 INT8 在 B≥2 时几乎不随 batch 摊薄（11–12 s 生成 128 step），说明系统层固定开销和双写占主导，而不是单纯「kernel 随 B 变慢」。Auto 在 ctx≈1973、B≥2 时会切到 V4（`work=B*max_blocks ≥ 192`），但 e2e 没有把 kernel 优势表现出来。
4. **不要和 V3 历史表比绝对值。** vLLM 0.26→0.16、eager、`gpu_memory_utilization`、prefix cache、warmup 形状全部不同。只比「这次 INT8 是否真的跑到自定义算子」以及「同一次 sweep 内 BF16 vs INT8」。

下一步若不在 V4 kernel 范围：去掉 BF16 shadow、做 native INT8 allocator，并把 INT8 attention 从 Python patch 收进 worker 热路径。

---

## 9. 复现

```bash
cd /root/autodl-tmp/int8-kvcache
source .venv/bin/activate
export PYTHONPATH=.
export HF_HUB_OFFLINE=1

# 冒烟（确认 TRITON_ATTN + INT8 hits）
python bench/run_qwen_case_final.py \
  --mode int8 --context-len 128 --new-tokens 8 --batch-size 2 \
  --output /tmp/int8_smoke_b2.json

# 全量 Batch Sweep
python bench/benchmark_batch_sweep.py
```

产物：

| 路径 | 内容 |
|---|---|
| `outputs/batch_sweep/summary.json` | 汇总表 |
| `outputs/batch_sweep/{bf16,int8}_b{1,2,4,8}.json` | 单次原始结果（含 hits） |
| `outputs/batch_sweep/env.txt` | 环境快照 |
| `outputs/batch_sweep/v4_e2e_run.log` | 完整日志 |
