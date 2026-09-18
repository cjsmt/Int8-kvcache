# INT8 KV Cache × PagedAttention × vLLM

> 基于 Triton 实现 INT8 KV Cache Write 与 INT8 PagedAttention，并接入 vLLM 0.26.0 / Qwen2.5-7B-Instruct 的 Decode 推理路径，完成正确性、KV Cache 存储压缩、Context Sweep 与 Batch Sweep 验证。

---

## 1. 项目简介

本项目面向大语言模型自回归 Decode 阶段的 KV Cache 显存与访存开销，围绕 **Qwen2.5-7B-Instruct + vLLM 0.26.0** 完成了一套 Static Per-Head INT8 KV Cache 原型。

项目主要完成了以下工作：

1. 采集 Qwen2.5-7B 各层 K/V 数据并分析数值分布；
2. 实现 Static Per-Head 对称 INT8 KV 量化与 scale calibration；
3. 基于 Triton 自研 `INT8 KV Cache Write` Kernel；
4. 基于 Triton 自研 `INT8 PagedAttention` Kernel；
5. 实现 BF16 Triton PagedAttention baseline；
6. 完成 `slot_mapping`、Paged KV Cache、Block Table、GQA 28→4 Head 映射与 Online Softmax；
7. 将 INT8 Cache Write 接入 vLLM 真实 Prefill / Decode KV Cache Write 路径；
8. 将 INT8 PagedAttention 接入 vLLM Decode Attention 路径；
9. 在 Qwen2.5-7B 的 28 层 Attention 上进行 BF16 / INT8 Shadow Compare；
10. 完成 `shadow → takeover → int8_only` 三阶段运行验证；
11. 完成 Context Length、Decode TPOT、Batch Size Sweep 和 KV Cache payload 统计。

当前 V3 版本的重点是：

> **完成从量化策略、Triton Kernel、vLLM Runtime 集成到 Qwen2.5-7B 端到端验证的完整闭环。**

Kernel 的进一步性能优化暂不属于 V3 范围，计划作为后续 V4 工作。

---

## 2. 当前结果

### 2.1 正确性

在 vLLM 真实 Decode Runtime 中，INT8 PagedAttention 与 vLLM BF16 Attention 在 Qwen2.5-7B 的 28 层 Attention 上进行了逐层对比：

```text
INT8 / BF16 Attention cosine ≈ 0.999
```

同时：

- `takeover / 32 tokens` 可正常生成；
- `int8_only / 32 tokens` 可正常生成；
- `int8_only` 与 `takeover` 的测试输出 token 可保持一致；
- 未观察到 NaN / Inf。

### 2.2 KV Cache Payload

实测缓存 tensor payload：

```text
BF16 KV Cache : 1004.50 MB
INT8 KV Cache :  502.25 MB

Reduction     : ≈ 50.00%
```

这里的 50% 指的是：

> **KV Cache payload / tensor storage 大小减少约 50%。**

注意：当前 vLLM 集成采用 Shadow INT8 Cache 方案，原生 BF16 KV Cache 仍然由 vLLM 分配，因此 **当前进程的 GPU 总显存并没有真正下降 50%**。

要实现真实 GPU persistent KV memory reduction，还需要进一步改造 vLLM KV Cache allocator / native KV dtype，这部分未纳入当前 V3。

### 2.3 Decode TPOT

#### V3（correctness-first）

| Context | BF16 TPOT (ms) | INT8 TPOT (ms) | BF16 Decode tok/s | INT8 Decode tok/s |
|---:|---:|---:|---:|---:|
| 512  | 16.851 | 25.785 | 59.34 | 38.78 |
| 1024 | 17.200 | 25.519 | 58.14 | 39.19 |
| 2048 | 16.414 | 28.735 | 60.92 | 34.80 |
| 4096 | 16.578 | 33.596 | 60.32 | 29.77 |

#### V4 Kernel Microbench（RTX 4090，Qwen2.5-7B 形状）

单次 Attention kernel 延迟（ms），见 `outputs/v4_baseline/microbench.csv`：

| B | Seq | BF16 | INT8 V3 | INT8 V4 | INT8 Auto | V3/V4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 0.055 | 0.132 | 0.296 | 0.125 | 0.45× |
| 1 | 1024 | 0.080 | 0.166 | 0.227 | 0.168 | 0.73× |
| 1 | 2048 | 0.157 | 0.240 | 0.227 | 0.241 | 1.06× |
| 1 | 4096 | 0.312 | 0.400 | 0.266 | **0.266** | **1.50×** |
| 8 | 4096 | 0.463 | 0.525 | 0.266 | **0.258** | **1.97×** |

V4 结论：

> **长上下文与大 batch 下，V4 INT8 kernel 已超过自研 BF16 baseline，并对 V3 有明显加速；短上下文仍由 occupancy 主导，默认 `impl="auto"` 自动回退 V3。**

端到端 vLLM TPOT 复测命令：`bench/benchmark_decode_tpot.py`（需可用的 CUDA PyTorch + vLLM 环境）。

### 2.4 Batch Sweep

Context=2048、生成 128 tokens 时：

| Mode | Batch | Time (s) | Throughput (tok/s) | Peak MB |
|---|---:|---:|---:|---:|
| BF16 | 1 | 2.368 | 54.06 | 16090.08 |
| INT8 | 1 | 3.789 | 33.78 | 16592.36 |
| BF16 | 2 | 3.457 | 74.05 | 16090.76 |
| INT8 | 2 | 2.705 | 94.62 | 16593.03 |
| BF16 | 4 | 2.462 | 208.00 | 16092.11 |
| INT8 | 4 | 2.720 | 188.24 | 16594.39 |
| BF16 | 8 | 2.499 | 409.75 | 16094.83 |
| INT8 | 8 | 2.800 | 365.73 | 16597.11 |

Batch=2 的单次结果出现 INT8 throughput 更高，但该点不作为最终性能结论；如需正式发布性能数据，应进行多轮重复并取 median / percentile。

---

## 3. 技术路线

整体数据流：

```text
                         Qwen2.5-7B
                              │
                              ▼
                        vLLM 0.26.0
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
                Prefill              Decode
                    │                   │
                    ▼                   ▼
                new K/V              new K/V
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    INT8 KV Cache Write
                       Triton Kernel
                              │
                              ▼
                      INT8 Paged KV Cache
                              │
                              ▼
                    INT8 PagedAttention
                       Triton Kernel
                              │
                              ▼
                      Attention Output
                              │
                              ▼
                         Next Token
```

V3 采用：

```text
Quantization:
Static Per-Head
Symmetric INT8
Range = [-127, 127]

Scale:
one K scale / KV head / layer
one V scale / KV head / layer
```

Qwen2.5-7B：

```text
num_layers      = 28
num_q_heads     = 28
num_kv_heads    = 4
head_dim        = 128
GQA group size  = 7
block_size      = 16
```

---

## 4. 核心 Kernel

### 4.1 INT8 KV Cache Write

输入：

```text
K/V:
[num_tokens, Hkv, D]
BF16

slot_mapping:
[num_tokens]

scale:
[Hkv]
```

输出：

```text
K Cache:
[num_blocks, block_size, Hkv, D]
INT8

V Cache:
[num_blocks, block_size, Hkv, D]
INT8
```

核心地址映射：

```text
slot
  │
  ├── physical_block = slot // block_size
  │
  └── offset         = slot % block_size
```

量化：

```text
q = round(x / scale)
q = clamp(q, -127, 127)
```

### 4.2 INT8 PagedAttention

Decode 阶段：

```text
Query
  │
  ▼
GQA Q Head → KV Head
  │
  ▼
Logical Block
  │
  ▼
Block Table
  │
  ▼
Physical Block
  │
  ▼
Load INT8 K
  │
  ▼
Dequant + QK
  │
  ▼
Online Softmax
  │
  ▼
Load INT8 V
  │
  ▼
Dequant + PV
  │
  ▼
Output
```

当前 V3 的重点是 **INT8 storage + fused dequantized attention compute**，并没有实现完整 INT8 Tensor Core MMA。

---

## 5. 项目目录

```text
int8-kvcache/
├── bench/                                  # 性能与系统 Benchmark
│   ├── bench_attention.py                  # 基础 Attention microbenchmark，用于早期量化 Attention 性能验证
│   ├── bench_paged_attention.py            # BF16 / INT8 PagedAttention 独立 Kernel benchmark
│   ├── benchmark_batch_sweep.py            # 固定 Context 下 Batch=1/2/4/8 的吞吐与峰值显存测试
│   ├── benchmark_decode_tpot.py             # 利用 T128-T1 近似拆分 Decode TPOT 的测试脚本
│   ├── benchmark_qwen_in8.py                # Qwen2.5-7B BF16 / INT8 Context Sweep 总 Benchmark 驱动脚本
│   ├── run_qwen_case.py                     # 早期单个 Qwen benchmark case 执行脚本
│   └── run_qwen_case_final.py               # 最终统一 Qwen benchmark worker，支持 BF16 / INT8 / Context / Batch
│
├── plots/                                  # KV 数据分布分析结果
│   ├── layer_0_k.png                       # Layer 0 K 分布可视化
│   ├── layer_0_v.png                       # Layer 0 V 分布可视化
│   ├── layer_7_k.png                       # Layer 7 K 分布可视化
│   ├── layer_7_v.png                       # Layer 7 V 分布可视化
│   ├── layer_14_k.png                      # Layer 14 K 分布可视化
│   ├── layer_14_v.png                      # Layer 14 V 分布可视化
│   ├── layer_21_k.png                      # Layer 21 K 分布可视化
│   ├── layer_21_v.png                      # Layer 21 V 分布可视化
│   ├── layer_27_k.png                      # Layer 27 K 分布可视化
│   └── layer_27_v.png                      # Layer 27 V 分布可视化
│
├── Qwen/
│   └── Qwen2.5-7B-Instruct/                # 本地 Qwen2.5-7B-Instruct 模型目录
│       ├── config.json                      # HuggingFace 模型结构配置
│       ├── configuration.json               # 模型相关配置文件
│       ├── generation_config.json           # generation 默认参数配置
│       ├── model-00001-of-00004.safetensors # 模型权重 shard 1
│       ├── model-00002-of-00004.safetensors # 模型权重 shard 2
│       ├── model-00003-of-00004.safetensors # 模型权重 shard 3
│       ├── model-00004-of-00004.safetensors # 模型权重 shard 4
│       ├── model.safetensors.index.json     # 权重 shard 索引
│       ├── tokenizer*.json                  # Tokenizer 配置与词表相关文件
│       ├── merges.txt                       # BPE merge 规则
│       ├── vocab.json                       # Tokenizer vocabulary
│       ├── LICENSE                          # Qwen 模型 License
│       └── README.md                        # Qwen 模型原始说明
│
├── scripts/                                # 项目实验流水线，建议按 00 → 09 顺序执行
│   ├── 00_check_model.py                    # 检查 Qwen 模型路径、config、dtype、head 数等基本信息
│   ├── 01_collect_kv.py                     # 运行 Qwen 并采集真实 K/V Tensor，用于后续分布分析
│   ├── 02_analyze_kv.py                     # 分析不同 Layer / Head 的 K/V 数值分布并生成 plots
│   ├── 03_accuracy.py                       # PyTorch 量化 Reference 的 Attention 精度测试
│   ├── 04_calibrate_static_scale.py         # 根据采集 K/V 生成 Static Per-Head K/V scale
│   ├── 05_real_kv_kernel_test.py            # 使用真实 Qwen KV 测试自研 Triton Kernel
│   ├── 06_inspect_vllm_attention_api.py     # 检查当前 vLLM 0.26.0 Attention / KV Cache API 与调用链
│   ├── 07_run_qwen_shadow_int8.py           # Step 3：真实 vLLM Runtime 同步写 INT8 Shadow KV Cache
│   ├── 08_run_qwen_shadow_attention.py      # Step 4A：BF16 与 INT8 Attention Shadow Compare
│   └── 09_run_qwen_int8_takeover.py         # Step 4B：INT8 Attention takeover / int8_only 端到端测试
│
├── src/                                    # 核心量化与 Triton 算子实现
│   ├── triton_ops/
│   │   ├── bf16_paged_attention.py          # Triton BF16 PagedAttention baseline，用于公平 Kernel 对比
│   │   ├── int8_cache_write.py              # 自研 Triton INT8 Quantize + Paged KV Cache Write Kernel
│   │   └── int8_paged_attention.py          # 自研 Triton INT8 PagedAttention Decode Kernel
│   ├── attention_ref.py                     # PyTorch Attention Reference，用于 correctness 对照
│   ├── paged_cache.py                       # Python/PyTorch Paged KV Cache 辅助结构与 Reference
│   └── quant.py                             # Static Per-Head INT8 量化 / 反量化辅助实现
│
├── tests/                                  # 单元测试与集成测试
│   ├── test_attention.py                    # 基础 Attention Reference 正确性测试
│   ├── test_cache_write.py                  # 初版 INT8 Cache Write Kernel correctness
│   ├── test_int8_paged_attention.py         # INT8 PagedAttention Triton vs PyTorch Reference
│   ├── test_paged_cache.py                  # Paged Cache logical/physical block mapping 测试
│   ├── test_quant.py                        # INT8 Static Per-Head 量化数学正确性测试
│   ├── test_runtime_cache_write.py          # V3 Step 1：Runtime new K/V + slot_mapping Cache Write 单测
│   ├── test_runtime_write_and_attention.py  # V3 Step 2：Cache Write → INT8 PagedAttention 首尾闭环测试
│   ├── test_vllm_cache_quant.py             # vLLM KV Cache Adapter / 量化结构验证
│   └── test_vllm_int8_cache_ops.py          # vLLM INT8 Cache Runtime 操作测试
│
├── vllm_int8/                              # vLLM 0.26.0 INT8 KV Cache 系统集成层
│   ├── __init__.py                          # Python package 初始化
│   ├── cache_manager.py                     # 按 Layer 管理 INT8 Shadow K/V Cache 与 static scales
│   ├── static_scales.py                     # 加载 calibration 输出，并按 layer_name 获取 K/V scale
│   ├── vllm_int8_attention_backend.py       # 实验性的自定义 Attention Backend 适配实现
│   ├── vllm_int8_attention_patch.py         # Step 4：shadow / takeover / int8_only Attention Patch
│   ├── vllm_int8_cache_ops.py               # vLLM Runtime Cache Write Adapter
│   ├── vllm_int8_kvcache_adapter.py         # vLLM KV Cache Tensor 与项目内部 INT8 Cache layout 的适配工具
│   ├── vllm_int8_patch.py                   # 早期 vLLM Monkey Patch 实验入口与验证代码
│   └── vllm_shadow_cache_patch.py            # Step 3：Hook vLLM Runtime Cache Write 并同步 INT8 Shadow Cache
│
├── run_qwen_int8.py                        # Qwen2.5-7B INT8 集成实验入口脚本
├── vllm_kv_cache_hook.py                   # Hook vLLM KV Cache，分析真实 shape/dtype/stride/layout
└── README.md                               # 当前项目说明文档
```

> `Qwen/Qwen2.5-7B-Instruct` 权重体积很大，实际开源仓库通常不建议直接提交模型权重；推荐通过 `.gitignore` 排除 `Qwen/` 并在 README 中提供模型下载说明。

---

## 6. 环境

当前 V3 验证环境核心版本：

```text
vLLM: 0.26.0
Model: Qwen2.5-7B-Instruct
Model dtype: BF16
KV baseline dtype: BF16
Custom KV dtype: INT8
Attention backend: TRITON_ATTN
TP: 1
```

建议保存完整环境：

```bash
mkdir -p outputs/v3/final

{
    echo "===== DATE ====="
    date

    echo
    echo "===== PYTHON ====="
    python --version

    echo
    echo "===== VLLM ====="
    python -c "import vllm; print(vllm.__version__)"

    echo
    echo "===== TORCH ====="
    python -c "import torch; print(torch.__version__)"

    echo
    echo "===== TRITON ====="
    python -c "import triton; print(triton.__version__)"

    echo
    echo "===== CUDA ====="
    python -c "import torch; print(torch.version.cuda)"

    echo
    echo "===== GPU ====="
    nvidia-smi

} > outputs/v3/final/environment.txt 2>&1
```

---

## 7. 快速开始

### 7.1 创建环境

示例：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装项目运行所需依赖，例如：

```bash
pip install torch
pip install triton
pip install transformers
pip install vllm==0.26.0
```

实际 PyTorch / CUDA wheel 请根据服务器 CUDA 与 GPU 环境选择。

进入项目根目录：

```bash
export PYTHONPATH=.
```

---

## 8. 模型准备

项目默认使用：

```text
Qwen2.5-7B-Instruct
```

模型可以放在：

```text
Qwen/Qwen2.5-7B-Instruct/
```

先检查：

```bash
python scripts/00_check_model.py
```

确认至少包括：

```text
num_hidden_layers = 28
num_attention_heads = 28
num_key_value_heads = 4
head_dim = 128
```

---

## 9. 完整复现流程

推荐严格按以下顺序执行。

### Step 0：模型检查

```bash
python scripts/00_check_model.py
```

目的：

- 确认模型路径；
- 检查 Qwen config；
- 确认 Attention/GQA 参数。

### Step 1：采集真实 KV

```bash
python scripts/01_collect_kv.py
```

### Step 2：分析 KV 分布

```bash
python scripts/02_analyze_kv.py
```

输出：

```text
plots/layer_*_k.png
plots/layer_*_v.png
```

### Step 3：PyTorch 量化精度验证

```bash
python scripts/03_accuracy.py
```

### Step 4：Static Scale Calibration

```bash
python scripts/04_calibrate_static_scale.py
```

典型输出 shape：

```text
k_scale: [28, 4]
v_scale: [28, 4]
```

### Step 5：真实 KV Kernel 测试

```bash
python scripts/05_real_kv_kernel_test.py
```

---

## 10. 独立 Kernel Correctness

```bash
python tests/test_quant.py
python tests/test_cache_write.py
python tests/test_paged_cache.py
python tests/test_int8_paged_attention.py
```

---

## 11. V3 Step 1：Runtime Cache Write

```bash
python tests/test_runtime_cache_write.py
```

实际通过结果：

```text
K max integer diff: 0
V max integer diff: 1
PASS
```

---

## 12. V3 Step 2：Cache Write + Attention 闭环

```bash
python tests/test_runtime_write_and_attention.py
```

它同时验证：

```text
Runtime Cache Write
        ↓
INT8 Paged KV Cache
        ↓
INT8 PagedAttention
```

---

## 13. V3 Step 3：真实 vLLM Runtime Cache Write

先检查 vLLM API：

```bash
python scripts/06_inspect_vllm_attention_api.py
```

然后：

```bash
python scripts/07_run_qwen_shadow_int8.py
```

预期：

```text
num shadow layers: 28
```

---

## 14. V3 Step 4A：Shadow INT8 Attention

```bash
python scripts/08_run_qwen_shadow_attention.py
```

真实实验中：

```text
28 Layer Attention cosine ≈ 0.999
```

---

## 15. V3 Step 4B：Takeover / INT8 Only

```bash
python scripts/09_run_qwen_int8_takeover.py
```

建议依次测试：

```text
max_tokens = 1
max_tokens = 4
max_tokens = 8
max_tokens = 32
```

---

## 16. V3 Step 5：Benchmark

### 16.1 Context Sweep

```bash
python bench/benchmark_qwen_in8.py
```

测试：

```text
Context:
512
1024
2048
4096
```

### 16.2 Decode TPOT

```bash
python bench/benchmark_decode_tpot.py
```

方法：

```text
T1   = Prompt + 1 Decode token
T128 = Prompt + 128 Decode tokens

Approx Decode TPOT
≈
(T128 - T1) / 127
```

### 16.3 Batch Sweep

```bash
python bench/benchmark_batch_sweep.py
```

当前测试：

```text
Context = 2048
Batch = 1 / 2 / 4 / 8
```

---

## 17. Microbenchmark

```bash
python bench/bench_attention.py
python bench/bench_paged_attention.py
```

---

## 18. vLLM KV Cache Layout 探测

```bash
python vllm_kv_cache_hook.py
```

本项目在 vLLM 0.26.0 上曾观测到类似：

```text
shape:
(num_blocks, 4, 16, 256)

dtype:
torch.bfloat16

stride:
(16384, 256, 1024, 1)
```

注意：

> 不应仅根据 shape 猜测底层 storage layout，必须结合 stride 与当前 backend 源码分析。

---

## 19. 当前完成内容

V3 已完成：

```text
[✓] Qwen2.5-7B KV 数据采集
[✓] KV 数值分布分析
[✓] Static Per-Head INT8 方案
[✓] Scale Calibration
[✓] PyTorch Quant Reference
[✓] Triton INT8 Cache Write
[✓] Triton INT8 PagedAttention
[✓] Triton BF16 PagedAttention baseline
[✓] Paged Block Table
[✓] slot_mapping
[✓] GQA 28 → 4
[✓] Online Softmax
[✓] 独立 Correctness
[✓] 真实 Qwen KV Correctness
[✓] vLLM 0.26.0 KV layout 分析
[✓] vLLM Runtime INT8 Cache Write
[✓] 28 Layer Shadow INT8 Cache
[✓] 28 Layer Shadow Attention Compare
[✓] INT8 Attention takeover
[✓] int8_only Decode
[✓] Qwen2.5-7B 端到端生成
[✓] KV payload memory benchmark
[✓] Context Sweep
[✓] Decode TPOT
[✓] Batch Sweep
```

---

## 20. 当前尚未完成

以下内容明确 **不属于当前 V3 已完成能力**。

### 20.1 Native INT8 KV Cache Allocation

当前采用：

```text
BF16 native cache
+
INT8 shadow cache
```

因此虽然 INT8 payload 本身约为 BF16 的 50%，但 `torch.cuda.max_memory_allocated()` 不会体现真正的 50% GPU 总显存下降。

### 20.2 INT8 PagedAttention 性能优化

当前 correctness-first Kernel 尚未超过 vLLM BF16 backend。

后续 V4 可研究：

```text
GQA-aware KV reuse
vectorized INT8 load
dequant fusion
BLOCK_N tuning
num_warps
num_stages
register pressure
occupancy
warp specialization
INT8 Tensor Core MMA
INT8×INT8 → INT32 accumulation
```

### 20.3 Multi-GPU

当前：

```text
TP = 1
single GPU
```

### 20.4 其他 Attention 模式

当前未支持：

```text
MLA
Sliding Window
Hybrid Attention
Prefix Cache
Speculative Decode
```

### 20.5 Production-grade Backend

当前主要通过：

```text
Monkey Patch
+
Shadow Cache
```

完成系统验证。

尚未将 `int8_static_per_head` 正式注册为 vLLM native KV Cache dtype / production backend。

---

## 21. 为什么 INT8 当前没有更快

V3 结果显示：

```text
KV Cache payload:
约 -50%

但是：
Decode TPOT > BF16
```

原因是：

> **减少 KV memory traffic 并不自动等价于端到端加速。**

当前 Kernel 仍存在：

```text
INT8 Load
    ↓
cast / dequant
    ↓
float compute
```

并且尚未充分优化：

```text
GQA KV reuse
Tensor Core
warp mapping
pipeline
occupancy
memory coalescing
```

因此本项目当前最准确的结论是：

> V3 已证明 Static Per-Head INT8 KV Cache 在 Qwen2.5-7B/vLLM Runtime 中具有良好的数值正确性和约 50% 的 KV payload 压缩能力；V4 通过 GQA reuse / `tl.dot` / split-KV / autotune，在长上下文与大 batch 的 kernel microbench 上已超过自研 BF16 baseline（见 §2.3），默认 `impl="auto"` 在短上下文回退 V3 以保证 occupancy。

---

## 22. 推荐 Git 管理

建议不要提交模型权重。

`.gitignore`：

```gitignore
.venv/
__pycache__/
*.pyc

Qwen/
outputs/

.vscode/
.idea/

*.log
```

---

## 23. 项目复现最短路径

如果不希望重新执行完整 KV 分布研究，只希望复现实验闭环：

```text
1. 准备 Qwen2.5-7B
        ↓
2. 准备 Static Per-Head scale
        ↓
3. tests/test_runtime_cache_write.py
        ↓
4. tests/test_runtime_write_and_attention.py
        ↓
5. scripts/07_run_qwen_shadow_int8.py
        ↓
6. scripts/08_run_qwen_shadow_attention.py
        ↓
7. scripts/09_run_qwen_int8_takeover.py
        ↓
8. bench/benchmark_decode_tpot.py
        ↓
9. bench/benchmark_batch_sweep.py
```

完整研究流程：

```text
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → Benchmark
```

---

## 24. 项目总结

该项目完成了一条完整的大模型推理优化实验链路：

```text
真实模型数据分析
        ↓
量化策略设计
        ↓
PyTorch Reference
        ↓
Triton Cache Write
        ↓
Triton PagedAttention
        ↓
Kernel Correctness
        ↓
vLLM Runtime Reverse Engineering
        ↓
Cache Write Integration
        ↓
Attention Read Integration
        ↓
Qwen2.5-7B End-to-End
        ↓
Accuracy / Memory / Performance Benchmark
```

当前 V3 最主要的成果是：

```text
28 层真实 Attention：
INT8 / BF16 cosine ≈ 0.999

KV Cache payload：
1004.5 MB → 502.25 MB
≈ 50% reduction

Qwen2.5-7B：
int8_only Decode 正常运行
```

当前最主要的不足是：

```text
短上下文 / 小 batch：
  V4 GQA grid 并行度不足，需 auto→V3 或更强 split-KV

端到端：
  仍可能落后于 vLLM 高度优化的 native BF16 backend
  （kernel 层长上下文已可超过自研 BF16）
```

因此项目下一阶段可将重点从纯算子优化转向：

```text
Native INT8 KV allocator
+
与 vLLM quantized backend 对齐 / 生产化接入
```

---

## 25. 简历描述参考

### INT8 KV Cache 与 PagedAttention 推理优化｜Triton / vLLM / Qwen2.5-7B

- 面向 LLM Decode 阶段 KV Cache 显存与访存瓶颈，分析 Qwen2.5-7B 的 GQA KV 分布，设计 Static Per-Head 对称 INT8 KV Cache 量化方案，并完成各 Transformer Layer / KV Head 的量化 scale calibration。
- 基于 Triton 自研 INT8 KV Cache Write 与 INT8 PagedAttention Kernel，实现 `slot_mapping` 分页写入、Block Table 物理页寻址、28/4 GQA Head 映射与 Online Softmax，并接入 vLLM 0.26.0 的真实 Prefill/Decode Cache Write 与 Decode Attention 路径。
- 在 Qwen2.5-7B 28 层真实 Attention 上实现 INT8/BF16 输出 cosine≈0.999，将 KV Cache payload 从约 1004.5 MB 压缩至 502.25 MB（约 50%）；完成 512～4096 Context 与 Batch 1～8 的 TPOT/吞吐 benchmark，并定位 correctness-first INT8 PagedAttention 相对 vLLM 原生 BF16 Kernel 的性能瓶颈。

---

## 26. 后续工作：V4 → 已完成核心项

```text
V4 — INT8 PagedAttention Kernel Performance Optimization

[x] Kernel profiling / microbench（bench/bench_paged_attention.py → outputs/v4_baseline/）
[x] GQA-aware KV reuse（grid = B × Hkv）
[x] 减少 INT8→FP 路径开销（V scale 延后到最终归一化）
[x] Vectorized/coalesced KV load + 单 page 快路径
[x] BLOCK_N / num_warps / num_stages autotune
[x] Split-KV（小 batch / 长上下文）
[x] tl.dot Tensor-Core 友好 QK/PV（bf16）
[x] 可选 Query INT8（quantize_q=True）
[x] impl="auto" 短上下文回退 V3 / 长上下文走 V4
[ ] 与 vLLM native quantized KV backend 对比（环境相关）
[ ] Native INT8 KV Cache allocator（显存真实减半，属系统改造）
```

复现 V4 microbench：

```bash
PYTHONPATH=. python bench/bench_paged_attention.py
# optional: bash scripts/10_ncu_profile.sh
```

V3 到此结束；V4 算子优化主线已合入 `src/triton_ops/int8_paged_attention.py`。
