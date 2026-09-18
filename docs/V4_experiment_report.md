# V4 INT8 算子性能优化实验报告

> 项目：`int8-kvcache`  
> 硬件：NVIDIA GeForce RTX 4090  
> 目标形状：Qwen2.5-7B（`Hq=28, Hkv=4, D=128, block_size=16`）  
> 文档日期：2026-09-19  

---

## 1. 背景与动机

### 1.1 V3 留下的问题

V3 已完成 Static Per-Head INT8 KV Cache 的正确性闭环与 vLLM 集成，主要结论是：

- **正确性**：28 层 Attention 上 INT8 / BF16 cosine ≈ 0.999
- **存储**：KV Cache payload 约减半（~1004 MB → ~502 MB）
- **性能**：端到端 Decode TPOT 明显慢于 vLLM 原生 BF16 backend

| Context | BF16 TPOT (ms) | INT8 TPOT (ms) | 差距 |
|---:|---:|---:|---:|
| 512 | 16.85 | 25.79 | ~1.5× |
| 1024 | 17.20 | 25.52 | ~1.5× |
| 2048 | 16.41 | 28.74 | ~1.8× |
| 4096 | 16.58 | 33.60 | ~2.0× |

V3 kernel 是 correctness-first 实现，README 已明确把 **Kernel Profiling / Optimization** 留给 V4。

### 1.2 V3 kernel 的结构性瓶颈（制定计划前的代码研判）

对 [`src/triton_ops/int8_paged_attention.py`](src/triton_ops/int8_paged_attention.py)（V3）阅读后，确认主要问题：

| 问题 | V3 现状 | 影响 |
|------|---------|------|
| GQA 重复读 KV | `grid=(B, Hq)`，28 个 Q head 各自读同一 KV（每组 7 次） | 长上下文 KV 流量放大约 7× |
| 非 Tensor Core 计算 | `tl.sum(k * q)` / `tl.sum(p * v)` 元素乘加 | 算力利用率低 |
| Dequant 路径重 | INT8 → FP32 cast → 乘 scale | 带宽收益被算术开销吃掉 |
| Launch 未调参 | 固定 `num_warps=4, num_stages=2`，tile=`block_size=16` | occupancy / 流水线次优 |

因此 V4 的核心不是“再压一点 INT8”，而是：**在保持正确性的前提下，把 GQA 复用、访存、计算路径和 launch 参数做对**。

---

## 2. 计划是如何制定的

### 2.1 范围决策

用户明确要求做 **V4 算子性能优化**。计划阶段将范围锁定为：

- **纳入**：INT8 PagedAttention / Cache Write 的 Triton kernel、microbench、正确性回归
- **不纳入本轮**：Native INT8 KV allocator、Multi-GPU、MLA、与 vLLM 官方量化 backend 的完整对齐

成功标准（计划内建议）：

1. Microbench：长上下文下 INT8 ≤ 自研 BF16（同 layout）
2. 正确性：cosine 不显著退化（目标 ≥ 0.999 量级）
3. E2E：`int8_only` TPOT 逼近 vLLM BF16（依赖可用 CUDA + vLLM 环境）

### 2.2 分阶段路线（M0–M4）

计划按 **Profiling 驱动、先访存后计算、最后长上下文** 排序：

```text
Phase0  Baseline / Profiling
   ↓
Phase1  GQA reuse + dequant 融合 + autotune   ← 预期最大收益
   ↓
Phase2  tl.dot / Q-INT8 / Tensor Core 友好计算
   ↓
Phase3  BLOCK_N + Split-KV + 复测与文档
```

对应 todo：

| ID | 内容 |
|----|------|
| m0-profiling | 强化 microbench + roofline/nsys 基线 |
| m1-gqa-reuse | GQA-aware KV reuse（grid 按 Hkv） |
| m2-autotune-dequant | dequant 融合 + warps/stages/BLOCK_N autotune |
| m3-tldot | `tl.dot` tile 化；实验 Q-INT8 |
| m4-splitkv-rebench | Split-KV + 全量复测 + README |

---

## 3. 各阶段如何执行

### 3.1 Phase 0：Baseline / Profiling

**做了什么**

1. 重写并扩展 [`bench/bench_paged_attention.py`](bench/bench_paged_attention.py)：
   - 固定 Qwen 形状
   - 扫 `seq ∈ {512,1024,2048,4096}`、`B ∈ {1,2,4,8}`
   - 对比自研 BF16 / INT8-V3 / INT8-V4 / Auto / Q-INT8 / 强制 Split-KV
   - 估算 KV bytes，给出粗粒度 roofline hint（memory / compute / mixed）
   - 结果写入 `outputs/v4_baseline/`（`microbench.json` / `.csv` / `decision.md`）
2. 新增可选 Nsight 脚本 [`scripts/10_ncu_profile.sh`](scripts/10_ncu_profile.sh)（本机无 `ncu`，自动 skip）

**环境插曲（影响执行路径，但不改变优化方向）**

- 项目 `.venv` 原为 `torch 2.11.0+cu130`，与驱动 CUDA 12.8 不兼容 → `cuda.is_available()=False`
- 实测阶段改用 conda `torch 2.7.0+cu128` 跑通
- 后续将 `.venv` 调整为 `torch 2.9.1+cu128`（可用）；`vllm==0.26.0` 需要 CUDA 13，本机驱动 12.8 无法加载
- E2E 最终改用 **vLLM 0.16.0 + torch 2.9.1+cu128** 跑通 Batch Sweep（见 §4.4 / README §2.4）

**Phase0 决策结论**

- 短上下文（B=1, seq≤1024）：更像 **compute / launch / occupancy** 主导
- 长上下文 / 大 batch：KV 流量重要性上升，**GQA reuse 应成为第一优先级**
- 这与计划中的 “先 Phase1 GQA” 一致；短上下文则提示后续需要 occupancy 策略（最终演化为 `impl="auto"`）

---

### 3.2 Phase 1：GQA reuse + dequant + autotune

**执行方式**

重写 [`src/triton_ops/int8_paged_attention.py`](src/triton_ops/int8_paged_attention.py) 主路径：

1. **GQA-aware kernel** `_int8_paged_attention_gqa_kernel`
   - `grid = (B, Hkv[, num_splits])`
   - 每个 program 负责一个 KV head 下全部 `GROUP_SIZE=7` 个 Q heads
   - Q 以 `[GROUP_PAD, D]` 驻留，一次加载 K/V tile，对组内全部 Q 做 QK/PV
2. **Dequant 融合**
   - K：`score = dot(q, k_int) * k_scale * sm_scale`（scale 在点积后）
   - V：循环内累加 **未乘 v_scale** 的 PV，最终 `out = (acc * v_scale) / l`
3. **Autotune**
   - 搜索 `NUM_BLOCKS_PER_TILE ∈ {1,2,4}`、`num_warps ∈ {2,4,8}`、`num_stages ∈ {2,3,4}`
   - `key` 含 `max_num_blocks`，短序列 prune 掉过大 tile / 过多 warps

**正确性门禁**

新增测试 `test_int8_paged_attention_v4_matches_reference`：

- V3 / V4 / V4-split 对比 torch reference
- 实测：cosine ≈ **0.999998**，rel L2 ≈ **0.002**

---

### 3.3 Phase 2：`tl.dot` 与 Q-INT8

**执行方式**

1. QK / PV 改为 `tl.dot`（操作数 cast 到 `bfloat16`，累加回 FP32）
2. 保留 `quantize_q=True` 实验路径（microbench 中的 V4q 列）
3. 未把 INT8×INT8→INT32 MMA 作为默认主路径（精度/实现成本权衡；主路径为 **bf16 tl.dot + INT8 KV load**）

**实现约束发现（见 §5）**

- Ada 上 `tl.dot` 要求 `M,N,K ≥ 16`
- `GROUP_SIZE=7` → `GROUP_PAD` 必须至少 **16**（不是 `next_power_of_2(7)=8`）

---

### 3.4 Phase 3：BLOCK_N / Split-KV / 复测

**执行方式**

1. `NUM_BLOCKS_PER_TILE>1`：多 page gather 成更大 `BLOCK_N` tile
2. Split-KV：小 batch 时沿 sequence 切分，写 partial `(m, l, out)`，再用 `_merge_split_kv_kernel` 在线 softmax 合并
3. Cache Write 轻量改动：[`int8_cache_write.py`](src/triton_ops/int8_cache_write.py) 增加可配置 `num_stages`（默认 2）
4. 更新 README §2.3 / §21 / §26，固化 microbench 表

---

## 4. 实验结果

### 4.1 正确性

| 对比 | cosine | rel L2 |
|------|-------:|-------:|
| V3 vs reference | 1.000000 | 0 |
| V4 vs reference | 0.999998 | ~0.002 |
| V4-split vs reference | 0.999998 | ~0.002 |
| 原 correctness 套件 | 通过 | — |

结论：**V4 在 bf16 `tl.dot` 路径下相对 FP32 reference 有可接受的数值差，正确性门禁通过。**

### 4.2 Kernel Microbench（重跑 2026-09-19）

数据文件：[`outputs/v4_baseline/microbench.csv`](outputs/v4_baseline/microbench.csv)

单位：单次 Attention kernel 平均延迟（ms）

> **是否受 B=1 门控影响？否。**  
> 本表来自 `bench/bench_paged_attention.py` 直接调用 Triton wrapper，**不经过** `vllm_int8_attention_patch._is_supported_decode`。  
> 因此表中 **B=2/4/8 行是真实算子结果**，结论成立。  
> 受门控影响的是 README §2.4 的 **V3 历史 vLLM Batch Sweep**（INT8×B>1 未跑到自定义 Attention）。  
> **V4 e2e 已重跑**：每条 INT8 行命中 `int8_paged_attention`（`int8_decode_hits=3556`）。

| B | Seq | BF16 | INT8 V3 | INT8 V4 | INT8 Auto | V4q | V4s(强制split) | V3/V4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 0.056 | 0.136 | 0.226 | **0.128** | 0.226 | 0.188 | 0.60× |
| 1 | 1024 | 0.080 | 0.167 | 0.228 | **0.167** | 0.228 | 0.189 | 0.74× |
| 1 | 2048 | 0.157 | 0.241 | 0.229 | 0.243 | 0.227 | 0.227 | 1.05× |
| 1 | 4096 | 0.342 | 0.414 | **0.270** | **0.270** | 0.270 | 0.267 | **1.53×** |
| 2 | 2048 | 0.160 | 0.241 | 0.227 | 0.227 | 0.227 | 0.212 | 1.06× |
| 4 | 2048 | 0.172 | 0.254 | 0.295 | **0.228** | 0.229 | 0.211 | 0.86× |
| 8 | 2048 | 0.232 | 0.339 | **0.227** | **0.227** | 0.229 | 0.212 | **1.49×** |
| 8 | 4096 | 0.464 | 0.519 | **0.266** | **0.261** | 0.262 | 0.254 | **1.95×** |

### 4.3 结果解读

1. **长上下文 / 大 batch：V4 达到目标**
   - B=1, seq=4096：V4 相对 V3 **约 1.53×**，且快于自研 BF16（0.270 vs 0.342）
   - B=8, seq=4096：V4 相对 V3 **约 1.95×**，明显快于自研 BF16（0.266 vs 0.464）
2. **短上下文：纯 V4 仍可能更慢，Auto 回退 V3**
   - B=1, seq=512：Auto 0.128 ≈ V3，优于强制 V4 0.226
   - 原因：V4 grid 只有 `B×Hkv` 个 program，短序列 **occupancy / launch 主导**
3. **`impl="auto"` 是必要工程决策**
   - 短上下文回退 V3，长上下文 / 大 batch 走 V4
   - 启发式：`work=B*max_blocks ≥ 192` 或 `max_blocks≥192` 或 `B≥8` → V4，否则 V3
4. **Q-INT8（V4q）**：多数点与 V4 接近，不作默认
5. **强制 Split-KV（V4s）**：短上下文有帮助，但仍常不及 Auto→V3

### 4.4 未完成 / 环境限制的部分

| 项 | 状态 | 原因 |
|----|------|------|
| Nsight Compute (`ncu`) 硬件计数 | 跳过 | 机器无 `ncu` |
| vLLM 端到端 Batch Sweep（门控修复后） | **已完成** | 过程与重跑记录见 [`V4_e2e_batch_sweep.md`](V4_e2e_batch_sweep.md)；汇总表见 README §2.4 |
| Native INT8 allocator | 明确不在本轮范围 | 属系统改造 |
| INT8 MMA 默认主路径 | 仅保留实验空间 | 主路径用 bf16 `tl.dot` |

---

## 5. 中间额外发现与额外改动

### 5.1 `tl.dot` 形状约束（关键 blocker）

**现象**：首次编译 GQA kernel 报错：

```text
AssertionError: Input shapes should have M >= 16, N >= 16 and K >= 16
```

**原因**：`GROUP_PAD = next_power_of_2(7) = 8 < 16`

**改动**：`group_pad = max(16, next_power_of_2(group_size))`

这会在寄存器侧为 7 个有效 Q head 多 pad 到 16，带来一定浪费，但是启用 Tensor Core 路径的前提。

### 5.2 多 page gather 在短序列上的负优化

**现象**：第一版统一用 “按 row gather physical block id” 组装 `BLOCK_N` tile，短上下文 V4 显著慢于 V3。

**改动**：

- `NUM_BLOCKS_PER_TILE == 1` 走 **单 page 直载快路径**（无 gather）
- 仅在更大 tile 时走 multi-page gather
- autotune prune：短 `max_num_blocks` 时禁用大 tile / 过多 warps

### 5.3 Autotune key 必须包含序列长度信息

**现象**：若 autotune key 只有 `HEAD_DIM/CACHE_BLOCK_SIZE/GROUP_SIZE`，所有 seq 共享一套 config，短/长序列互相拖累。

**改动**：key 增加 `max_num_blocks`，并配合 `_prune_gqa_configs`。

### 5.4 Occupancy vs KV-reuse 的 crossover（最重要产品化发现）

**发现**：GQA reuse 不是在所有工况下都更快。

- **长序列**：KV 流量主导 → reuse 赢
- **短序列 / 极小 batch**：program 数主导 → per-head V3 赢

**额外改动（计划中未预先写死，但实验后必须做）**：

- 保留 V3 kernel 作为 `impl="v3"`
- 新增 `impl="auto"` 并设为 **默认**
- vLLM 接入路径不改参数即可享受 auto 选择（`vllm_int8_*` 仍调用 `int8_paged_attention(...)`）

### 5.5 Split-KV merge 数值细节

**改动要点**：

- partial 存的是已归一化的 `out = (acc*v_scale)/l`
- merge 时用 `num = out * l` 恢复分子，再按 online softmax 合并
- 空 split（`m=-inf` / `l=0`）要显式置零，避免 NaN

### 5.6 PyTorch / CUDA 环境反复折腾

时间线摘要：

1. `.venv` cu130 → CUDA 不可用  
2. 尝试装 cu128 → 成功可用  
3. 后台 cu124 任务把 cu128 **覆盖**成 2.6.0+cu124（仍可用，但与依赖声明更差）  
4. 恢复 cu128 后，`vllm==0.26.0` 仍因 `libcudart.so.13` 无法 import  
5. **最终可用栈**：`torch 2.9.1+cu128` + `vllm==0.16.0`（`--no-deps` 安装，避免覆盖 torch）；驱动 570.124.04 / CUDA 12.8 / RTX 4090

**影响**：E2E 数字来自 vLLM 0.16.0，不能与 V3 时期 0.26.0 表直接比绝对值。Kernel microbench 不依赖 vLLM，结论不受此影响。

### 5.7 Cache Write 的定位确认

Decode 每步只写 1 token，Cache Write 不是 TPOT 主因。本轮只做轻量 `num_stages` 可配，**未作为主里程碑**。

### 5.8 vLLM patch 曾把 INT8 Attention 限制为 B=1（重要数据口径）

**发现**：`vllm_int8_attention_patch._is_supported_decode` 历史上强制 `query/seq_lens/block_table` 的 batch 维均为 1；B>1 decode 会回退到 vLLM BF16 Attention。

**影响范围**：

| 数据 | 是否受影响 |
|------|------------|
| V4 kernel microbench（直接调 `int8_paged_attention`） | **否** — B>1 结论有效 |
| V3 文档中的 **e2e Batch Sweep INT8×B>1** | **是** — 未跑到自定义算子，已降为存档 |
| **V4 e2e Batch Sweep（2026-09-19 重跑）** | **否** — 每条 INT8 行 `int8_decode_hits=3556` 且 `int8_last_batch==B` |
| e2e TPOT（默认 B=1） | **否** |

**改动**：门控已放宽为「任意 B 的纯 decode」；vLLM 0.16 可能 pad query，改为优先用 `num_actual_tokens`。Shadow patch 用 `_infer_native_page_dims` 兼容 0.16 的 `[num_blocks,2,block_size,Hkv,D]`。

**V4 e2e 结果摘要**（ctx≈1973，128 new tokens）：

| B | BF16 tok/s | INT8 tok/s | INT8/BF16 time | INT8 hits |
|--:|----------:|----------:|---------------:|----------:|
| 1 | 52.31 | 39.95 | 1.31× | 3556 |
| 2 | 104.74 | 22.62 | 4.63× | 3556 |
| 4 | 179.60 | 43.37 | 4.14× | 3556 |
| 8 | 274.47 | 82.61 | 3.32× | 3556 |

E2E INT8 仍慢于 vLLM 原生 BF16：shadow 双写 + patch 开销；Peak MB 因 dual cache 更高（约 18.5→20.3–21.1 GB）。Kernel 层加速没有在这条集成路径上变成端到端加速。

---

## 6. 关键文件与具体修改

### 6.1 核心算子（重写级）

#### [`src/triton_ops/int8_paged_attention.py`](src/triton_ops/int8_paged_attention.py)

**性质**：本轮主改文件（几乎整体重写，保留 torch reference）

| 模块 | 作用 |
|------|------|
| `_int8_paged_attention_gqa_kernel` | V4 主 kernel：GQA reuse + `tl.dot` + BLOCK_N + Split-KV 写出 |
| `_merge_split_kv_kernel` | Split-KV partial 合并 |
| `_int8_paged_attention_kernel_v3` | 保留的 per-head baseline（A/B 与 auto 回退） |
| `_GQA_AUTOTUNE_CONFIGS` / `_prune_gqa_configs` | launch 配置搜索与剪枝 |
| `_choose_num_splits` | 自动选择 split 数 |
| `_auto_impl` | 短/长上下文启发式选择 |
| `_launch_v3` / `_launch_v4` | 封装 launch |
| `int8_paged_attention(..., impl="auto"\|"v3"\|"v4", num_splits=..., quantize_q=...)` | 对外 API（默认 auto） |
| `torch_int8_paged_attention_reference` | 正确性参考（逻辑保留） |

对外兼容：原调用方不传 `impl` 时走 `auto`，无需改 vLLM patch。

#### [`src/triton_ops/int8_cache_write.py`](src/triton_ops/int8_cache_write.py)

**性质**：轻量修改

- `int8_kv_cache_write` 增加参数 `num_stages: int = 2`
- kernel launch 由硬编码 `num_stages=1` 改为使用该参数

### 6.2 Benchmark / Profiling

#### [`bench/bench_paged_attention.py`](bench/bench_paged_attention.py)

**性质**：重写扩展

- BF16 / V3 / V4 / Auto / V4q / V4s 对比
- roofline hint、JSON/CSV 落盘
- 输出目录：`outputs/v4_baseline/`

#### [`scripts/10_ncu_profile.sh`](scripts/10_ncu_profile.sh)

**性质**：新增

- 有 `ncu` 时采集 `int8_paged_attention` profile
- 无 `ncu` 时友好退出并提示看 `decision.md`

### 6.3 测试

#### [`tests/test_int8_paged_attention.py`](tests/test_int8_paged_attention.py)

**性质**：新增用例

- `test_int8_paged_attention_v4_matches_reference`
  - 覆盖 V3、V4（`num_splits=1`）、V4-split（`num_splits=4`）
  - 相对 torch reference 的 cosine / rel L2 断言

### 6.4 文档与产物

| 文件 | 修改内容 |
|------|----------|
| [`README.md`](README.md) | §2.3 增加 V4 microbench；§21/§24 结论更新；§26 勾选 V4 完成项 |
| [`outputs/v4_baseline/microbench.csv`](outputs/v4_baseline/microbench.csv) | 最终数值表 |
| [`outputs/v4_baseline/microbench.json`](outputs/v4_baseline/microbench.json) | 完整行数据 + 决策字段 |
| [`outputs/v4_baseline/decision.md`](outputs/v4_baseline/decision.md) | Phase0/终态决策摘要 |
| **本文件** `docs/V4_experiment_report.md` | 完整实验过程文档 |

### 6.5 未改但相关的接入层

以下文件 **本轮未改代码**，但会间接受益于默认 `impl="auto"`：

- `vllm_int8/vllm_int8_attention_backend.py`
- `vllm_int8/vllm_int8_attention_patch.py`

它们继续调用 `int8_paged_attention(...)`，无需改签名即可使用 V4/auto。

---

## 7. 最终架构（V4 数据流）

```text
int8_paged_attention(impl="auto")
        │
        ├─ short / tiny batch ──► V3 per-head kernel
        │                         grid (B, Hq)
        │                         elementwise QK/PV
        │
        └─ long / large batch ──► V4 GQA kernel
                                  grid (B, Hkv[, S])
                                  load KV once per group
                                  tl.dot QK/PV (bf16)
                                  optional Split-KV + merge
                                  deferred v_scale
```

---

## 8. 复现命令

```bash
cd /root/autodl-tmp/int8-kvcache
# 当前 .venv：torch 2.9.1+cu128 + vLLM 0.16.0（驱动 CUDA 12.8）
source .venv/bin/activate
export PYTHONPATH=.

# 正确性
python -c "import tests.test_int8_paged_attention as t; t.test_int8_paged_attention_v4_matches_reference(); print('ok')"

# Microbench
python bench/bench_paged_attention.py

# 端到端 Batch Sweep（BF16 vs INT8，B=1/2/4/8）
python bench/benchmark_batch_sweep.py

# 可选 Nsight
bash scripts/10_ncu_profile.sh
```

---

## 9. 结论与下一步建议

### 9.1 本轮结论

1. V4 在 **长上下文 / 大 batch** 的 kernel 层已证明有效：相对 V3 最高约 **2×**，并可超过自研 BF16。
2. **短上下文必须特殊处理**；默认 `auto` 是实验驱动的必要产品化改动，不是“偷懒回退”。
3. 正确性保持在 cosine ≈ 0.999998 量级，可进入后续系统层工作。
4. 端到端 Batch Sweep 已在 **vLLM 0.16.0** 上重跑，且 **B>1 INT8 命中自定义算子**；e2e 仍慢于原生 BF16（shadow 双写 + patch），不能把 kernel 加速直接当成系统加速。

### 9.2 建议的下一步（超出本轮 V4 算子范围）

1. Native INT8 KV Cache allocator，去掉 BF16 shadow，兑现真实显存减半并减少双写
2. 降低 Python patch 热路径开销（或把 INT8 attention 编进 vLLM worker）
3. 深化短上下文 V4（更激进 split-KV、或 group 内再并行）以减少对 V3 回退的依赖
4. 若追求极致，再评估真正的 INT8 MMA 主路径

---

## 附录 A：计划 vs 实际对照

| 计划项 | 实际结果 |
|--------|----------|
| Kernel profiling | 完成（microbench + roofline hint；无 ncu） |
| GQA-aware KV reuse | 完成 |
| 减少 dequant overhead | 完成（V scale 延后） |
| Vectorized/coalesced load | 完成（单 page 快路径 + D 连续断言） |
| BLOCK_N / warps / stages autotune | 完成 |
| Register/occupancy 分析 | 以实验现象驱动（短上下文 occupancy 结论），无 ncu 计数 |
| INT8 Q quantization | 完成开关与 bench 列，未作默认 |
| INT8 Tensor Core MMA | 以 bf16 `tl.dot` 落地；纯 INT8 MMA 未默认启用 |
| Split-KV | 完成 |
| 与 vLLM quantized backend 对比 | 未做（环境） |
| Native allocator | 明确不在本轮 |

## 附录 B：关键 API

```python
from src.triton_ops.int8_paged_attention import int8_paged_attention

out = int8_paged_attention(
    query,           # [B, Hq, D]
    key_cache,       # [num_blocks, block_size, Hkv, D] int8
    value_cache,     # same
    block_tables,    # [B, max_blocks]
    seq_lens,        # [B]
    k_scale,         # [Hkv]
    v_scale,         # [Hkv]
    quantize_q=False,
    impl="auto",     # "auto" | "v3" | "v4"
    num_splits=None, # None=自动；仅对 v4 生效
)
```
