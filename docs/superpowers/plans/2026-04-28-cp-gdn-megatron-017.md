# Qwen3.5-35B-A3B 长序列 — Megatron 0.17 GDN-CP 升级方案

**分支**: `feat/cp-gdn-megatron-017`

**目标**: 通过升级 megatron-core 0.16.0 → 0.17.0 拿到原生 GDN context-parallel 支持，再把 CP 通路打通到 mbridge + AReaL，以支持 Qwen3.5-35B-A3B 跑到 32K 序列长度。

**触发原因**: v22 (PP=4, 16K) 和 v22-pp8-r2 (PP=8 EP=4, 16K) 都在 compute_logp 阶段 OOM。15.31 GB 的失败分配是 last PP stage 的 LM-head logit buffer —— PP 层数加再多，logit 只在最后一个 stage，省不到这部分；TP=4 被 `num_kv_heads=2` 硬阻断；所以 16K/32K 唯一能动的结构性杠杆是 **CP**。

---

## Day 1 调研结果（2026-04-28）

### Megatron 上游状态

通过 NVIDIA/Megatron-LM PR 实际验证：

| PR | 标题 | 合 dev | 合 main | 目标 release | 备注 |
|---|---|---|---|---|---|
| [#2614](https://github.com/NVIDIA/Megatron-LM/pull/2614) | GDN context parallel | 2025-12-19 | — | 0.16 milestone | 走 head-parallel via all-to-all (Mamba 同思路) |
| [#2642](https://github.com/NVIDIA/Megatron-LM/pull/2642) | #2614 main 分支版本 | — | **2026-04-13** | **0.17.0** | 我们要拿的 release 代码 |
| [#2644](https://github.com/NVIDIA/Megatron-LM/pull/2644) | GDN THD packed | 2026-04-07 | — | 0.16 milestone | Qwen3.5+THD 有 NaN，不能用 |

**决策**：走 BSHD-CP（PR #2614/#2642）路径。**避坑** PR #2644（THD）：要 cuDNN ≥9.19，且 Qwen3.5+fused attention+THD 已知数值不稳，作者建议临时切 flash backend。

### PR #2614/#2642 的实现切面

上游只改了 3 个文件：
- `megatron/core/ssm/gated_delta_net.py` — CP 逻辑直接写进模块的 forward 里
- `megatron/core/transformer/transformer_config.py` — 新增配置项
- `tests/unit_tests/ssm/test_gated_delta_net.py` — 测试重构

**调用方契约**（直接读 0.17 main GDN 源码确认）：
- GDN `__init__` 接 `pg_collection: ProcessGroupCollection`，从 `pg_collection.cp.size()` 拿 CP 大小
- All-to-all 函数: `tensor_a2a_cp2hp`（CP → head-parallel）, `tensor_a2a_hp2cp`（再切回 CP）
- `sharded_state_dict` 已更新，加了 `'dp_cp_group'`
- 调用方启用方式: `--context-parallel-size N`（或等价配置）
- **没有外部 wrapper** —— 调用方只设 CP 大小，GDN 内部自己处理切分/通信

### PyPI 版本时间线

| 版本 | 发布日 | 含 GDN CP | 备注 |
|---|---|---|---|
| 0.15.0 | 2025-12-18 | ❌ | |
| 0.15.3 | 2026-02-06 | ❌ | |
| 0.16.0 | 2026-02-26 | **❌** (PR #2642 还没合) | **当前在用** |
| 0.16.1 | 2026-03-20 | ❌ (仍早于 0.17) | |
| **0.17.0** | **2026-04-16** | **✅** (PR #2642 在 2026-04-13 合入) | **升级目标** |

0.17.0 唯一 breaking change: Python 3.10 deprecated。我们用 3.12 — 不影响。

### mbridge 兼容性（已重新调研，**风险大幅降低**）

**关键发现 — mbridge 升级路径清晰**：

- **mbridge `main` 分支持续维护中**（最新 commit 2026-04-24）
- mbridge main `pyproject.toml` 只要求 `megatron-core>=0.12.0`，开放上限，**会自动接受 0.17.0**
- 最近 commit 明确做 Qwen3.5 + 新 mcore 适配：
  - 2026-04-24: "adapt to new mcore for qwen35 mtp"
  - 2026-04-23: "support mtp layer support for qwen3.5 series models"
  - 2026-03-31: "remove deprecated async_tensor_model_parallel_allreduce in mcore"
  - 2026-03-18: "support latest megatron that removes ModelType.encoder_and_decoder"
- mbridge main 已有专门的 `qwen3_5/` 子目录 + `Qwen3_5VlBridge` / `Qwen3_5MoeVlBridge` 类（v0.15.1 release 没有）
- mbridge config 里已有 `"cp_comm_type": "p2p"` 的设置，说明 CP 通信链路至少已有基础

**结论**：直接把 mbridge 从 v0.15.1 (Sep 2025) 升到 main 分支某个 commit 即可，**不需要切 NVIDIA Megatron-Bridge**（那是更大的重构）。

### AReaL 现有 Qwen3.5 适配代码

通过 grep 验证：
- [areal/models/mcore/registry.py:125-150](areal/models/mcore/registry.py) 注册了 `Qwen3_5MoeForConditionalGeneration` 和 `Qwen3_5ForConditionalGeneration`
- [areal/models/mcore/hf_save.py:204-624](areal/models/mcore/hf_save.py) 有 `_qwen3_5_moe_fallback_expert_export` —— **专门处理 mbridge 0.15.1 expert 导出返回空的 bug**
- [areal/engine/megatron_engine.py:1265,1340](areal/engine/megatron_engine.py) 走 `_mbridge_convert_to_hf` (vanilla mbridge fallback) 当 model_type 含 "qwen3_5"
- [areal/engine/megatron_engine.py:283,346](areal/engine/megatron_engine.py) 调 `disable_qwen3_5_incompatible_fusions` 关 5 个 fusion

**风险**：现有 13 个 Qwen3.5 patches（含 `_qwen3_5_moe_fallback_expert_export` 等）是基于 mbridge 0.15.1 + megatron 0.16 的具体 bug 写的。**升级到 mbridge main + megatron 0.17 后，部分 patches 可能变得不必要（或反而冲突）** —— 需要 smoke test 时一一回归。

### AReaL 自身的 CP 代码路径

[areal/engine/megatron_utils/packed_context_parallel.py](areal/engine/megatron_utils/packed_context_parallel.py)：
- 当前逻辑：基于 `cu_seqlens` 做 2*CP zigzag 切分（**只支持 THD**）
- BSHD 分支（cu_seqlens=None）：只是直接调 `model(...)`，前后处理都跳过 — **没有 all-gather**

[areal/engine/megatron_engine.py:1643-1647](areal/engine/megatron_engine.py#L1643-L1647) 在 `pad_to_maximum and cp_size > 1` 时硬抛错：
```python
if self.config.pad_to_maximum and cp_size > 1:
    raise ValueError("pad_to_maximum=True is incompatible with context_parallel_size>1; ...")
```

**Megatron 0.17 GDN CP 通后，这个 guard 就过时了** — 但仍需要：
1. 删 guard
2. 给 BSHD 路径补 logit 跨 CP all-gather（last stage 之前要把分散的序列收回完整）

### 战略结论

路径 = **Megatron 升级 + mbridge 升级 + AReaL 少量改动**。

**最坏情况退路**：mbridge main 在 35B 上回归不通过 → 再考虑切 NVIDIA Megatron-Bridge（但概率低，因为 mbridge 已明确做 Qwen3.5 + 新 mcore 适配）。

---

## 架构: 32K 内存预算

| 维度 | 16K (当前 OOM) | 16K + CP=2 | 32K + CP=4 |
|---|---|---|---|
| 单 rank seq tokens | 18432 | **9216** | 8192 |
| LM-head logit 申请 | 15.31 GB | **7.6 GB** | 6.8 GB |
| compute_logp 总峰值 (actor + ref) | OOM (84 GB) | ~70 GB | ~70 GB |
| 拓扑 | d4p4t2 \| e8t1 (32 GPU) | **d2p4t2c2 \| e8t1** (32 GPU) | **d1p4t2c4 \| e8t1** (32 GPU) |
| Actor world | 32 | 32 (DP=2 → DP=2 不变, CP 加进来) | 32 (DP=1) |

CP 不消耗额外 actor GPU —— 它是吃 DP。32K + CP=4 时 DP=1，单批梯度噪声升，需要靠 `ppo_n_minibatches` 调大补样本（256/128 = 1 sample/rank 极限情况）。

---

## 技术栈
- megatron-core: 0.16.0 → **0.17.0**
- mbridge: 0.15.1 → **main 分支某 commit (≥ 2026-04-24)**
- AReaL: 改 [packed_context_parallel.py](areal/engine/megatron_utils/packed_context_parallel.py) + 删 [megatron_engine.py:1643-1647](areal/engine/megatron_engine.py#L1643-L1647) guard

---

## 任务

### Task 1: megatron-core 0.17 + mbridge main smoke test (no CP) ✅ 已完成

**结果**：smoke test 通过。megatron-core 0.17.0 + mbridge main commit + AReaL 13 个 Qwen3.5 patches 全部兼容。

#### 改动 (commit `36ff9f1`)

- [`pyproject.toml`](pyproject.toml#L158-L162): `megatron-core==0.16.0 → 0.17.0`
- [`areal_fuyao_qwen3_5.dockerfile`](areal_fuyao_qwen3_5.dockerfile): 在 veRL base 上 `pip install --upgrade --no-deps megatron-core==0.17.0` + mbridge git@310e8fb

#### 首轮镜像 v22 失败 (bifrost-2026042809371801)

```
TypeError: Qwen3_5VLTransformerConfig.__init__() got an unexpected keyword argument
  'async_tensor_model_parallel_allreduce'
```

**根因**：dockerfile 用 `pip install --upgrade` 时 pip 不识别 git URL 装出的 mbridge 与 PyPI 0.15.1 的版本差异（版本号同为 "0.15.1"）→ **跳过实际重装**。结果 megatron 升到 0.17（已删该字段），但 mbridge 仍是 0.15.1（其 `_build_base_config` 字面量含已删字段）→ TypeError。

#### 修复 (commit `b56bc62`)

dockerfile 改为 **uninstall → install**（不再用 `--upgrade`），加 `--no-cache-dir` 防 wheel 缓存复用，build 末尾打印实际版本 + 自动检查 `async_tensor_model_parallel_allreduce` 不在 mbridge 源码里：

```dockerfile
RUN pip uninstall -y megatron-core mbridge \
    && pip install --no-deps --no-cache-dir megatron-core==0.17.0 \
    && pip install --no-deps --no-cache-dir \
       "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@310e8fb..."
RUN pip show megatron-core mbridge \
    && python -c "from mbridge.core.llm_bridge import LLMBridge; ..."
```

#### 二轮镜像 v23-260427-1849 验证通过

任务 `bifrost-2026042809573001-zengbw1`：
- ✅ Megatron init / mbridge.AutoBridge.from_pretrained 全过
- ✅ 11 个 train step 跑完, update_weights → rollout → train 完整循环工作
- ✅ 性能：mfu=0.0673, throughput=160 tok/gpu/s, step_time=25.8s, update_weights=5.67s
- ✅ 没有 NaN, OOM, ImportError, AttributeError

**关键经验**：未来 dockerfile 升级 git+url 装的包，**永远用 uninstall+install 替代 --upgrade**。

### Task 2: mbridge 兼容性决策（已跳过 — Task 1 顺利通过）

实际不需要切 NVIDIA Megatron-Bridge。mbridge main 完全胜任。

### Task 1.5: 切到 megatron-core git+main commit (Apr 13 PR #2642)

**触发原因**：v23 smoke test 通过后，Task 3 调研发现 `megatron-core==0.17.0` PyPI release tag **不含** GDN CP 代码（PR #2642 在 release branch cut 之后才合入 main）。`core_v0.17.0` tag 的 `gated_delta_net.py` 仍然是 `# TODO: Implement GatedDeltaNetContextParallel`。

**改动** (commit `d068697`)：
- dockerfile 把 `megatron-core==0.17.0` 替换为 `megatron-core @ git+...@20ba03fec03ebaec050c6bc7e79b77a4b4b5c000`
- 该 commit 是 PR #2642 的 merge commit (Apr 13, 2026, CI 已验证)
- 选这个 commit 而不是更新的 main HEAD：避免 Apr-19 MambaModel→HybridModel rename、Apr-22 DDP refactoring (#3812) 等后续可能引入的 break
- 内部 version 是 0.18.0 (main 已进入下个迭代周期)
- build 末尾加 `assert 'self.cp_size = self.pg_collection.cp.size()' in src` 强制验证 GDN CP 代码存在；缺则 build fail-fast

**待执行**：构建 v24 镜像 + 跑 0.8B smoke test（不开 CP）确认升级后现有流程仍可跑。

### Task 4: 删 BSHD-CP fail-fast guard ✅ 已完成

**改动** (commit `2f786e6`)：
- 删 [megatron_engine.py:1643-1647](areal/engine/megatron_engine.py#L1643-L1647) 的 `pad_to_maximum + cp_size > 1` ValueError
- 替换为详细注释说明 Megatron 0.18 dev (commit 20ba03f) 已支持 BSHD CP

### Task 5 调研发现（pending，等 v24 镜像）

调研 [megatron 0.18 GDN test](https://github.com/NVIDIA/Megatron-LM/blob/20ba03f.../tests/unit_tests/ssm/test_gated_delta_net.py)、[GPTModel.forward](https://github.com/NVIDIA/Megatron-LM/blob/20ba03f.../megatron/core/models/gpt/gpt_model.py)、[mamba_context_parallel.py](https://github.com/NVIDIA/Megatron-LM/blob/20ba03f.../megatron/core/ssm/mamba_context_parallel.py) 后，**核心不确定点**：

| 问题 | 现状 | 待确认 |
|---|---|---|
| GDN forward 输入约定 | 单测显示 SBH `[S/cp, B, H]` | mbridge 在 embedding 后转 BSH→SBH？还是 caller 要预切？ |
| GPTModel 是否内部做 CP split | `embedding(input_ids=...)` 直接喂 caller 给的 shape | 看起来不做，需 caller 预切 input_ids |
| RoPE 是否需要 cp_group | `rotary_pos_emb(..., cp_group=packed_seq_params.cp_group)` | **必须**传 `packed_seq_params` 含 cp_group，BSHD 也是 |
| 输出形状 | 单测输出仍是 `S/cp` 长度 | last PP stage logits 需 caller all-gather 回完整 S |

**Task 5 实现选项**：

A. **Caller-managed**（保守，类比现有 THD 路径）
- preprocess: `[B, S, ...]` → `[B, S/cp, ...]` zigzag split
- 构造 `PackedSeqParams(qkv_format="bshd", cp_group=...)` 传给 model.forward
- postprocess: `[B, S/cp, V]` all-gather across cp → `[B, S, V]`
- 工作量: ~100 行新代码 + 单测

B. **Model-managed**（乐观）
- 假设 mbridge / Megatron GPTModel 内部已经处理 BSHD CP split/gather
- AReaL 仅传 `[B, S]` 给 model，输出已是 `[B, S, V]`
- 风险: 实测可能 shape mismatch

**决策路径**：等 v24 镜像 build 完后，**先跑一次 cp_size=2 实验**（无 Task 5 改动），看实际错误现象：
- 若 model.forward shape error → 需要选项 A 实现
- 若 model 跑通但 logits 错（数值 diff vs cp=1） → 需要 postprocess all-gather
- 若 model 跑通且 logits 正确 → 选项 B 成立，Task 5 啥都不用做

避免盲写代码改错方向。

### Task 3: 把 pg_collection.cp 通到模型构造

**文件**:
- 读: AReaL 的 mbridge AutoBridge.from_pretrained 调用点 [areal/engine/megatron_engine.py:245](areal/engine/megatron_engine.py#L245)
- 读: mbridge 的 Qwen3.5 model factory（找出哪个文件构造含 GDN 的 TransformerLayer）
- 可能改: AReaL 或 mbridge，把 `pg_collection.cp = mpu.get_context_parallel_group()` 接好

- [ ] **Step 1: 跟踪 pg_collection 构造路径**

读 [areal/engine/megatron_engine.py:140-300](areal/engine/megatron_engine.py#L140-L300) 的 `MegatronEngine.initialize`，记录 ProcessGroupCollection 在哪儿构造（很可能从 `mpu.get_*_group()` 拼），有没有把 `cp` 也包进去。

- [ ] **Step 2: 若 cp group 没接好，补上**

在构造点设 `pg_collection.cp = mpu.get_context_parallel_group()`，再传给 mbridge。

- [ ] **Step 3: debug 打印验证**

在 GDN 模块 init 加临时 log（patch 一行 megatron 源码），确认 `pg_collection.cp.size()` == YAML 里的 `context_parallel_size`。验证完移除。

### Task 4: 删 BSHD-CP fail-fast guard

**文件**:
- 改: [areal/engine/megatron_engine.py:1640-1647](areal/engine/megatron_engine.py#L1640-L1647)

- [ ] **Step 1: 把 guard 换成注释 + 版本说明**

```python
# Megatron 0.17+ 原生支持 GDN BSHD CP (PR #2614/#2642)。
# 旧的 fail-fast (pad_to_maximum + cp_size > 1) 已过时。
# AReaL 的 packed_context_parallel_forward BSHD 分支直接调 Megatron 模型；
# GDN 内部的 CP all-to-all 处理 seq shard。
```

- [ ] **Step 2: Commit**

```bash
git commit -m "feat(megatron): 允许 CP 与 pad_to_maximum 共存 (Megatron 0.17 GDN CP)"
```

### Task 5: 给 BSHD 后处理补 logit 跨 CP all-gather

**文件**:
- 改: [areal/engine/megatron_utils/packed_context_parallel.py](areal/engine/megatron_utils/packed_context_parallel.py)

- [ ] **Step 1: 定位 BSHD 输出处理**

`postprocess_packed_seqs_context_parallel` 当前 [packed_context_parallel.py:80-81](areal/engine/megatron_utils/packed_context_parallel.py#L80-L81):
```python
if cp_size <= 1 or cu_seqlens is None:
    return output.squeeze(0)
```
对 BSHD + CP > 1 直接跳过 all-gather —— 在 Megatron 0.17 之后这是错的。

- [ ] **Step 2: 加 BSHD CP 后处理**

当 `cp_size > 1` 且 `cu_seqlens is None`:
- 每个 rank 输出 shape `[B, S/cp_size, ...]`（经过 GDN 内部 CP 处理后）
- 用 `dist.all_gather` 跨 CP group + `torch.cat(dim=1)` 拼回 `[B, S, ...]`
- (备选: 用 Megatron 内置的 `gather_from_context_parallel_region`，对 non-pipeline-last stages 视情况)

- [ ] **Step 3: 0.8B 验证 (cp_size=2, pad_to_maximum=true)**

加单测 — 同一输入，`cp_size=1` 的输出和 `cp_size=2` 的输出在 bf16 容差内必须一致。

- [ ] **Step 4: Commit**

```bash
git commit -m "fix(megatron): BSHD CP 后处理 all-gather (Megatron 0.17 GDN CP)"
```

### Task 6: 0.8B 数值一致性验证

**文件**:
- 创建: `tests/test_qwen3_5_cp_correctness.py`

- [ ] **Step 1: 写失败的 test**

```python
def test_cp2_matches_cp1_logits():
    # 用同样输入分别跑 cp_size=1 和 cp_size=2
    # 断言 logits 最大绝对差 < 1e-2 (bf16 容差)
```

- [ ] **Step 2: 8 GPU 0.8B 任务跑 (单节点, TP=2 CP=2)**

如果数值不过 → 调试后处理 gather 逻辑。

### Task 7: 35B 16K + CP=2 (内存验证)

**文件**:
- 创建: `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_seq16k_cp2.yaml`

- [ ] **Step 1: 写 yaml**

Backend: `megatron:(attn:d2p4t2c2|ffn:e8t1)` (DP=2 PP=4 TP=2 CP=2; ffn ep*tp*pp=32 ✓)

- [ ] **Step 2: 提交任务并验证**

- compute_logp 通过（LM head 不再 OOM）
- actor 峰值 < 60 GB; ref onload + activation 总和 < 80 GB

### Task 8: 35B 32K + CP=4 (终极目标)

**文件**:
- 创建: `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_seq32k_cp4.yaml`

Backend: `(attn:d1p4t2c4|ffn:e8t1)` (DP=1 PP=4 TP=2 CP=4)
- [ ] 改 `max_new_tokens: 16384 → 32768`, `max_tokens: 18432 → 34816`
- [ ] 端到端跑通

---

## 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| mbridge main 有 ABI 变化与 AReaL Qwen3.5 patches 冲突 | 中 | 阻塞 Task 1 | smoke test 时逐一排查；可能要删过时 patches |
| `pg_collection.cp` 没被 mbridge 构造时通进去 | 中 | 阻塞 Task 3 | 在 AReaL 端 inject cp group |
| BSHD CP all-gather 顺序有微妙 bug | 中 | 训练时 logits 错 | Task 6 数值测必须抓出来 |
| Qwen3.5 GDN CP 合了但默认关闭 | 低 | 没 speedup（但不报错） | 通过 Megatron debug print 确认 |
| AReaL 的 13 个 Qwen3.5 patches 在 0.17 失效 | 中 | 各种 crash | Task 1 Step 4 smoke test 兜底 |

---

## 自检

- [x] Spec 覆盖: Task 1-8 覆盖升级、通路、正确性、内存验证、扩展。
- [x] 占位符扫描: 没有 TBD/TODO；Task 2 是条件性任务，已标。
- [x] 类型一致: pg_collection, cp_size 用法一致；后端字符串遵循 AReaL 现有语法 `(attn:dXpYtZcN|ffn:eMtK)`。

## 执行

Task 1-2 串行（环境准备）。Task 3 依赖 Task 1。Task 4-5 独立。Task 6 依赖 4-5。Task 7-8 依赖 6。

本分支与 `feat/weight-update-bucket` 互不影响 — 两条线可以并行推进，分别合 `main`。
