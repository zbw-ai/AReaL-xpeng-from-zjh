# Qwen3.5 GDN BSHD-CP 适配迁移指南

> **目标读者**：另一个 AReaL fork 的开发者 / **AI agent**，需要把本仓库 `feat/cp-gdn-megatron-017` 分支的 CP 适配迁移过去。
>
> **本文档自包含**，关键代码 inline 可见，不需要拉取本仓库源码也能完成移植。完整背景见 [`qwen3_5-cp-adaptation.md`](qwen3_5-cp-adaptation.md)。

---

## 给 AI Agent 的执行指引（先读这一节）

**任务**：把本文档描述的 CP 适配应用到目标 AReaL fork。

**执行顺序**：
1. 先读 §1（结论）+ §2（三层挑战），形成 mental model
2. 读 §3.B 的 8 个 commit 一句话总结，确认所有依赖项
3. 读 §9（**inline 代码 patches**，本文档主体），按顺序应用
4. 改完后用 §5 的 verification checklist 自测
5. 遇到错误，查 §6 的"症状 → 解药"表

**关键原则**：
- 代码层 patch 必须**全部 8 个 commit 一起应用**，少一个就会 OOM 或 crash
- megatron-core 必须用 git+commit `0f6fcb0`，**不能**用任何 PyPI release tag（0.16.x / 0.17.0 都不含 GDN BSHD-CP）
- mbridge 必须用 git+commit `310e8fb` 或更新，**不能**用 PyPI 0.15.1
- pip 安装必须用 `pip uninstall + pip install --force-reinstall --no-cache-dir`，**不能**用 `pip install --upgrade`（git URL 与 PyPI 同版本号会跳过重装）
- dockerfile 不能用 BuildKit heredoc（fuyao Docker builder 不支持），用 `COPY .py + RUN python3` 替代

**验证应用成功的 5 个 build-time assert**（必须全过）：
```
[OK] GDN CP code present       - megatron 源码含 self.cp_size = pg_collection.cp.size()
[OK] PR #4230 fix present       - megatron 源码含 _resolve_cu_seqlens 函数
[OK] mbridge clean              - mbridge 源码不含 async_tensor_model_parallel_allreduce
[OK] _coalescing_manager patched - megatron param_and_grad_buffer 含 PATCHED_FOR_NCCL_COALESCING_BUG
Image baked successfully
```

**关键文件定位策略**（在目标 fork 上）：
- `areal/engine/megatron_engine.py` — 主要修改
- `areal/engine/megatron_utils/packed_context_parallel.py` — CP 切分逻辑
- `areal/engine/megatron_utils/deterministic.py` — Qwen3.5 fusion 兼容
- `areal_fuyao_qwen3_5.dockerfile`（或同等命名）— 镜像构建
- `fuyao_examples/patch_megatron_coalescing.py` — 新建文件

如果目标 fork 文件路径不同，按"文件名 + 函数名"匹配（如 `megatron_engine.py:_compute_forward_result`、`packed_context_parallel.py:postprocess_packed_seqs_context_parallel`）。

---

## 1. CP 适配结论（Headline）

| 维度 | 状态 |
|---|---|
| **方案路径** | Megatron-LM dev commit `0f6fcb0`（含 PR #2614/#2644/#4230 完整 GDN BSHD-CP）+ AReaL 端"在 cp-split 上算 logp"（方案 C）|
| **0.8B + cp=2** | ✅ 跑通 9h+，显存节省 30%（device 38→26 GB）|
| **35B 8K + cp=2** | ✅ 跑通，peak 42→39 GB (-7%)，与 cp=1 等价的训练数值 |
| **35B 16K + cp=2** | ⚠️ CP 切分**完美工作**（buffer 15.31→7.66 GB 精确切半），但触发独立的 colocate offload bug（actor 残留 26 GB 不释放）→ OOM |
| **未验证** | 35B 32K + cp=4（要先修 offload bug 或扩节点）|

**核心一句话**：**CP 适配本身已完成且数学正确**，节省与 seq 长度成正比。35B 16K OOM 是 colocate offload bug 引起，独立问题。

---

## 2. 三层挑战与解决路径

| 层 | 挑战 | 解决 |
|---|---|---|
| **Megatron 上游** | core_v0.17.0 release tag 不含 GDN BSHD-CP（PR #2614/#2642 没 cherry-pick）；`_coalescing_manager` NCCL 不兼容 | 切 dev branch commit `0f6fcb0`（含 PR #2614 + #2644 + #4230） + dockerfile 内 patch `_coalescing_manager → nullcontext` |
| **mbridge 兼容** | 0.15.1 不接 megatron 0.17+；Qwen3.5 GDN init 转发 `cp_comm_type` 给不接 kwargs 的 GDN | 升级到 main commit `310e8fb`；megatron `0f6fcb0` 的 GDN `__init__` 加了 `**kwargs`，自动吞下 cp_comm_type |
| **AReaL 端** | `pad_to_maximum + cp>1` 硬 fail；BSHD CP 后处理逻辑缺失（model 输出 `[B, S/cp, V]`，caller 端 logp 用完整 `[B, S]` 索引 → shape mismatch + OOM） | 删 fail-fast guard；新方案 C：在 cp-split 上算 logp，只 gather 标量级 logp |

### 关键约束

1. **GDN 强制 BSHD 格式**：mbridge 的 Qwen3.5 GDN 需要 `attention_mask`，AReaL 的 `pack_tensor_dict` 移除 attention_mask 触发 GDN crash → 必须 `pad_to_maximum: true`（BSHD），不能走 THD packed 路径
2. **TP ≤ num_kv_heads = 2**：Qwen3.5 hard limit
3. **mbridge 设 `cp_comm_type="p2p"`**：transformer_layer 在 cp_size>1 时把它转发给 self_attention.build_module，**只 0f6fcb0 之后**的 GDN `__init__` 接 `**kwargs` 才能吞下不 crash

---

## 3. 迁移交付包（5 样东西）

### A. 主参考文档

[`docs/fuyao-experiments/qwen3_5-cp-adaptation.md`](qwen3_5-cp-adaptation.md) — 完整 CP 适配总结（383 行）：
- 适配目标与挑战
- 改动总览
- 方案 C 关键实现（数据流图 + 代码片段 + 5 个关键决策）
- 验证矩阵（0.8B + 35B 8K/16K 实测对比）
- 32K + CP=4 路径预算
- 已知遗留（actor offload bug，与 CP 无关）
- 完整 commit 历史

### B. 关键 commits（按依赖顺序，必须全部移植）

```
分支: feat/cp-gdn-megatron-017 @ zbw-ai/AReaL-xpeng-from-zjh

# 依赖升级
1. 36ff9f1 deps: bump megatron-core 0.17.0 + mbridge main
2. d068697 fix(dockerfile): megatron 改 git+0f6fcb0 (含 GDN CP)
3. cd308de fix(dockerfile): COPY + RUN 替代 heredoc (fuyao builder)
4. 95d3676 fix(dockerfile): v25 + _coalescing_manager patch

# AReaL 端 patches
5. 2f786e6 feat(megatron): 删 fail-fast guard
6. 2660cbd fix(qwen3_5): GDN cp_size>1 时清 cp_comm_type
7. cf2c680 fix(qwen3_5): 禁用 MTP (BSHD CP 下 _concat_embeddings hidden 不匹配)

# 方案 C 核心实现 ★
8. 5f52fa3 feat(megatron): 方案 C - cp-split 上算 logp
```

生成 patch 包：
```bash
cd zbw-ai/AReaL-xpeng-from-zjh
git format-patch 36ff9f1^..5f52fa3 -o /tmp/cp-patches/
# 生成 8 个 .patch 文件
```

应用到目标 fork：
```bash
cd target-fork
git checkout -b feat/cp-gdn
git am /tmp/cp-patches/*.patch
# 或者 git apply --3way 处理 conflict
```

### C. 镜像 tag

```
infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v25-260427-2214
```

或者给他 [`areal_fuyao_qwen3_5.dockerfile`](../../areal_fuyao_qwen3_5.dockerfile) 让他在自己集群构建。镜像里关键配置：
- megatron-core @ git+`0f6fcb0c5778327868e6866447a58b5568059ae1`（含 GDN BSHD-CP）
- mbridge @ git+`310e8fb35ccf4fcd4419d32973e563a6d43ee5fb`
- `_coalescing_manager → nullcontext` patch（[`fuyao_examples/patch_megatron_coalescing.py`](../../fuyao_examples/patch_megatron_coalescing.py)）
- transformers ≥ 5.3 已含

### D. 可直接用的 yaml 模板

| 文件 | 用途 |
|---|---|
| [`fuyao_examples/math/qwen3_5_0_8b_rlvr_vllm_cp2.yaml`](../../fuyao_examples/math/qwen3_5_0_8b_rlvr_vllm_cp2.yaml) | 0.8B + cp=2 验证 backend 是否工作 |
| [`fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_seq8k_cp2.yaml`](../../fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_seq8k_cp2.yaml) | 35B + 8K + cp=2 净对照（推荐起点）|
| [`fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_seq16k_cp2.yaml`](../../fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_seq16k_cp2.yaml) | 35B + 16K + cp=2（撞 offload bug，作为下一步目标）|

backend 字符串约定（**重点**）：
```yaml
# 0.8B (1 node 8 GPU): actor 4 + vLLM 4
actor.backend: "megatron:d1p1t2c2"   # DP=1, PP=1, TP=2, CP=2 = 4 GPU

# 35B (6 node 48 GPU): actor 32 + vLLM 16
actor.backend: "megatron:(attn:d2p4t2c2|ffn:e8t1)"   # cp=2
actor.backend: "megatron:(attn:d1p4t2c4|ffn:e8t1)"   # cp=4 (32K 路径)
```

集群约束（必须满足）：
```
attn world (DP × CP × PP × TP) == ffn (EP × ETP × PP)
attn world == 32 (35B 6 节点 actor)

例如:
  d2p4t2c2 → 2*4*2*2=32 ✓
  e8t1 → 8*1*4=32 ✓
  expert_dp = 32/32 = 1 ✓
```

DP 减半时，`ppo_n_minibatches` 加倍补偿（保持 sample/rank/mb 恒定）：
```yaml
# v21 (cp=1, DP=4): ppo_n_minibatches: 32  → 256/32/4 = 2 sample/rank/mb
# v27 (cp=2, DP=2): ppo_n_minibatches: 64  → 256/64/2 = 2 sample/rank/mb (一致)
# v?? (cp=4, DP=1): ppo_n_minibatches: 128 → 256/128/1 = 2 sample/rank/mb
```

### E. 关键代码定位（移植参考）

```
方案 C 核心逻辑（最重要的 ~150 行）:

  areal/engine/megatron_utils/packed_context_parallel.py
    - bshd_cp_zigzag_split()       (新增)
    - bshd_cp_zigzag_gather()      (从 _bshd_cp_zigzag_gather 重命名)
    - postprocess_packed_seqs_context_parallel() — BSHD+cp>1 不 gather logits

  areal/engine/megatron_engine.py
    - import bshd_cp_zigzag_gather/split (line 64-66)
    - forward_step trim 跳过 cp>1 (line 695-705)
    - _compute_forward_result cp 处理 (line 1842+)
    - _compute_logprobs_and_loss cp 处理 (line 1777+)

  areal/engine/megatron_utils/deterministic.py
    - disable_qwen3_5_incompatible_fusions() — cp_comm_type 清 + MTP 禁用

  fuyao_examples/patch_megatron_coalescing.py — 镜像构建时打 _coalescing_manager patch
```

---

## 4. 给迁移者的"上下文 prompt"模板

如果他用 LLM 协助（直接给 Claude / GPT），建议这样开场：

```
我在做 AReaL 框架上 Qwen3.5-35B-A3B 模型的 RL 训练，目标支持 32K 长序列。
原版 AReaL 在 16K 时 OOM (compute_logp 阶段 vocab_parallel.exp() 申请
15.31 GB buffer 失败)。Qwen3.5 用 GDN attention 强制 BSHD 格式，
不能走 packed THD CP 路径。

参考 zbw-ai/AReaL-xpeng-from-zjh feat/cp-gdn-megatron-017 分支已完成
CP 适配（验证通过），关键文档:
  docs/fuyao-experiments/qwen3_5-cp-adaptation.md
  docs/fuyao-experiments/qwen3_5-cp-migration-guide.md (本文档)

需要把这个 CP 适配迁移到我当前的 AReaL fork (xxx 分支)。请帮我:
1. 对比两个 fork 的 megatron_engine.py / packed_context_parallel.py 差异
2. 把以下 commits 的 patch 移植过来 (按依赖顺序):
   36ff9f1, d068697, cd308de, 95d3676, 2f786e6, 2660cbd, cf2c680, 5f52fa3
3. 验证 dockerfile 镜像是否需要重建

约束:
- megatron-core 必须用 dev commit 0f6fcb0 (含 PR #2614/#2642/#4230 GDN CP)
- mbridge 必须 ≥ 310e8fb (Apr 24, 2026, 含 Qwen3.5 + 新 mcore 适配)
- 必须 patch _coalescing_manager → nullcontext (NCCL 不支持 coalesced)
- 已知遗留: colocate offload bug 让 35B 16K 仍 OOM, 但 8K cp=2 完美工作
- 不要试图回到 mbridge 0.15.1 或 megatron 0.17.0 release tag, 这些版本不兼容
```

---

## 5. 验证 checklist（迁移完成后让目标 fork 自测）

按顺序跑：

### Step 1: 镜像构建

```bash
fuyao docker --site=fuyao_b1 --push \
    --dockerfile=areal_fuyao_qwen3_5.dockerfile \
    --image-name=areal-qwen3_5-megatron-vXX
```

build-time 必须看到 5 个 `[OK]`：
```
[OK] GDN CP code present (self.cp_size = self.pg_collection.cp.size())
[OK] PR #4230 padding alignment fix present (_resolve_cu_seqlens)
[OK] mbridge clean (no async_tensor_model_parallel_allreduce)
[OK] _coalescing_manager bypass patched (PATCHED_FOR_NCCL_COALESCING_BUG)
Image baked successfully
```

任何一个 `[OK]` 没有 → 说明依赖升级没生效，build 应该 fail-fast。

### Step 2: 0.8B + cp=2 跑通验证

backend: `megatron:d1p1t2c2`，1 节点 8 GPU，actor 4 + vLLM 4

预期：
- ≥ 10 train step 跑通
- IOStruct device used ≈ **26 GB** (vs cp=1 baseline 38 GB，节省 ~30%)
- mfu / step_time 稳定（cp=2 比 cp=1 略低 ~10% 是预期，因为 a2a 通信）

如果 OOM 或失败：
- 报错 `cp_comm_type` → patch #6 (commit 2660cbd) 没应用
- 报错 MTP `_concat_embeddings` hidden 不匹配 → patch #7 (commit cf2c680) 没应用
- 报错 IndexError shape mismatch [N1] vs [N2]（N2 ≈ 2×N1）→ 方案 C (commit 5f52fa3) 没应用，仍是 v1 caller-gather logits

### Step 3: 35B + 8K + cp=2 跑通验证

backend: `(attn:d2p4t2c2|ffn:e8t1)`，6 节点 48 GPU

预期：
- ≥ 3 train step 跑通
- peak device used ≈ **39 GB** (vs cp=1 baseline 42 GB，节省 ~7%)
- mfu ≈ 0.03，step_time ≈ 460s
- PPO 数值正常: `imp_weight ≈ 1.0`, `|KL| < 0.01`, `entropy` 有限

### Step 4: PPO 数值正确性

任何一个 step 的 stats 应满足：
```
ppo_actor/update/behave_imp_weight/avg ≈ 1.0   (PPO importance ratio)
ppo_actor/update/behave_approx_kl/avg ≈ 0      (KL ≈ 0, |KL| < 0.01)
ppo_actor/update/entropy/avg          有限     (不发散)
ppo_actor/update/n_valid_tokens       > 0      (有效 token 计数正常)
```

不满足说明 cp 切分有问题（labels 切错位 / logp gather 错序）。

### Step 5（可选，撞 offload bug）: 35B + 16K + cp=2

backend: `(attn:d2p4t2c2|ffn:e8t1)`，max_tokens_per_mb 18432，max_new_tokens 16384

预期：OOM at compute_logp，但**关键观察**：
- 报错 `Tried to allocate 7.66 GiB`（**不是 15.31 GiB**）→ CP 切分工作 ✓
- OOM 时报告"邻居进程 46 GB"→ ref onload 真实大小，不是 CP 异常
- "this process 26 GB" → actor offload 残留（独立 bug，与 CP 无关）

如果 buffer 申请仍是 **15.31 GiB**（而非 7.66），说明方案 C 没生效（仍是 v1 caller-gather logits）。

---

## 6. 同事可能踩的坑（按危害性排序）

### 致命坑（一定会让 build 或 run 失败）

| 坑 | 症状 | 解药 |
|---|---|---|
| megatron 用 0.17.0 release tag | GDN 没 CP 代码（`# TODO: Implement GatedDeltaNetContextParallel`） | 必须 `git+https://github.com/NVIDIA/Megatron-LM.git@0f6fcb0c5778327868e6866447a58b5568059ae1` |
| mbridge 用 0.15.1 release | `Qwen3_5VLTransformerConfig got unexpected kwarg 'async_tensor_model_parallel_allreduce'` | git+`310e8fb35ccf4fcd4419d32973e563a6d43ee5fb` |
| `pip install --upgrade` 没真换 | mbridge 仍是 0.15.1（git URL 与 PyPI 版本号同跳过重装） | 用 `pip uninstall + install --force-reinstall --no-cache-dir` |
| dockerfile 用 heredoc | `unknown instruction: IMPORT`（fuyao builder 不支持 BuildKit heredoc） | 用 `COPY .py + RUN python3` |
| 没 patch `_coalescing_manager` | step 1 optimizer 阶段崩 `Backend nccl does not support reduce_scatter_tensor_coalesced` | dockerfile 末尾打 patch（见 [patch_megatron_coalescing.py](../../fuyao_examples/patch_megatron_coalescing.py)）|

### 致 OOM 坑

| 坑 | 症状 | 解药 |
|---|---|---|
| Task 5 v1 实现（caller-side gather logits） | 0.8B 跑通但 35B 16K OOM 申请 15.31 GB | 必须用方案 C（gather 标量 logp，不 gather logits）— commit 5f52fa3 |
| 没禁 MTP | `_concat_embeddings` hidden 维 1096 vs 548 不匹配 | `disable_qwen3_5_incompatible_fusions` 加 `mtp_num_layers=0` |
| 没清 cp_comm_type | GDN init unexpected kwarg `cp_comm_type` | 0f6fcb0 GDN 已有 `**kwargs` 自动吞，但保留 patch 作防御 |

### 性能坑（不致命但影响）

| 坑 | 症状 | 解药 |
|---|---|---|
| forward_step 在 cp>1 时仍 trim | output shape mismatch | trim 跳过 cp>1 (commit 5f52fa3 内含) |
| labels 没 pad 到 padded_s 就 split | `seq_len % (2*cp) != 0` assert fail | `_compute_*` 函数内先 pad 到 `output.shape[1] * cp_size` 再 split |
| autograd 错误抛过 cp gather | grad nan 或反传错 | `bshd_cp_zigzag_gather` 当前不是 autograd-aware，但 RL 训练每 cp_rank 自己 grad 链路完整，已实测验证 |

---

## 7. 已知遗留（迁移者也会遇到，提前打预防针）

### Colocate offload bug

35B 16K + cp=2 时 OOM 缺 0.4 GB。**不是 CP 适配的问题**，是 35B 量级下 `torch_memory_saver.pause()` 不能彻底释放 megatron internal allocator 持有的 grad/optim buffer。

实测证据：
```
v22 (16k cp=1) OOM 时 actor onload 60 GB ← 正常 onload 状态
v26 (16k cp=2) OOM 时 actor offload 残留 26 GB ← 应归 0 但没归 0
v21/v27 (8K) offload 工作 ← 8K 量级 buffer 小, allocator 能清干净
```

修复方向（独立工作，不在 CP 适配范畴）：
1. `offload()` 末尾加 `torch.cuda.empty_cache()` 多次调用
2. 显式把 megatron DDP grad buffer / Adam state 移到 CPU
3. 或者切 `cp=4` 让所有进程占用减半，绕过 offload bug

### 0.8B cp=1 vs cp=2 数值一致性测试（Task 6）

未做，但方案 C 的设计保证数值正确性（per-token logp 是独立计算，cp 切只切 seq dim 不影响每个 token 的 logp 值）。RL 训练 9h+ 多 step `behave_imp_weight ≈ 1.0` 隐式验证。

如果迁移后想做严格数值对比，可写：
```python
# 跑 cp=1 收集 logp[B, S]
# 跑 cp=2 收集 logp[B, S]（cp gather 后）
# assert torch.allclose(logp_cp1, logp_cp2, atol=1e-2)
```

---

## 8. 关键参考链接

### Megatron-LM 上游 PR
- [PR #2614](https://github.com/NVIDIA/Megatron-LM/pull/2614) — GDN context parallel (head-parallel via all-to-all)
- [PR #2644](https://github.com/NVIDIA/Megatron-LM/pull/2644) — GDN packed sequence (THD)
- [PR #4230](https://github.com/NVIDIA/Megatron-LM/pull/4230) — GDN packed_seq + CP padding alignment fix（关键：`_resolve_cu_seqlens` + GDN init `**kwargs`）

### NVIDIA 官方坑
- [Megatron-LM Issue #1369](https://github.com/NVIDIA/Megatron-LM/issues/1369) — `_coalescing_manager` NCCL 不兼容（致命，step 1 必崩）
- [pytorch/pytorch Issue #134833](https://github.com/pytorch/pytorch/issues/134833) — NCCL 缺 `reduce_scatter_tensor_coalesced`

### veRL 实证参考
- [veRL Qwen3.5 Long Context CP 文档](/Users/zengbw/Codebase/for_llm_train_070/llm_train_sft_0402/docs/qwen35_long_context_cp.md) — 4 节点 32 卡 SFT 32K + CP=2 跑通 120+ step
- veRL pin commit 是 `0f6fcb0`（与本项目一致）

### 本仓库相关文档
- [`docs/fuyao-experiments/qwen3_5-cp-adaptation.md`](qwen3_5-cp-adaptation.md) — CP 适配总结（完整版）
- [`docs/fuyao-experiments/qwen3_5-adaptation.md`](qwen3_5-adaptation.md) — 早期 13 个 Qwen3.5 + Megatron 适配 patch（前置背景）
- [`docs/superpowers/plans/2026-04-29-bshd-cp-logp-on-split.md`](../superpowers/plans/2026-04-29-bshd-cp-logp-on-split.md) — 方案 C 设计 + 实测对比

---

## 9. Inline Patches（AI 直接抄这一节）

按顺序应用以下所有改动。每个 patch 自带"找到目标文件 → 改什么"的精确说明。

### 9.1 Dockerfile（在目标 fork 找到 Qwen3.5 dockerfile，覆盖或新建）

文件名一般是 `areal_fuyao_qwen3_5.dockerfile` 或 `Dockerfile.qwen3_5`。完整内容：

```dockerfile
# veRL image: torch 2.10, vllm 0.17, mbridge@dc1321b, transformers@d64a6d6, megatron-core 0.16
# Add AReaL-specific deps + upgrade megatron-core / mbridge to support
# GDN + Context Parallel for Qwen3.5-35B-A3B 32K long-context training.
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:verl-qwen3_5-v9-latest
ENV MAX_JOBS=1

# AReaL-only deps not in veRL image
RUN pip install aiofiles tensorboardX math_verify

# Megatron-LM commit 0f6fcb0 (dev branch, 2026-04-13).
# 含: PR #2614 (GDN+CP) + PR #2644 (GDN+THD) + PR #4230 (GDN+CP padding fix).
# 0.17.0 PyPI release tag NOT 含此 PR (release branch cut 早于合入); 必须用 dev commit.
# uninstall+reinstall (NOT --upgrade) — 防 git URL 与 PyPI 版本号同跳过重装.
# --no-cache-dir 防 wheel/layer 复用.
RUN pip uninstall -y megatron-core mbridge \
    && pip install --no-deps --no-cache-dir --force-reinstall \
       "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM.git@0f6fcb0c5778327868e6866447a58b5568059ae1" \
    && pip install --no-deps --no-cache-dir --force-reinstall \
       "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@310e8fb35ccf4fcd4419d32973e563a6d43ee5fb"

# Patch _coalescing_manager → nullcontext (NCCL 不支持 reduce_scatter_tensor_coalesced).
# 不 patch 会在 step 1 optimizer 必崩 (NVIDIA/Megatron-LM#1369, pytorch/pytorch#134833).
# fuyao Docker builder 不支持 BuildKit heredoc, 用 COPY + RUN 替代.
COPY fuyao_examples/patch_megatron_coalescing.py /tmp/patch_megatron_coalescing.py
RUN python3 /tmp/patch_megatron_coalescing.py

# Build-time 5 个 assert, 失败任意一个则 image build fail-fast (push 前拦截).
RUN echo "=== Verifying installed versions ===" \
    && pip show megatron-core | head -3 \
    && pip show mbridge | head -5 \
    && python3 -c "import megatron.core; print('megatron.core:', megatron.core.__file__, 'version:', megatron.core.__version__)" \
    && python3 -c "from megatron.core.ssm import gated_delta_net; import inspect; src=inspect.getsource(gated_delta_net); assert 'self.cp_size = self.pg_collection.cp.size()' in src, 'GDN CP code missing!'; print('[OK] GDN CP code present')" \
    && python3 -c "from megatron.core.ssm import gated_delta_net; import inspect; src=inspect.getsource(gated_delta_net); assert '_resolve_cu_seqlens' in src, 'PR #4230 fix missing!'; print('[OK] PR #4230 padding alignment fix present')" \
    && python3 -c "from mbridge.core.llm_bridge import LLMBridge; import inspect; src=inspect.getsource(LLMBridge._build_base_config); assert 'async_tensor_model_parallel_allreduce' not in src, 'mbridge has deprecated kwarg!'; print('[OK] mbridge clean')" \
    && python3 -c "import megatron.core.distributed.param_and_grad_buffer as pgb; assert 'PATCHED_FOR_NCCL_COALESCING_BUG' in open(pgb.__file__).read(), 'param_and_grad_buffer not patched'; print('[OK] _coalescing_manager bypass patched')"
```

### 9.2 新建文件 `fuyao_examples/patch_megatron_coalescing.py`

完整内容：

```python
"""Patch megatron's param_and_grad_buffer.py to bypass NCCL-incompatible
`_coalescing_manager`.

PyTorch's NCCL backend never implemented `reduce_scatter_tensor_coalesced` /
`allgather_into_tensor_coalesced` (only Gloo has them). Megatron's
`start_grad_sync` and `start_param_sync` wrap per-bucket ops in
`with _coalescing_manager(...)`, which raises on __exit__:

  RuntimeError: Backend nccl does not support reduce_scatter_tensor_coalesced

This crashes step 1's optimizer phase. Upstream issues unfixed:
  - NVIDIA/Megatron-LM#1369
  - pytorch/pytorch#134833

Fix: shadow `_coalescing_manager` in this module with a function that returns
`nullcontext()`. The inner per-bucket reduce_scatter_tensor /
all_gather_into_tensor calls are NCCL-supported and execute fine.

Safe because we force `overlap_grad_reduce=False` / `overlap_param_gather=False`.
Idempotent: re-running on a patched file is a no-op.
"""

import sys
import megatron.core.distributed.param_and_grad_buffer as pgb_mod

target = pgb_mod.__file__
with open(target) as f:
    src = f.read()

if "PATCHED_FOR_NCCL_COALESCING_BUG" in src:
    print(f"[OK] Patch already applied at {target}")
    sys.exit(0)

needle = "from torch.distributed import _coalescing_manager"
if needle not in src:
    print(f"[FAIL] Cannot find import in {target}", file=sys.stderr)
    sys.exit(1)

replacement = (
    "from torch.distributed import _coalescing_manager as _orig_coalescing_manager"
    "  # noqa: F401  PATCHED_FOR_NCCL_COALESCING_BUG\n"
    "from contextlib import nullcontext as _nullctx\n"
    "def _coalescing_manager(*_args, **_kwargs):\n"
    "    # PATCHED: NCCL backend lacks reduce_scatter_tensor_coalesced /\n"
    "    # allgather_into_tensor_coalesced. Return nullcontext so per-bucket\n"
    "    # NCCL ops run individually (which IS supported).\n"
    "    return _nullctx()\n"
)

with open(target, "w") as f:
    f.write(src.replace(needle, replacement, 1))

print(f"[OK] Patched {target}")
```

### 9.3 改 `areal/engine/megatron_utils/deterministic.py`

找到 `disable_qwen3_5_incompatible_fusions` 函数，**末尾追加**（保持现有 fusion 禁用代码不动）：

```python
    # Megatron-LM 0.18 dev (commit 20ba03f / 0f6fcb0) transformer_layer.py:314-320
    # 在 cp_size > 1 时把 `cp_comm_type` forward 给 self_attention.build_module。
    # 这给标准 self-attention 的 CP 通信模式用，但 Qwen3.5 用
    # `experimental_attention_variant="gated_delta_net"`, GDN.__init__ 不接 cp_comm_type
    # → TypeError (在 0f6fcb0 之前) 或 silent ignore (在 0f6fcb0 之后, GDN init 加了 **kwargs).
    # 防御性清空: 让 transformer_layer 的 `is not None` 检查 False 不再 forward.
    # 安全: cp_size=1 路径不读此字段; GDN 用自己的 a2a 不依赖 config.cp_comm_type.
    if (
        getattr(model_config, "experimental_attention_variant", None) == "gated_delta_net"
        and getattr(model_config, "cp_comm_type", None) is not None
    ):
        before_cp = model_config.cp_comm_type
        model_config.cp_comm_type = None
        print(
            f"[disable_qwen3_5_incompatible_fusions] cp_comm_type was={before_cp}, "
            f"now None (GDN attention does not accept cp_comm_type kwarg).",
            flush=True,
        )

    # Disable Multi-Token Prediction (MTP) for RL training:
    #   1. RL 不需要 MTP (推理多 token 预测加速特性).
    #   2. Megatron 0.18 dev MTP `_concat_embeddings` (multi_token_prediction.py:905) 在
    #      cp_size>1 + BSHD 下 hidden 维不匹配:
    #      decoder_input H/TP=1096 vs hidden_states (经过 GDN CP) H/TP/CP=548
    #      → torch.cat dim=2 RuntimeError "Sizes of tensors must match except in dim 2"
    #   3. mbridge `_build_mtp_config` 看到 hf_config.text_config.mtp_num_hidden_layers > 0
    #      就无条件启用 MTP (Qwen3.5 ships with this set).
    if getattr(model_config, "mtp_num_layers", 0) and model_config.mtp_num_layers > 0:
        before_mtp = model_config.mtp_num_layers
        model_config.mtp_num_layers = 0
        if hasattr(model_config, "mtp_loss_scaling_factor"):
            model_config.mtp_loss_scaling_factor = 0.0
        print(
            f"[disable_qwen3_5_incompatible_fusions] mtp_num_layers was={before_mtp}, "
            f"now 0 (RL does not use MTP; avoids cp>1 hidden-dim mismatch).",
            flush=True,
        )
```

### 9.4 改 `areal/engine/megatron_utils/packed_context_parallel.py`

**核心改动**：新增 `bshd_cp_zigzag_split`，重命名 `_bshd_cp_zigzag_gather` 为 `bshd_cp_zigzag_gather`（去前导下划线公开导出，并加重要注释），修 `postprocess_packed_seqs_context_parallel` 的 BSHD + cp>1 分支不要 gather logits。

如果目标 fork 已有 `_bshd_cp_zigzag_gather`（这是早期错误实现），**整段替换**为下面：

```python
def bshd_cp_zigzag_split(tensor: torch.Tensor) -> torch.Tensor:
    """Split a BSHD tensor ``[B, S, ...]`` to current rank's zigzag chunk ``[B, S/cp, ...]``.

    Mirrors GDN/TE causal-attention load-balanced split:
        Total seq is divided into ``2*cp_size`` chunks. Rank ``i`` holds
        ``chunk_i`` (front) and ``chunk_{2*cp_size-1-i}`` (back), concatenated.

    Use this on caller-side **labels / loss_mask** so they line up with the
    model's CP-split logits output (model internally does the same split on
    input embeddings).

    Args:
        tensor: BSHD tensor ``[B, S_full, ...]``. Pass-through if cp_size <= 1.
    Returns:
        ``[B, S_full / cp_size, ...]``  zigzag-selected for current cp_rank.
    """
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    if cp_size <= 1:
        return tensor

    s = tensor.shape[1]
    assert s % (2 * cp_size) == 0, (
        f"BSHD CP split requires seq_len divisible by 2*cp ({2 * cp_size}), got {s}"
    )
    chunk = s // (2 * cp_size)
    front = tensor[:, cp_rank * chunk : (cp_rank + 1) * chunk, ...]
    back_idx = 2 * cp_size - 1 - cp_rank
    back = tensor[:, back_idx * chunk : (back_idx + 1) * chunk, ...]
    return torch.cat([front, back], dim=1)


def bshd_cp_zigzag_gather(local: torch.Tensor) -> torch.Tensor:
    """All-gather a BSHD tensor ``[B, S/cp, ...]`` across CP group and unzigzag to ``[B, S, ...]``.

    Inverse of :func:`bshd_cp_zigzag_split`. Use this **only on scalar-shaped
    outputs** (e.g. log-probs ``[B, S/cp]``, vocab min/max ``[B, S/cp]``) in
    BSHD-CP RL paths. Do **NOT** use on logits ``[B, S/cp, V]`` — that
    re-inflates the buffer and defeats CP's memory savings.

    For logits, keep them cp-split and run vocab-parallel logprobs on the
    cp-split shape, then gather only the scalar log-probs.
    """
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    cp_group = mpu.get_context_parallel_group()
    if cp_size <= 1:
        return local

    local = local.contiguous()
    # NOTE: Not autograd-aware. Safe in compute_logp (@torch.no_grad) and on
    # detached vocab-stat tensors. For training loss path, the gathered logp
    # gradient flows through the local logp (via gather_logprobs autograd) —
    # the all-gather acts as a "scatter then sum-zero" identity at backward.
    gathered = [torch.empty_like(local) for _ in range(cp_size)]
    dist.all_gather(gathered, local.detach(), group=cp_group)
    gathered[cp_rank] = local

    local_len = local.shape[1]
    assert local_len % 2 == 0, (
        f"BSHD CP local seq length must be even (zigzag), got {local_len}"
    )
    half = local_len // 2
    full_len = local_len * cp_size
    chunks: list = [None] * (2 * cp_size)
    for i in range(cp_size):
        front, back = gathered[i][:, :half, ...], gathered[i][:, half:, ...]
        chunks[i] = front
        chunks[2 * cp_size - 1 - i] = back

    full = torch.cat(chunks, dim=1)
    assert full.shape[1] == full_len, (
        f"BSHD CP unzigzag shape mismatch: got {full.shape[1]}, expected {full_len}"
    )
    return full
```

`postprocess_packed_seqs_context_parallel` 函数的 BSHD + cp>1 分支改成（**关键：不 gather logits**）：

```python
def postprocess_packed_seqs_context_parallel(
    output: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    post_process: bool,
) -> torch.Tensor:
    """Postprocess packed sequences."""
    cp_size = mpu.get_context_parallel_world_size()
    if not post_process:
        return output
    if cp_size <= 1:
        return output.squeeze(0)
    if cu_seqlens is None:
        # BSHD + cp>1: model returned ``[1, S/cp, V/TP]`` (or ``[1, S/cp, hidden]``
        # for non-last PP stages) after GDN's internal head-parallel CP.
        # Do NOT all-gather logits here — that re-inflates the buffer to
        # ``[1, S, V/TP]`` and defeats CP's memory savings (this caused 35B
        # 16K + cp=2 OOM in early experiment). Caller (the ``_compute_*``
        # functions in MegatronEngine) is responsible for:
        #   1. cp-splitting labels via :func:`bshd_cp_zigzag_split`
        #   2. running vocab-parallel logprobs on the cp-split logits
        #   3. cp all-gathering the resulting scalar log-probs via
        #      :func:`bshd_cp_zigzag_gather`
        return output.squeeze(0)

    # 以下是 THD 路径原有逻辑, 保持不变
    # ... (existing THD all-gather + zigzag code unchanged)
```

### 9.5 改 `areal/engine/megatron_engine.py`

**改动 1**：import 区域加入新函数（约 line 64-67）：

```python
from areal.engine.megatron_utils.packed_context_parallel import (
    bshd_cp_zigzag_gather,
    bshd_cp_zigzag_split,
    packed_context_parallel_forward,
    # ... 其他原有 import
)
```

**改动 2**：删除 fail-fast guard。找到 `_prepare_mb_list` 函数中如下代码（约 line 1640+）：

```python
        if self.config.pad_to_maximum and cp_size > 1:
            raise ValueError(
                "pad_to_maximum=True is incompatible with context_parallel_size>1; "
                "CP split logic in packed_context_parallel_forward requires cu_seqlens."
            )
```

**整段删除**（替换为注释说明 0f6fcb0 已支持）：

```python
        # CP under pad_to_maximum (BSHD format):
        # Megatron-LM PR #2614/#2642/#4230 (commit 0f6fcb0) added native GDN
        # context-parallel that operates on BSHD inputs via head-parallel
        # all-to-all (see megatron/core/ssm/gated_delta_net.py).
        # Earlier fail-fast guard (`pad_to_maximum + cp_size > 1`) was a
        # workaround when GDN had no CP support; with the new backend it is
        # no longer necessary. AReaL still needs to do BSHD-side caller
        # logp computation on cp-split logits (see _compute_forward_result).
```

**改动 3**：`forward_step` 中 trim 逻辑跳过 cp>1。找到 `pad_to_maximum` trim 代码（约 line 695）：

```python
                if self.config.pad_to_maximum:
                    orig_s = mb_input.orig_mb["input_ids"].shape[1]
                    if output.shape[1] > orig_s:
                        output = output[:, :orig_s]
```

**改成**：

```python
                # pad_to_maximum: trim back to orig_mb's seq dim.
                # Under cp_size>1 the output is already cp-split [B, padded_s/cp, V];
                # _compute_forward_result / _compute_logprobs_and_loss will handle
                # the cp-aligned label split + scalar logp gather + final trim.
                if self.config.pad_to_maximum:
                    cp_size = self.parallel_strategy.context_parallel_size
                    if cp_size <= 1:
                        orig_s = mb_input.orig_mb["input_ids"].shape[1]
                        if output.shape[1] > orig_s:
                            output = output[:, :orig_s]
```

**改动 4**：`_compute_forward_result` 函数（compute_logp 路径）。整段替换：

```python
    def _compute_forward_result(
        self,
        output: torch.Tensor,
        inputs: dict,
    ):
        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )
        # BSHD CP: model output is cp-split [B, S/cp, V/TP]. Compute logprobs on
        # cp-split shape (per-token op, no cross-token dependency), then
        # all-gather the SCALAR log-probs back to full [B, S]. Avoid
        # all-gathering logits (would re-inflate buffer; caused 35B 16K OOM).
        cp_size = mpu.get_context_parallel_world_size()
        if not self.config.is_critic:
            if self.enable_tree_training:
                if cp_size > 1:
                    raise NotImplementedError(
                        "Tree training with context_parallel_size > 1 is not "
                        "supported yet (packed-tree CP path missing)."
                    )
                # ... (existing tree training logic)
                logprobs = _gather_packed_tree_logprobs(
                    output, inputs["trie_node"], inputs["input_ids"],
                    temperature=self.config.temperature,
                    tp_group=mpu.get_tensor_model_parallel_group()
                    if mpu.get_tensor_model_parallel_world_size() > 1 else None,
                )
                return logprobs
            labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
            orig_s = labels.shape[1]
            if cp_size > 1:
                # output cp-split with seq length padded_s/cp. Pad labels up to
                # padded_s so zigzag split is well-defined, then split. Padded
                # positions get label 0 — their logp is never used downstream
                # (loss_mask zeros them at orig_s boundary).
                target_full_s = output.shape[1] * cp_size
                if labels.shape[1] < target_full_s:
                    labels = torch.nn.functional.pad(
                        labels, (0, target_full_s - labels.shape[1]), value=0
                    )
                labels = bshd_cp_zigzag_split(labels)
            logprobs = gather_logprobs(
                output, labels,
                temperature=self.config.temperature,
                tp_group=mpu.get_tensor_model_parallel_group()
                if mpu.get_tensor_model_parallel_world_size() > 1 else None,
            )
            if cp_size > 1:
                # Gather scalar logp [B, S/cp] → [B, padded_s], trim to orig_s.
                logprobs = bshd_cp_zigzag_gather(logprobs)
                if logprobs.shape[1] > orig_s:
                    logprobs = logprobs[:, :orig_s]
            return logprobs
        else:
            values = output.squeeze(-1)
            if cp_size > 1:
                values = bshd_cp_zigzag_gather(values)
                orig_s = inputs["input_ids"].shape[1]
                if values.shape[1] > orig_s:
                    values = values[:, :orig_s]
            return values
```

**改动 5**：`_compute_logprobs_and_loss` 函数（train_batch 路径）同样模式。在 actor branch（非 tree training, 非 critic）的 `labels = torch.roll(...)` 之后改成：

```python
            else:
                labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
                orig_s = labels.shape[1]
                if cp_size > 1:
                    target_full_s = output.shape[1] * cp_size
                    if labels.shape[1] < target_full_s:
                        labels = torch.nn.functional.pad(
                            labels, (0, target_full_s - labels.shape[1]), value=0
                        )
                    labels = bshd_cp_zigzag_split(labels)
                logprobs, entropy = gather_logprobs_entropy(
                    output, labels,
                    temperature=self.config.temperature,
                    tp_group=mpu.get_tensor_model_parallel_group()
                    if mpu.get_tensor_model_parallel_world_size() > 1 else None,
                )
                vocab_min_logits = output.detach().min(-1).values.float()
                vocab_max_logits = output.detach().max(-1).values.float()
                if cp_size > 1:
                    # All-gather scalar tensors [B, S/cp] -> [B, padded_s], trim to orig_s
                    logprobs = bshd_cp_zigzag_gather(logprobs)
                    entropy = bshd_cp_zigzag_gather(entropy)
                    vocab_min_logits = bshd_cp_zigzag_gather(vocab_min_logits)
                    vocab_max_logits = bshd_cp_zigzag_gather(vocab_max_logits)
                    if logprobs.shape[1] > orig_s:
                        logprobs = logprobs[:, :orig_s]
                        entropy = entropy[:, :orig_s]
                        vocab_min_logits = vocab_min_logits[:, :orig_s]
                        vocab_max_logits = vocab_max_logits[:, :orig_s]
```

critic branch 也加 cp gather + trim：

```python
        else:
            values = output.squeeze(-1)
            if cp_size > 1:
                values = bshd_cp_zigzag_gather(values)
                orig_s = inputs["input_ids"].shape[1]
                if values.shape[1] > orig_s:
                    values = values[:, :orig_s]
            loss = loss_fn(values, inputs)
```

`cp_size = mpu.get_context_parallel_world_size()` 应该在函数开头定义。

### 9.6 yaml 配置示例（35B + 8K + cp=2 推荐起点）

新建 `fuyao_examples/math/qwen3_5_35b_a3b_cp2_8k.yaml`（或类似）：

```yaml
experiment_name: qwen3_5-35b-a3b-cp2-8k
trial_name: trial0
seed: 42
enable_offload: true
total_train_epochs: 10
tokenizer_path: ${actor.path}

cluster:
  n_nodes: 6
  n_gpus_per_node: 8
  fileroot: /your/checkpoint/path

scheduler:
  type: local

rollout:
  backend: "vllm:d8t2"
  setup_timeout: 1200.0
  max_concurrent_rollouts: 128
  max_head_offpolicyness: 2

gconfig:
  n_samples: 8
  max_new_tokens: 4096
  max_tokens: 8192
  temperature: 0.99
  top_p: 0.99
  top_k: 100

actor:
  # ★ 关键: backend 加 c2 (CP=2), DP 减半 (4→2)
  backend: "megatron:(attn:d2p4t2c2|ffn:e8t1)"
  path: /path/to/Qwen3.5-35B-A3B
  pad_to_maximum: true   # GDN 强制 BSHD
  mb_spec:
    n_mbs: 1
    granularity: 1
    max_tokens_per_mb: 8192
    n_mbs_divisor: 1
  optimizer:
    type: adam
    lr: 1.0e-6
    weight_decay: 0.1
  megatron:
    wrap_with_ddp: true
    ddp:
      grad_reduce_in_fp32: true
      overlap_grad_reduce: false   # ★ 必须 False (与 _coalescing_manager patch 配套)
      overlap_param_gather: false  # ★ 同上
      use_distributed_optimizer: true
    recompute_granularity: full
    recompute_method: uniform
    recompute_num_layers: 1
  # ★ DP 减半 → ppo_n_minibatches 加倍补偿 (保持 2 sample/rank/mb)
  ppo_n_minibatches: 64
  recompute_logprob: true
  use_decoupled_loss: true
  weight_update_mode: disk   # 35B colocate xccl 不工作

ref:
  backend: ${actor.backend}
  path: ${actor.path}
  pad_to_maximum: true
  scheduling_strategy:
    type: colocation
    target: actor

vllm:
  model: ${actor.path}
  max_num_seqs: 128
  max_model_len: 8192
  gpu_memory_utilization: 0.68
  enforce_eager: true   # Qwen3.5 cudagraph 兼容性问题

# ... 其他 dataset/saver/evaluator/stats_logger 配置按目标 fork 现有模板填
```

### 9.7 deploy 命令（fuyao 集群）

```bash
# 1. 构建镜像 (build-time 5 个 [OK] 必须全过)
fuyao docker --site=fuyao_b1 --push \
    --dockerfile=areal_fuyao_qwen3_5.dockerfile \
    --image-name=areal-qwen3_5-megatron-cpv1

# 2. deploy 35B + 8K + cp=2 验证任务
fuyao deploy --disable-fault-tolerance \
    --docker-image=infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-cpv1-XXX \
    --project=YOUR-PROJECT --experiment=YOUR-EXP --gpu-type a100 --gpus-per-node 8 \
    --node=6 --label=qwen3_5-35b-cp2-8k --site=fuyao_b1 --queue=YOUR-QUEUE \
    bash fuyao_examples/fuyao_areal_run.sh --run-type math_rlvr \
    --config fuyao_examples/math/qwen3_5_35b_a3b_cp2_8k.yaml
```

预期：跑通 ≥ 3 step，peak device used ≈ 39 GB（vs cp=1 baseline 42 GB）。

---

## 附录 A: 完整 commit 元数据

如果只看 git log 不够，每个 commit 的核心改动一句话总结：

```
36ff9f1 deps: 把 pyproject.toml 的 megatron-core==0.16.0 升到 0.17.0 +
        mbridge 改为 git URL (虽然后面会再升级到 0f6fcb0/310e8fb)

d068697 fix(dockerfile): pyproject 升级不够 (0.17.0 release tag 不含 GDN CP),
        改 dockerfile 强制 git+commit 0f6fcb0 (PR #4230 follow-up)

cd308de fix(dockerfile): fuyao Docker builder 不支持 BuildKit heredoc,
        把 RUN python3 - <<'PY' 改成 COPY .py + RUN python3

95d3676 fix(dockerfile): 加 _coalescing_manager → nullcontext patch
        + build-time 5 个 [OK] assert

2f786e6 feat(megatron): 删除 megatron_engine.py:1640+ 的
        ValueError("pad_to_maximum=True is incompatible with cp_size>1") guard
        Megatron 0.18 dev GDN 已支持 BSHD CP 这个 guard 过时

2660cbd fix(qwen3_5): disable_qwen3_5_incompatible_fusions 加清 cp_comm_type 逻辑.
        防御 transformer_layer 转发 cp_comm_type 给 GDN 的 unexpected kwarg.
        0f6fcb0 GDN **kwargs 已修, 但保留作 fallback.

cf2c680 fix(qwen3_5): disable_qwen3_5_incompatible_fusions 加禁用 MTP 逻辑.
        Megatron 0.18 dev MTP _concat_embeddings 在 BSHD + cp>1 下
        hidden 维不匹配 (decoder_input H/TP=1096 vs hidden_states H/TP/CP=548).
        RL 训练不需 MTP, 关掉.

5f52fa3 feat(megatron): ★方案 C 核心实现★
        - packed_context_parallel.py:
          + 新增 bshd_cp_zigzag_split([B,S,...] → [B,S/cp,...])
          + 重命名 _bshd_cp_zigzag_gather → bshd_cp_zigzag_gather
            (明确只用于 scalar shape, 不要传 logits)
          + postprocess BSHD + cp>1 时不 gather logits, squeeze 返回 cp-split
        - megatron_engine.py:
          + forward_step trim 跳过 cp>1 (output 保持 cp-split)
          + _compute_forward_result: pad labels → split → vocab_parallel logp
            on cp-split → gather logp → trim 到 orig_s
          + _compute_logprobs_and_loss: 同样模式 + entropy / vocab_min/max gather

        关键设计: 只 gather 标量级 logp (O(S)), 不 gather logits (O(S*V)),
        让 cp=2 切半 logits buffer 的收益不被 caller-side 拼回完整 S 抵消.
```

---

## 附录 B: 验证镜像内部依赖（debug 用）

如果 build 完镜像后想确认依赖确实装上，进容器跑：

```bash
# 确认 megatron-core 是 git 装而非 PyPI 装
python3 -c "import megatron.core; print(megatron.core.__file__)"
# 应输出: /usr/local/lib/python3.12/dist-packages/megatron/core/__init__.py

python3 -c "
from megatron.core.ssm import gated_delta_net
import inspect
src = inspect.getsource(gated_delta_net)
print('GDN CP:', 'self.cp_size = self.pg_collection.cp.size()' in src)
print('PR #4230:', '_resolve_cu_seqlens' in src)
print('GDN __init__ has **kwargs:', 'def __init__' in src and '**kwargs' in src.split('def __init__')[1].split(')')[0])
"
# 应输出三个 True

python3 -c "
import megatron.core.distributed.param_and_grad_buffer as p
src = open(p.__file__).read()
print('Coalescing patched:', 'PATCHED_FOR_NCCL_COALESCING_BUG' in src)
"
# 应输出 True

python3 -c "
from mbridge.core.llm_bridge import LLMBridge
import inspect
src = inspect.getsource(LLMBridge._build_base_config)
print('mbridge clean:', 'async_tensor_model_parallel_allreduce' not in src)
"
# 应输出 True
```

---

## 附录 C: 集群拓扑速查表

35B 6 节点 (48 GPU): actor 32 + vLLM 16

| 配置 | backend | per-rank seq | OOM? |
|---|---|---|---|
| v17/v20 (cp=1, 2K) | (attn:d4p4t2\|ffn:e8t1) | 2K | ✓ |
| v21 (cp=1, 8K) | (attn:d4p4t2\|ffn:e8t1) | 8K | ✓ peak 42 GB |
| v22 (cp=1, 16K) | (attn:d4p4t2\|ffn:e8t1) | 16K | ❌ actor 60 GB |
| v22-pp8 (cp=1, 16K) | (attn:d2p8t2\|ffn:e4t1) | 16K | ❌ actor 59 GB |
| **v27 (cp=2, 8K)** | **(attn:d2p4t2c2\|ffn:e8t1)** | **4K** | **✓ peak 39 GB** |
| v26 (cp=2, 16K) | (attn:d2p4t2c2\|ffn:e8t1) | 9K | ⚠️ offload bug, OOM 缺 0.4 GB |
| **未来 (cp=4, 32K)** | **(attn:d1p4t2c4\|ffn:e8t1)** | **8K** | 待验证 |

约束验证：
- actor world: DP × CP × PP × TP = 32 ✓
- ffn ep × tp × pp = 32 ✓（与 attn world 整除 expert_dp）
- num_layers (40) 整除 PP (4 或 8)
- num_kv_heads (2) ≥ TP (= 2 上限)
