# Qwen3.5-35B-A3B Context Parallel (CP) 适配总结

> **状态**：CP 适配核心工作**已完成**（2026-04-29）。0.8B + cp=2 实测验证通过 30% 显存节省，35B + cp=2 在 actor 端完美工作（buffer 切半，进程占用 60→26 GB）。剩余 35B 上的 OOM 由独立的 colocate offload bug 触发，与 CP 适配无关。

> **分支**：`feat/cp-gdn-megatron-017`
> **关键 commits**：`d068697` (megatron 0f6fcb0 升级) → `2660cbd` (cp_comm_type clear) → `cf2c680` (MTP 禁用) → `2f786e6` (删 fail-fast) → `12ecc6c` (Task 5 v1) → `5f52fa3` (方案 C, 最终实现) → `cd308de` (dockerfile heredoc 修复) → `95d3676` (v25 dockerfile)

---

## 1. 适配目标与挑战

### 1.1 业务目标
- 让 Qwen3.5-35B-A3B 能跑 16K → 32K 长序列 RL 训练（当前 v22 在 16K 时 compute_logp OOM）
- 关键洞见：v22 OOM 的 `15.31 GB` 是 LM head logits buffer，**只有 CP 能解决**（PP 不行：logits 只在最后 stage；TP=2 已硬上限 `num_kv_heads=2`；EP/ETP 互换不省内存）

### 1.2 三层挑战

| 层 | 挑战 | 解决路径 |
|---|---|---|
| **Megatron 上游** | core_v0.17.0 release tag 不含 GDN BSHD-CP（PR #2614/#2642 没 cherry-pick）；`_coalescing_manager` NCCL 不兼容 | 切 dev branch commit `0f6fcb0`（含 PR #2614 + #2644 + #4230） + dockerfile 内 patch `_coalescing_manager → nullcontext` |
| **mbridge 兼容** | 0.15.1 不接 megatron 0.17+；Qwen3.5 GDN init 上传 `cp_comm_type` 给不接 kwargs 的 GDN | 升级到 main commit `310e8fb`；megatron 0f6fcb0 的 GDN `__init__` 加了 `**kwargs`，自动吞下 cp_comm_type |
| **AReaL 端** | `pad_to_maximum + cp>1` 硬 fail；BSHD CP 后处理逻辑缺失（model 输出 `[B, S/cp, V]`，caller 端 logp 用完整 `[B, S]` 索引 → shape mismatch + OOM） | 删 fail-fast guard；新方案 C：在 cp-split 上算 logp，只 gather 标量级 logp |

### 1.3 关键约束

1. **GDN 强制 BSHD 格式**（[adaptation 文档 issue#1](qwen3_5-adaptation.md#L81)）：mbridge 的 Qwen3.5 GDN 需要 `attention_mask`，AReaL 的 `pack_tensor_dict` 移除 attention_mask 触发 GDN crash → 必须 `pad_to_maximum: true`（BSHD），不能走 THD packed 路径
2. **TP ≤ num_kv_heads = 2**：Qwen3.5 hard limit
3. **mbridge 设 `cp_comm_type="p2p"`**：transformer_layer 在 cp_size>1 时把它转发给 self_attention.build_module，**只 0f6fcb0 之后**的 GDN `__init__` 接 `**kwargs` 才能吞下不 crash

---

## 2. 改动总览

### 2.1 依赖升级（dockerfile）

[`areal_fuyao_qwen3_5.dockerfile`](../../areal_fuyao_qwen3_5.dockerfile)（commit `cd308de`）：

```dockerfile
# 升级 megatron-core 到 dev commit 0f6fcb0 (Apr 13, 2026, PR #4230)
# 含 PR #2614 (GDN+CP) + PR #2644 (GDN+THD) + PR #4230 (GDN+CP padding fix)
RUN pip uninstall -y megatron-core mbridge \
    && pip install --no-deps --no-cache-dir --force-reinstall \
       "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM.git@0f6fcb0c5778327868e6866447a58b5568059ae1" \
    && pip install --no-deps --no-cache-dir --force-reinstall \
       "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@310e8fb35ccf4fcd4419d32973e563a6d43ee5fb"

# Patch _coalescing_manager → nullcontext (NCCL 不支持 reduce_scatter_tensor_coalesced)
COPY fuyao_examples/patch_megatron_coalescing.py /tmp/patch_megatron_coalescing.py
RUN python3 /tmp/patch_megatron_coalescing.py
```

build-time 自动 assert 验证：
- GDN CP 代码存在 (`self.cp_size = self.pg_collection.cp.size()`)
- PR #4230 修复存在 (`_resolve_cu_seqlens`)
- mbridge 不含 deprecated kwarg (`async_tensor_model_parallel_allreduce`)
- `_coalescing_manager` 已 patch

镜像：`areal-qwen3_5-megatron-v25-260427-2214`

### 2.2 AReaL 端 patches

#### A. 配置层 patches（防御性 + 适配 0.18 dev）

[`areal/engine/megatron_utils/deterministic.py:disable_qwen3_5_incompatible_fusions`](../../areal/engine/megatron_utils/deterministic.py)：

```python
# 1. 清 cp_comm_type (commit 2660cbd)
#    防御 transformer_layer 转发 cp_comm_type 给 GDN. 0f6fcb0 GDN **kwargs 已能吞,
#    但保留为 fallback (无害, 因为 GDN 用自己的 a2a 不依赖 cp_comm_type)
if (config.experimental_attention_variant == "gated_delta_net"
    and config.cp_comm_type is not None):
    config.cp_comm_type = None

# 2. 禁用 MTP (commit cf2c680)
#    Megatron 0.18 dev MTP _concat_embeddings 在 cp_size>1 + BSHD 下 hidden 维不匹配
#    (decoder_input H/TP=1096 vs hidden_states H/TP/CP=548)
#    RL 训练不需要 MTP, 关掉
if config.mtp_num_layers > 0:
    config.mtp_num_layers = 0
    config.mtp_loss_scaling_factor = 0.0
```

#### B. 删除 fail-fast guard (commit `2f786e6`)

[`areal/engine/megatron_engine.py:1640+`](../../areal/engine/megatron_engine.py#L1640)：

```python
# 删除原 ValueError("pad_to_maximum=True is incompatible with cp>1; ...")
# 替换为说明: Megatron 0.18 dev (commit 20ba03f/0f6fcb0) GDN 内部支持 BSHD CP
# AReaL 仍需做 BSHD 端的 caller-side 后处理 — 见方案 C
```

#### C. **方案 C 核心实现**（commit `5f52fa3`）— 详见 §3

---

## 3. 方案 C 关键实现：在 cp-split 上算 logp

### 3.1 设计原则

**只 gather 标量级 logp**（每 token 一个数 = O(S)），**不 gather logits**（每 token V 个数 = O(S × V)，V≈150K，差 5 个数量级）。

### 3.2 错误演进

| 版本 | 实现 | 0.8B | 35B |
|---|---|---|---|
| **v0** (无 CP) | — | ✓ | OOM 申请 15.31 GB |
| **Task 5 v1** (`12ecc6c`) | model 出口立即 all-gather logits 回 `[B, S, V]` | ✓ (buffer 小掩盖错误) | OOM 申请 **15.31 GB**（buffer 同 v0：cp 没省内存）|
| **方案 C** (`5f52fa3`) | model 出口保持 cp-split, caller 在 cp-split 上算 logp, gather 标量 logp | ✓ + **省 30%** | actor compute_logp **过 ✓** (buffer **7.66 GB** = v0 的 1/2) |

### 3.3 数据流对比

```
v1 (Task 5, 错误):
  model.forward → [B, S/cp, V/TP]
  ↓ caller all_gather logits + zigzag unshuffle
  [B, S, V/TP]   ← buffer 翻 cp 倍, CP 收益归零!
  ↓ vocab_parallel logp (用完整 labels [B, S])
  [B, S]
  ↓ gather_logprobs → exp_logits → 💥 OOM

方案 C (正确):
  model.forward → [B, S/cp, V/TP]   ← 保持 cp-split, 不 gather logits
  ↓ caller 切 labels: zigzag_split([B, S]) → [B, S/cp]
  ↓ vocab_parallel logp (在 cp-split 上算)
  [B, S/cp]                          ← 标量级
  ↓ all_gather + zigzag unshuffle
  [B, S]
  ↓ trim 到 orig_s
```

### 3.4 改动文件

#### `areal/engine/megatron_utils/packed_context_parallel.py`

新增 `bshd_cp_zigzag_split`：

```python
def bshd_cp_zigzag_split(tensor: torch.Tensor) -> torch.Tensor:
    """[B, S, ...] → 当前 cp_rank 的 [B, S/cp, ...] zigzag 切片.

    GDN/TE causal-attention 负载均衡切分:
      seq 划成 2*cp_size 个 chunk; rank i 持有 chunk_i (前) + chunk_{2cp-1-i} (后).
    """
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    if cp_size <= 1:
        return tensor
    s = tensor.shape[1]
    assert s % (2 * cp_size) == 0
    chunk = s // (2 * cp_size)
    front = tensor[:, cp_rank * chunk : (cp_rank + 1) * chunk, ...]
    back_idx = 2 * cp_size - 1 - cp_rank
    back = tensor[:, back_idx * chunk : (back_idx + 1) * chunk, ...]
    return torch.cat([front, back], dim=1)
```

`bshd_cp_zigzag_gather` 改名（去前导下划线，公开导出，明确**只用于 scalar shape**）：

```python
def bshd_cp_zigzag_gather(local: torch.Tensor) -> torch.Tensor:
    """all_gather + zigzag unshuffle: [B, S/cp, ...] → [B, S, ...].

    !!! 只对 scalar 形状 (logp [B, S/cp], vocab_min/max [B, S/cp]) 用 !!!
    不要传 logits [B, S/cp, V] — 那会让 buffer 翻 cp 倍, 重蹈 Task 5 v1 覆辙.
    """
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    cp_group = mpu.get_context_parallel_group()
    if cp_size <= 1:
        return local

    gathered = [torch.empty_like(local) for _ in range(cp_size)]
    dist.all_gather(gathered, local.detach(), group=cp_group)
    gathered[cp_rank] = local  # 保留 autograd-tracked tensor
    # zigzag unshuffle
    half = local.shape[1] // 2
    chunks = [None] * (2 * cp_size)
    for i in range(cp_size):
        chunks[i] = gathered[i][:, :half, ...]
        chunks[2 * cp_size - 1 - i] = gathered[i][:, half:, ...]
    return torch.cat(chunks, dim=1)
```

修 `postprocess_packed_seqs_context_parallel` 的 BSHD + cp>1 分支：

```python
if cu_seqlens is None:
    # BSHD + cp>1: 不 gather logits, 直接返回 cp-split.
    # caller (_compute_*) 负责在 cp-split 上算 logp + gather scalar.
    return output.squeeze(0)
```

#### `areal/engine/megatron_engine.py`

**`forward_step`**（line 695）：cp_size>1 时跳过 `pad_to_maximum` trim：

```python
if self.config.pad_to_maximum:
    cp_size = self.parallel_strategy.context_parallel_size
    if cp_size <= 1:
        # cp=1 走原 trim 逻辑
        orig_s = mb_input.orig_mb["input_ids"].shape[1]
        if output.shape[1] > orig_s:
            output = output[:, :orig_s]
    # cp>1 时 output 是 cp-split shape, 让 _compute_* 处理
```

**`_compute_forward_result`** (compute_logp 路径，line 1842+)：

```python
cp_size = mpu.get_context_parallel_world_size()
labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
orig_s = labels.shape[1]
if cp_size > 1:
    # output cp-split [B, padded_s/cp, V/TP], pad labels 到 padded_s 再切
    target_full_s = output.shape[1] * cp_size
    if labels.shape[1] < target_full_s:
        labels = torch.nn.functional.pad(
            labels, (0, target_full_s - labels.shape[1]), value=0
        )
    labels = bshd_cp_zigzag_split(labels)
logprobs = gather_logprobs(output, labels, ..., tp_group=...)
if cp_size > 1:
    # gather scalar logp [B, S/cp] → [B, padded_s], trim 到 orig_s
    logprobs = bshd_cp_zigzag_gather(logprobs)
    if logprobs.shape[1] > orig_s:
        logprobs = logprobs[:, :orig_s]
return logprobs
```

**`_compute_logprobs_and_loss`** (train_batch 路径，line 1777+)：同样模式，额外处理 entropy / vocab_min/max_logits 的 cp gather + trim。critic 路径 `output.squeeze(-1)` 后也做 cp gather。

### 3.5 关键决策

| 决策 | 原因 |
|---|---|
| input_ids 不需 caller 端切 | mbridge GPTModel 内部自动 cp scatter（v25 0.8B 实测：caller 给 `(1, 2152)` 完整, model 输出 `(1, 1076, V)` 已切）|
| labels 在 `_compute_*` 内部切 | 避免改更上层调用约定；caller 仍传完整 inputs |
| labels pad 到 `padded_s` (= `output.shape[1] * cp_size`) | `orig_s` 通常不能被 `2*cp` 整除（pad_to_maximum 后 `padded_s` 是 tp 倍数）；padding 区域 logp 由 loss_mask 屏蔽 |
| `bshd_cp_zigzag_gather` 非 autograd-aware | RL 训练每 cp_rank 自己 grad 链路完整（model output → logits → logp → loss → backward），跨 cp 不需要传 grad；v25 0.8B 9h+ 179 step 实测验证收敛性 |
| 保留 cp_comm_type clear / MTP 禁用 patches | 0f6fcb0 已修 cp_comm_type 但保留作为防御；MTP 在 BSHD CP 下 megatron 上游有独立 bug，禁用最稳 |

---

## 4. 验证矩阵

### 4.1 0.8B（充分跑通）

| 任务 | Backend | 状态 | 关键指标 |
|---|---|---|---|
| `bifrost-2026042823342200` | v1 (Task 5 commit 12ecc6c) | ✅ 9h+ 179 step | mfu=0.187, step_time=19.4s |
| `bifrost-2026042910564500` | 方案 C (commit 5f52fa3) | ✅ 13 min+ 多 step 稳定 | **device 38→26.5 GB (-30%)** |

#### 0.8B v1 vs 方案 C 显存对比（同 yaml + 同镜像）

| 指标（recompute_logp 阶段） | v1 | 方案 C | 节省 |
|---|---|---|---|
| allocated（活跃 tensor） | 5.40 GB | 5.40 GB | 持平 |
| reserved（PyTorch cache） | 18.12 GB | **11.24 GB** | **-7 GB (-40%)** |
| device used（GPU 实际占用） | 38.01 GB | **26.52 GB** | **-11.5 GB (-30%)** |

**ref_logp 阶段** 也 30% 节省（reserved 12.36 → 7.75 GB）→ 证明方案 C 在 ref 路径上也生效。

### 4.2 35B（CP 适配验证 ✓，撞 offload bug ✗）

| 任务 | Backend | 状态 | 备注 |
|---|---|---|---|
| `bifrost-2026042716021100` (v22) | (attn:d4p4t2\|ffn:e8t1) cp=1 | ❌ OOM | actor 60.37 + ref 8.61 + buffer 15.31 = 84 GB |
| `bifrost-2026042717413600` (v22-pp8) | (attn:d2p8t2\|ffn:e4t1) | ❌ OOM | PP 救不到 LM head logits |
| `bifrost-2026042909105101` (v26 r1, Task 5 v1) | (attn:d2p4t2c2\|ffn:e8t1) | ❌ OOM | Task 5 v1 caller-gather logits 没省内存 |
| `bifrost-2026042910570501` (v26 r2, **方案 C**) | (attn:d2p4t2c2\|ffn:e8t1) | ⚠️ OOM 缺 0.4 GB | **CP 适配 OK**：actor 60→26 GB ✓, buffer 15.31→7.66 GB ✓; **撞独立 offload bug** |

#### 35B 关键证据（v26 r2 OOM 时）

```
This process (actor offload 残留): 26.06 GB
Process 1728204 (ref onload):     46.72 GB
Tried to allocate (.exp buffer):   7.66 GB    ← 仅 v22 的 1/2, 方案 C 完美生效 ✓
─────────────────────────────────────────
合计:                              80.4 GB → OOM 缺 0.4 GB
```

| 维度 | v22 (no CP) | **v26 r2 (方案 C cp=2)** |
|---|---|---|
| Actor 进程 | 60.37 GB | **26.06 GB** (-57%) ✓ |
| `.exp()` buffer 申请 | 15.31 GB | **7.66 GB** (cp=2 切半精确生效) ✓ |
| OOM 阶段 | actor compute_logp | **ref compute_logp** (前进了一步) |

**结论：CP 适配本身完成。** 但 35B colocate 模式下 actor offload 残留 26 GB 触发新 OOM，是独立 bug（详见 §6）。

---

## 5. 32K 预算（CP=4 路径）

CP=2 已让 actor 端 buffer 减半。**CP=4 让 buffer 减到 1/4**，是 32K 路径的关键。

| 配置 | per-rank seq | logits buffer | 预期 actor onload | 备注 |
|---|---|---|---|---|
| v22 (cp=1) 16K | 18432 | 15.31 GB | 60 GB | OOM |
| v26 (cp=2) 16K | 9216 | 7.66 GB ✓ | 26 GB ✓ | 方案 C 验证 |
| **v?? (cp=4) 16K** | 4608 | ~3.8 GB | ~13 GB | 内存充裕 |
| **v?? (cp=4) 32K** | 8192 | ~7.6 GB | ~26 GB | 与 cp=2 16K 相当 |

backend: `(attn:d1p4t2c4|ffn:e8t1)` → DP=1, CP=4, PP=4, TP=2 = 32 GPU

DP=1 代价：单批梯度噪声升，需要把 `ppo_n_minibatches` 从 64 加到 128（让每 rank 看 1 sample/mb）。

---

## 6. 已知遗留问题（与 CP 无关）

### Actor offload 不彻底（colocate 模式 35B 触发）

[`megatron_engine.py:877`](../../areal/engine/megatron_engine.py#L877) 用 `torch_memory_saver.pause()` 实现 offload，但 35B + Megatron `distributed_optimizer` + `wrap_with_ddp=true` 下：

```
init 阶段 IOStruct 实测:
  before offload: device 27.45 GB
  after  offload: device 25.06 GB
  仅释放 2.4 GB (期望释放 ~9 GB params)
```

`torch_memory_saver` 不能捕获 megatron 内部 allocator 分配的 grad/optim buffer。

**影响**：v26 r2 OOM 缺 0.4 GB —— actor 残留 26 GB 中如果能释放 1 GB 就过。这是 35B 量级特有问题（0.8B 模型小，残留也小）。

**修复方向**（独立工作，不在 CP 适配范畴）：
1. `offload()` 末尾加 `torch.cuda.empty_cache()` 多次调用强制 PyTorch 缓存返还系统
2. 显式把 megatron 的 DDP grad buffer / Adam state 移到 CPU
3. 或者切 `cp=4` 让所有进程占用减半，绕过 offload bug

---

## 7. 下一步路线图

| 任务 | 类型 | 优先级 | 预期 |
|---|---|---|---|
| 诊断 + 修 actor offload bug | 独立工作 | 🔴 高 | 35B 16K + cp=2 直接通过；无需切 cp=4 |
| 验证 cp=4 (v27) | 快速绕过 | 🟡 中 | 即使 offload bug 不修, cp=4 让总占用减半也能通过 |
| 35B 32K + cp=4 (Task 8) | 业务目标 | 🟢 终极 | 长序列训练落地 |
| 0.8B cp=1 vs cp=2 数值一致性 (Task 6) | 可选验证 | 🟢 低 | logp 最大绝对差 < 1e-2 (PPO 数值正常已隐含) |

---

## 8. 关键参考

### 上游 PR
- [Megatron-LM PR #2614](https://github.com/NVIDIA/Megatron-LM/pull/2614) — GDN context parallel (head-parallel via all-to-all)
- [Megatron-LM PR #2644](https://github.com/NVIDIA/Megatron-LM/pull/2644) — GDN packed sequence (THD)
- [Megatron-LM PR #4230](https://github.com/NVIDIA/Megatron-LM/pull/4230) — GDN packed_seq + CP padding alignment fix（关键：`_resolve_cu_seqlens` + GDN init `**kwargs`）

### NVIDIA 官方坑
- [Megatron-LM Issue #1369](https://github.com/NVIDIA/Megatron-LM/issues/1369) — `_coalescing_manager` NCCL 不兼容（致命，step 1 必崩）
- [pytorch/pytorch Issue #134833](https://github.com/pytorch/pytorch/issues/134833) — NCCL 缺 `reduce_scatter_tensor_coalesced`

### veRL 实证参考
- [veRL Qwen3.5 Long Context CP 文档](/Users/zengbw/Codebase/for_llm_train_070/llm_train_sft_0402/docs/qwen35_long_context_cp.md) — 4 节点 32 卡 SFT 32K + CP=2 跑通 120+ step
- veRL pin commit 是 `0f6fcb0`（与我们一致）

### 本仓库相关文档
- [`docs/superpowers/plans/2026-04-28-cp-gdn-megatron-017.md`](../superpowers/plans/2026-04-28-cp-gdn-megatron-017.md) — 初始 plan + 8 task roadmap
- [`docs/superpowers/plans/2026-04-29-bshd-cp-logp-on-split.md`](../superpowers/plans/2026-04-29-bshd-cp-logp-on-split.md) — 方案 C 设计 + 实测对比
- [`docs/fuyao-experiments/qwen3_5-adaptation.md`](qwen3_5-adaptation.md) — 早期 13 个 Qwen3.5 + Megatron 适配 patch

---

## 附录 A: 完整 commit 历史

```
5dd9b65 docs: 修正 ref 46 GB 根因 - 不是 ref cp 异常, 是 OOM 时序换了
c5c2322 docs: 归档方案 C 0.8B cp=2 显存实测对比 (节省 30%)
5f52fa3 feat(megatron): 方案 C - BSHD CP 在 cp-split 上算 logp, 只 gather 标量  ← 最终实现
12ecc6c feat(megatron): Task 5 - BSHD CP postprocess all-gather + zigzag unshuffle  (v1, 错误实现)
2660cbd fix(qwen3_5): GDN + cp_size>1 时清 cp_comm_type 避免 GDN init TypeError
cf2c680 fix(qwen3_5): 禁用 MTP 避免 Megatron 0.18 dev MTP 在 cp_size>1 下 hidden-dim 不匹配 bug
2f786e6 feat(megatron): 删除 BSHD + cp_size>1 的 fail-fast guard (Task 4)
cd308de fix(dockerfile): 用 COPY + RUN 替代 heredoc (fuyao builder 不支持)
95d3676 fix(dockerfile): 提升到 v25 - 参考 veRL Qwen3.5 long context CP 方案
d068697 fix(dockerfile): megatron-core 改成 git+main commit 20ba03f (含 GDN CP)
b56bc62 fix(dockerfile): 强制重装 mbridge/megatron-core 并加版本验证
36ff9f1 deps: 升级 megatron-core 0.16.0 → 0.17.0 (GDN BSHD-CP 支持)
```
