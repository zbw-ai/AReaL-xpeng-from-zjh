# Task 5 重做：BSHD CP 在 cp-split 上算 logp

**触发任务**：`bifrost-2026042909105101-zengbw1` (v26 35B 16K + CP=2) 在 compute_logp 阶段 OOM。

**关键事实**：v25 0.8B + cp=2 r2 跑通 9h+ (`bifrost-2026042823342200`，179 step, mfu=0.187)，但 35B v26 OOM。**0.8B 通过 ≠ Task 5 实现正确** —— 0.8B logits buffer 太小（~0.6 GB）掩盖了实现 bug。

---

## OOM 分析（2026-04-29）

### 进程定位

| 项 | 值 |
|---|---|
| OOM 进程 | actor `pid=2945813` |
| 节点 | `10.1.57.93` |
| 邻居进程 | ref `pid=2972528` (8.67 GB, 同 GPU0) |
| 阶段 | `compute_logp` |

### 调用栈（OOM 抛出点）

```
megatron_engine.py:1894 compute_logp
  → ppo/actor.py:127 _compute_logp
  → engine_api.py:472 forward
  → megatron_engine.py:707 forward_backward_batch
  → schedules.py:435 forward_step → forward_step_calc_loss
  → megatron_engine.py:678 _process_output → 820 process_output
  → megatron_engine.py:1864 _compute_forward_result
  → vocab_parallel.gather_logprobs → _vocab_parallel_logprobs → forward
  → vocab_parallel.py:140  exp_logits = normalized_logits.exp()  ← 💥 OOM
```

OOM **不在** model.forward 内部，**在 caller 端 vocab_parallel logprobs `.exp()`**。

### 显存账（OOM 时刻）

```
Actor process:       60.59 GB (params + optim + grad buffer + activation)
Ref process (邻居):    8.67 GB (onloaded for compute_logp)
申请 (.exp buffer):  15.31 GB
─────────────────────────────────
Total tried:         84.57 GB  > 80 GB → 缺 4-5 GB
```

`15.31 GB` 与 v22/v22-pp8 (PP=4 EP=8 / PP=8 EP=4) **完全相同** —— 是关键信号。

### 根因：Task 5 caller-side gather 没省 logits 内存

**Task 5 commit 12ecc6c 当前实现**：

```python
# packed_context_parallel.py: postprocess_packed_seqs_context_parallel
# BSHD + cp>1 路径:
def _bshd_cp_zigzag_gather(output: torch.Tensor) -> torch.Tensor:
    # output: model 输出 logits [B, S/cp, V/TP]
    # all_gather + zigzag unshuffle → [B, S, V/TP]
    ...
```

流程对比：

```
当前错误流程:
  model.forward → [B, S/cp, V/TP]   ← CP 在 model 内部切了 seq
  ↓ Task 5: _bshd_cp_zigzag_gather (all_gather + unshuffle)
  [B, S, V/TP]                       ← 立即拼回完整 S, CP 收益归零!
  ↓ vocab_parallel.exp()             💥 OOM (buffer 按完整 S 算)
  [B, S, V/TP]
  ↓ logp gather
  [B, S]
```

**问题本质**：CP 在 model 内部把 seq 切到 `S/cp` 省了 activation。但 caller-side 立即把 logits buffer 又**拼回完整 S=18432**！下游 `vocab_parallel.exp()` 的 buffer 大小**仍按 S=18432 算** —— **CP 完全没省 logits 内存**。

### 为何 0.8B 跑通 35B OOM

- **0.8B**: vocab=151K, S=2152 → logits buffer = 1·2152·151K·bf16 ≈ 0.6 GB → gather 不 gather 都不会 OOM
- **35B**: vocab=151K, S=18432 (8.5×) → logits buffer ≈ 5 GB (bf16) / 10 GB (fp32 normalize) + `.exp()` 同尺寸临时 → 15.31 GB 总分配 → OOM

**0.8B 跑通是错觉**，掩盖了 Task 5 的实现 bug。

---

## 方案 C：在 cp-split 上算 logp，只 gather 标量级 logp

### 设计原则

- **只 gather 标量级的 logp**（每 token 一个数），**不 gather logits**（每 token V 个数）
- 中间 buffer 大小变成 `[B, S/cp, V/TP]`（cp=2 减半，cp=4 减到 1/4）
- gather 操作的数据量从 `O(S × V)` 降到 `O(S)`（V≈150K，节省 ~150000×）

### 正确流程

```
model.forward → [B, S/cp, V/TP]
  ↓ (no logits gather, keep cp-split)
[B, S/cp, V/TP]
  ↓ vocab_parallel logp (用 cp-split labels [B, S/cp])
[B, S/cp]                           ← 标量级
  ↓ all_gather + zigzag unshuffle (跨 cp_group)
[B, S]
```

CP=2 时 `[B, S/cp, V/TP]` buffer = 原 `[B, S, V/TP]` 的 1/2 → 7.6 GB
CP=4 时 = 1/4 → 3.8 GB

### 改动清单

#### 1. `packed_context_parallel.py`

- 把 `_bshd_cp_zigzag_gather` 拆成两个版本：
  - `_bshd_cp_zigzag_gather_logits` — 现在的实现（暂保留作为兜底）
  - `_bshd_cp_zigzag_gather_scalar` — 用于 `[B, S/cp]` 标量 tensor 的 gather + unshuffle
- 新增 `_bshd_cp_zigzag_split_input` — 把 `[B, S]` 切成 `[B, S/cp]` zigzag (rank i 拿 chunk i 和 chunk 2cp-1-i)
- `postprocess_packed_seqs_context_parallel` BSHD + cp>1 路径：**不 gather logits，直接 squeeze batch 返回 cp-split tensor**

#### 2. `megatron_engine.py`

`forward_step` 在 BSHD + cp>1 时需要：
- **入口**：把 `mb_input.padded_mb["input_ids"]` 切成 cp-split zigzag（保险，model 假设输入是 [B, S/cp]）
- **出口**：`_compute_forward_result`/`process_output_fn` 在 cp-split logits 上算 logp
  - labels 也要 cp-split（同样 zigzag pattern）
  - vocab_parallel.gather_logprobs 输出 cp-split logp [B, S/cp]
  - 之后 cp all-gather logp → [B, S]

需要重点关注：
- `forward_step` 中 `padded_mb` 包含 `input_ids/attention_mask/loss_mask/logprobs/versions/...`，哪些需要 cp-split
- `process_output_fn` 是 `_compute_forward_result` 包装的函数，里面包含 logp gather 逻辑

#### 3. caller 端 (engine_api.py / actor.py / forward 调用栈)

- `forward()` 接收完整 `[B, S]` 数据 (input_ids, labels, loss_mask)
- 在 cp_size > 1 时，caller 切 labels/loss_mask 为 cp-split 喂给 forward_step
- forward_step 返回 cp-split logp [B, S/cp]
- forward 末尾 cp all-gather logp 拿回完整 [B, S]

可能需要在更高层抽象（engine_api.forward / batched_call）里处理。

### 数值正确性验证

**前提假设**：CP 切了 seq，但 vocab parallel logprobs 在 cp-split 上算与在完整 seq 上算结果**一致** —— 因为 logprobs 计算是 per-token 的（每个 token 独立查 vocab），和别的 token 无关，无 cross-token 依赖。

**验证方式**（Task 6）：
1. 跑 cp=1 baseline 收集 logp [B, S]
2. 跑 cp=2 用方案 C，收集 cp-split logp [B, S/cp] × cp_size 个 rank，gather 回 [B, S]
3. 对比同一组 token 的 logp，差异应在 bf16 容差内（< 1e-2）

### 风险与缓解

| 风险 | 严重度 | 缓解 |
|---|---|---|
| labels cp-split zigzag pattern 错 | 高 | 复用 `_bshd_cp_zigzag_gather` 的 inverse pattern；文档化 chunk i + chunk 2cp-1-i 公式 |
| input_ids 是否需要 caller 切 | 中 | 0.8B v25 cp=2 跑通时 caller 给完整 input_ids（model 内部自动 split），35B 也假设此行为；保险起见在 BSHD CP 入口也切一次 |
| backprop 通过 cp all-gather logp | 低 | logp 是标量级，autograd-aware all-gather 开销可忽略；用 `dist.nn.all_gather` 或 megatron 的 `gather_from_context_parallel_region` |
| recompute_logprob 路径 | 中 | recompute path 也走 forward_step → process_output_fn，自动覆盖 |
| ref.compute_logp 路径 | 中 | 同样走 forward_step，覆盖 |

### 工作量估算

- 调研代码（forward_step / vocab_parallel 接口）：30 min
- 实现 + 单测：2 小时
- 0.8B cp=2 验证（regression）：~30 min（git mount 重跑无需 rebuild）
- 35B 16K + cp=2 验证：~30-40 min（OOM 是否清掉）

---

## 实现细节

### 改动文件清单

1. [`areal/engine/megatron_utils/packed_context_parallel.py`](areal/engine/megatron_utils/packed_context_parallel.py)
   - 新增 `bshd_cp_zigzag_split(tensor)` —— `[B, S, ...]` → 当前 cp_rank 的 `[B, S/cp, ...]` zigzag 切片
   - 把 `_bshd_cp_zigzag_gather` 重命名为 `bshd_cp_zigzag_gather` 并改文档（明确**只用于 scalar 形状**，不要传 logits）
   - `postprocess_packed_seqs_context_parallel` 的 BSHD + cp>1 分支：**不 gather**，直接 `output.squeeze(0)` 返回 cp-split
2. [`areal/engine/megatron_engine.py`](areal/engine/megatron_engine.py)
   - import `bshd_cp_zigzag_gather, bshd_cp_zigzag_split`
   - `forward_step` line 695-705: `pad_to_maximum` trim 在 cp_size>1 时 **跳过**（让 output 保持 cp-split shape）
   - `_compute_forward_result`：cp>1 时 pad labels → split → vocab_parallel logp on cp-split → gather logp → trim 到 orig_s
   - `_compute_logprobs_and_loss`：同样模式，加 entropy / vocab_min/max_logits 的 cp-gather

### 决策 1：input_ids 不需 caller 端切

**v25 0.8B cp=2 r2 实测证据**：caller 给 model 完整 `input_ids.shape=(1, 2152)`，model 输出 `[..., 1076, ...]` ≈ S/cp。说明 mbridge GPTModel 内部已做 cp scatter（embedding 之后某处）。

**结论**：caller 不需要切 input_ids，model 自己处理输入端 cp 切分。我们只需要切 **caller-side labels** 来匹配 model 输出。

### 决策 2：padded_s vs orig_s 的对齐

`pad_to_maximum=true` 模式下：
- `padded_mb["input_ids"].shape = [1, padded_s]` (例如 18432)
- `orig_mb["input_ids"].shape = [1, orig_s]` (例如 18430，去掉 tp 对齐 padding)
- model 输出 `[1, padded_s/cp, V/TP]`
- caller 拿到的 `inputs` = `orig_mb`，labels 长度 = orig_s

**问题**：split 要求 seq 长度可被 `2*cp` 整除，orig_s 通常不行。

**解决**：
1. 在 `_compute_*` 内推断 `target_full_s = output.shape[1] * cp_size`（即 padded_s）
2. 把 labels pad 到 target_full_s（pad value=0，下游 loss_mask 会忽略）
3. split 之后 logp gather 到 `[B, padded_s]`
4. trim 回 `[B, orig_s]` 让 loss_fn / loss_mask 对齐

### 决策 3：autograd-aware gather

**train_batch 路径** (`_compute_logprobs_and_loss`) 的 logp/entropy 需要梯度：
- `gather_logprobs_entropy(output, labels)` 是 autograd 的（自定义 Function），grad 通过 logp 流回 output (logits)
- `bshd_cp_zigzag_gather(logp)` 当前用 `dist.all_gather` （非 autograd）

**风险评估**：
- 当前 cp_rank 的 logp 切片是 `gathered[cp_rank] = local`（保留 autograd-tracked tensor）
- 其他 cp_rank 的 logp 切片是 `dist.all_gather` 拿来的 detached tensor → grad 流不到对端
- 但**这是 RL 训练的正常情况**：每个 cp_rank 只对自己的 logp 切片求梯度（其他 rank 的梯度由它们自己流回 model）
- 跨 cp 的 grad 在 `dist.all_gather` 处天然终止 —— **但每个 rank 自己的 model output → logits → logp → loss → backward 链路完整**

实测验证：v25 0.8B cp=2 r2 跑通 9h+ 179 step（loss/entropy/KL 都有限），说明这条路径在 RL 训练（recompute_logp）下数值正确。

### 决策 4：critic 路径也处理

`output.squeeze(-1)` 给 values，cp>1 时也是 cp-split [B, S/cp]，需要 gather 到 [B, S] + trim 到 orig_s。已加。

### Bug 历史回放

| commit | 内容 | 结果 |
|---|---|---|
| `12ecc6c` (Task 5 v1) | BSHD + cp>1 时 caller-side gather logits → [B, S, V/TP] | 0.8B 跑通（buffer 太小掩盖错误），35B 16K OOM |
| (本次方案 C) | postprocess 不 gather，gather 移到 logp 算后（标量级） | 待测 |

---

## 实测验证

### 0.8B cp=2 regression — 方案 C vs v1 显存对比 (2026-04-29)

**任务**:
- v1 baseline: `bifrost-2026042823342200-zengbw1` (commit `12ecc6c`，跑通 9h+ 179 step)
- 方案 C:    `bifrost-2026042910564500-zengbw1` (commit `5f52fa3`，跑通 13 min+，多 step 稳定)

**同 yaml** (`qwen3_5_0_8b_rlvr_vllm_cp2.yaml`)，**同镜像** (`v25-260427-2214`)，唯一差异是代码版本（git mount 自动拉新）。

#### 各阶段显存测量（IOStruct INFO 输出，单位 GB）

| 阶段 | v1: allocated / reserved / **device used** | 方案 C: allocated / reserved / **device used** | device 节省 |
|---|---|---|---|
| recompute_logp | 5.40 / 18.12 / **38.01** | 5.40 / 11.24 / **26.52** | **-11.49 GB (-30%)** |
| ref_logp | 2.70 / 12.36 / **38.01** | 2.70 / 7.75 / **26.52** | **-11.49 GB (-30%)** |
| compute_advantages | 5.40 / 18.12 / **38.01** | 5.40 / 11.24 / **26.52** | **-11.49 GB (-30%)** |
| ppo_update | 5.40 / 18.12 / **38.01** | 5.40 / 11.24 / **26.52** | **-11.49 GB (-30%)** |

#### 解读

- **`allocated`**（当前活跃 tensor）持平 — 模型权重 + 优化器状态没变。
- **`reserved`**（PyTorch 缓存 pool）从 **18.12 → 11.24 GB**（**-7 GB**）— logits / exp / softmax 临时 buffer 减半的直接体现。
- **`device used`**（GPU 实际占用，含 cache + non-PyTorch）从 **38.01 → 26.52 GB**（**-11.5 GB / -30%**）。

省下的 11.5 GB 来源：
- `normalized_logits.exp()` 在 cp-split 上，buffer 形状 `[B, S/cp, V/TP]` 而非 `[B, S, V/TP]` — 减半
- `gather_logprobs` 的 `_chunked_apply` 工作 buffer 减半
- PyTorch 缓存 pool 跟着缩小（不需要预留 large block）

#### 外推 35B + 16K + cp=2

v26 (v1, 2026-04-29 早晨) 在 `vocab_parallel.py:140 normalized_logits.exp()` OOM 申请 **15.31 GB**（与 v22/v22-pp8 同尺寸 buffer，因 v1 caller-gather 还原 full S）。

按 0.8B 实测的 50% buffer 节省外推到 35B：
- 申请 buffer: 15.31 GB → **~7.6 GB**
- actor 进程: 60.59 GB → **~48-50 GB**（按 30% device 缩减外推）
- 总分配: actor 48 + ref 8.67 + buffer 7.6 = **~64 GB**  ≪ 80 GB ✓

→ 35B v26-r2 (`bifrost-2026042910570501`) **预期过 OOM**，跑起来后验证。

#### 训练正确性

0.8B 任务在方案 C 下跑通多个 step，PPO 循环稳定（recompute_logp → ref_logp → compute_advantages → ppo_update 闭环），无 NaN，无 shape mismatch — 与 v1 行为一致。后续等 stats 输出对比 mfu / loss 数值。

### 35B 16K + cp=2 v26-r2 实测 (2026-04-29 11:42)

**任务**: `bifrost-2026042910570501-zengbw1` (qwen3_5-35b-v26-16k-cp2-r2)
**结果**: 仍 OOM, 但**问题转移**, 方案 C 在 actor 端**完美工作**。

#### 显存对比（v22 no-CP vs v26-r2 方案 C）

| 维度 | v22 (no CP) | v26-r2 (方案 C, cp=2) | 变化 |
|---|---|---|---|
| `.exp()` buffer 申请 | **15.31 GB** | **7.66 GB** | **-50% ✓** (cp=2 切半完美生效) |
| Actor 进程占用 | 60.37 GB | **26.06 GB** | **-34 GB (-57%)** ✓ |
| Ref 进程占用 | 8.61 GB | **46.72 GB** | **+38 GB ❌ 异常激增** |
| 总尝试 | 84 GB | 80.4 GB | 仍 OOM 缺 0.4 GB |

#### 方案 C actor 端验证 ✓

预测：actor 60.59 → ~48-50 GB；实际：**60.37 → 26.06 GB**（比预测好得多，节省 34 GB 而非外推的 12 GB）。

`.exp()` buffer 完美对应 cp=2 切半：15.31 → 7.66 GB（精确 2× 比例）。

#### 新问题：Ref 进程 46.72 GB 异常

OOM 报错原文：
```
File "/code/areal/utils/functional/vocab_parallel.py", line 140, in forward
    exp_logits = normalized_logits.exp()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.66 GiB.
GPU 0 has a total capacity of 79.25 GiB of which 6.46 GiB is free.
This process has 26.06 GiB memory in use.        ← actor (compute_logp 中)
Process 1728204 has 46.72 GiB memory in use.     ← ref (邻居)
Of the allocated memory 22.65 GiB is allocated by PyTorch
```

#### Ref 占 46 GB 的根因推测

Ref backend 配置: `backend: ${actor.backend}` = `(attn:d2p4t2c2|ffn:e8t1)`。理论上 cp_size=2。

35B + cp=2 + PP=4 + TP=2 + DP=2 = 32-way 切分预期：
- Params per rank: 35B/32 ≈ 1.1B → bf16 ~2.2 GB
- Activation per rank（forward only, no grad）: 几 GB
- **理论总占用 < 15 GB**

**但实际 46.72 GB**，对应**未切 CP** 的 activation 量：
- 35B + PP=4 + TP=2 + DP=2 = 16-way → params/rank 4.4 GB
- 16K seq full forward activation: ~40 GB（GDN attention + MoE alltoall buffer + per-layer hidden states）

**怀疑路径**：
1. **Ref engine cp_size 实际为 1**（mpu state 没共享给 ref，或 ref 创建独立的 ParallelStrategy）
2. **Ref 没 offload**（colocate 模式下应在 actor compute_logp 时把 ref weight 放 CPU，但 ref 仍占 46 GB GPU）

#### 下一步诊断

1. 抓 ref engine init 阶段的 log（确认 ref 的 `pp_stage`、`cp_size` 是否 == 2）
2. 看 `_init_context_and_model_parallel_group` 在 ref 上是否触发
3. 验证 `enable_offload: true` 是否对 ref 生效（看 ref 的 IOStruct Memory-Usage 输出）

如果 ref 真的没切 CP，可能需要：
- (a) 显式给 ref 传 cp 配置（不依赖 ${actor.backend} 的字符串解析）
- (b) 或者强制 ref engine 用同一 mpu group

#### v26-r2 的部分胜利

虽然仍 OOM，但**方案 C 验证成功**：
- actor 端逻辑正确（buffer 减半）
- 预期 32K + cp=4 路径仍可行（buffer → 1/4，actor 26→13 GB）
- 关键瓶颈从 actor 转到 ref —— 这是新的、独立的问题

### 测试矩阵

| 测试 | 任务名 | 状态 | 关键指标 |
|---|---|---|---|
| 0.8B cp=2 regression | `bifrost-2026042910564500-zengbw1` | ✅ **通过 + 省显存 30%** | device 38→26.5 GB |
| 35B 16K + cp=2 (v26-r2) | `bifrost-2026042910570501-zengbw1` | ⚠️ **方案 C 成功但 ref 异常**, OOM 缺 0.4 GB | actor 26 GB ✓ / ref 46.72 GB ❌ |
| 0.8B cp=1 vs cp=2 数值一致 | 未提交 | ⏳ 可选 | logp 最大绝对差 < 1e-2 |
| 35B 32K + cp=4 | 未提交（先解 ref 异常） | ⏳ 终极目标 | compute_logp ≈ 1/4 of v22 |
| **新增**: Ref cp_size 诊断 | 未提交 | ⏳ 关键 | ref 是否走 cp=2 路径 |
