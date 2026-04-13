# AReaL RL 训练吞吐量与 MFU 指标设计
> 日期：2026-04-13 | 状态：已批准 | 作者：zengbw

## 1. 问题

AReaL 缺少吞吐量（tokens/gpu/s）和 MFU（Model FLOPs Utilization）指标。verl v070 已有这两项，导致无法进行公平的吞吐对比。AReaL 已有每阶段的计时（`timeperf/*`）和 token 计数（`ppo_actor/n_tokens`），但没有从中计算出衍生的性能指标。

## 2. 目标

为 AReaL RL 训练添加吞吐量和 MFU 指标，同时在 SwanLab 实时看板和训练日志中可见。指标计算口径必须与 v070 对齐，以支持公平对比。

## 3. 不做的事

- SFT trainer 的指标（只改 RL trainer）
- Rollout 引擎内部的性能分析（SGLang/vLLM 有自己的指标）
- 重构现有的 stats_tracker 或 perf_tracer

## 4. 整体架构

```
训练循环 (rl_trainer.py)
  │
  ├── rollout ──────→ PerfMetrics.record("rollout", tokens, time)
  ├── recompute_logp ──→ PerfMetrics.record("recompute_logp", tokens, time)
  ├── train_step ───→ PerfMetrics.record("train_step", tokens, time)
  ├── update_weights → PerfMetrics.record("update_weights", 0, time)
  │
  └── 步结束 ───────→ PerfMetrics.compute()
                        │
                        ├── FlopsCounter.estimate_flops(seqlens, time)
                        │     └── 按模型架构选择对应公式计算
                        │
                        ├── GPU 峰值算力查表
                        │
                        └── 输出指标 ──→ stats_tracker.scalar()  → SwanLab/WandB
                                      └→ logger.info()           → main.log
```

### 新增文件

| 文件 | 用途 |
|---|---|
| `areal/utils/flops_counter.py` | GPU 峰值算力表 + 按模型架构估算 FLOPs。从 v070 `verl/utils/flops_counter.py` 移植 |
| `areal/utils/perf_metrics.py` | PerfMetrics 类：累积每阶段数据，计算吞吐量和 MFU |

### 修改文件

| 文件 | 改动 |
|---|---|
| `areal/trainer/rl_trainer.py` | 初始化 PerfMetrics；每个阶段调用 `record()`；步结束调用 `compute()` 并上报 |
| `areal/trainer/ppo/actor.py` | `_ppo_update()` 返回 `n_tokens`，供 rl_trainer 传给 PerfMetrics |

## 5. FlopsCounter

从 v070 的 `verl/utils/flops_counter.py`（约 600 行）移植，保留核心能力并适配 AReaL 接口。

### 5.1 GPU 峰值算力表（BF16 TFLOPS）

| GPU | TFLOPS | 匹配模式 |
|---|---|---|
| A100 | 312 | `"A100"` |
| H100 | 989 | `"H100"` |
| H800 | 989 | `"H800"` |
| L40S | 362 | `"L40S"` |

检测方式：`torch.cuda.get_device_name()` → 子串匹配 → 查表。未知 GPU 打 warning，默认 312（A100）。

### 5.2 模型架构 FLOPs 估算

| 架构 | v070 函数 | 公式概要 |
|---|---|---|
| qwen2, qwen3, llama（dense） | `_estimate_qwen2_flops` | `6 * dense_params * tokens + 6 * seqlen_sq_sum * head_dim * n_heads * n_layers` |
| qwen2_moe, qwen3_moe | `_estimate_qwen2_moe_flops` | attention FLOPs + 激活的 expert FLOPs（非全部参数） |
| deepseek_v3 | `_estimate_deepseek_v3_flops` | DeepSeek MoE 特化公式 |

架构检测：读取 HuggingFace `config.json` 中的 `model_type` 字段。

### 5.3 接口

```python
class FlopsCounter:
    def __init__(self, hf_config_path: str, device_name: str | None = None):
        """加载模型配置，检测 GPU，选择 FLOPs 估算函数。"""

    def estimate_flops(
        self, batch_seqlens: list[int], delta_time: float
    ) -> tuple[float, float]:
        """估算一个 batch 的 FLOPs。

        参数:
            batch_seqlens: batch 中每条序列的长度列表。
            delta_time: 墙钟时间（秒）。

        返回:
            (estimated_tflops_per_sec, promised_tflops_per_sec)
        """
```

## 6. PerfMetrics

### 6.1 接口

```python
class PerfMetrics:
    def __init__(self, flops_counter: FlopsCounter, n_gpus: int, n_train_gpus: int):
        """
        参数:
            flops_counter: 用于 MFU 计算。
            n_gpus: 所有节点的 GPU 总数（用于计算整步 per-GPU 吞吐）。
            n_train_gpus: 训练节点 GPU 数（用于 MFU 计算，不含推理节点）。
        """

    def record(self, phase: str, n_tokens: int, elapsed_sec: float,
               seqlens: list[int] | None = None):
        """记录一个阶段的 token 数和耗时。

        参数:
            phase: 阶段名（"rollout", "train_step", "update_weights" 等）。
            n_tokens: 该阶段处理的 token 总数。
            elapsed_sec: 该阶段的墙钟时间。
            seqlens: 每条序列的长度（用于精确计算 attention FLOPs）。
                     为 None 时从 n_tokens / n_seqs 近似。
        """

    def compute(self) -> dict[str, float]:
        """计算当前步的所有指标并重置累积器。

        返回字典，包含:
            perf/throughput          — 整步 tokens/gpu/s
            perf/throughput/train    — 仅 train_step 阶段的 tokens/gpu/s
            perf/throughput/rollout  — 仅 rollout 阶段的 tokens/gpu/s
            perf/mfu                 — MFU（仅训练阶段，不含 rollout）
            perf/time_per_step       — 整步墙钟耗时（秒）
            perf/total_tokens        — 整步处理的 token 总数
        """
```

### 6.2 计算公式

**吞吐量：**
```
perf/throughput         = total_tokens / total_time / n_gpus
perf/throughput/train   = train_tokens / train_time / n_gpus
perf/throughput/rollout = rollout_tokens / rollout_time / n_gpus
```

其中 `total_time = 所有阶段耗时之和`，`total_tokens = 所有阶段 token 之和`。

**MFU：**
```
estimated, promised = flops_counter.estimate_flops(train_seqlens, train_time)
perf/mfu = estimated / promised / n_train_gpus
```

MFU 只统计训练阶段（forward + backward + optimizer）。Rollout 在独立的推理 GPU 上运行（SGLang），不纳入 MFU 计算。`n_train_gpus` = actor GPU 数量（如单训练节点为 8）。

### 6.3 各阶段 Token 来源

| 阶段 | Token 数来源 | 说明 |
|---|---|---|
| rollout | 返回 batch 中的 `sum(seq_lens)` | 生成的 token 总数 |
| recompute_logp | 同一 batch 的 `n_tokens` | 对所有 token 做 forward pass |
| train_step | actor `_ppo_update()` 返回的 `n_tokens` | 参与梯度计算的 token |
| update_weights | 0 | 只有通信，无 token 处理 |
| compute_advantage | 0 | CPU 端计算 |

## 7. rl_trainer.py 集成方式

### 7.1 初始化

```python
from areal.utils.flops_counter import FlopsCounter
from areal.utils.perf_metrics import PerfMetrics

flops_counter = FlopsCounter(
    hf_config_path=config.actor.path,
    device_name=None,  # 自动检测
)
self.perf_metrics = PerfMetrics(
    flops_counter=flops_counter,
    n_gpus=config.cluster.n_nodes * config.cluster.n_gpus_per_node,
    n_train_gpus=actor_gpu_count,  # 从 backend 配置推算
)
```

### 7.2 每阶段记录

在现有的 `with stats_tracker.record_timing("xxx"):` 块之后追加 PerfMetrics 记录：

```python
# 示例：train_step 阶段
with stats_tracker.record_timing("train_step"):
    train_tokens = actor.train_step(batch)  # actor 返回 n_tokens

self.perf_metrics.record(
    "train_step",
    n_tokens=train_tokens,
    elapsed_sec=stats_tracker.last_timing("train_step"),
)
```

注：`stats_tracker.last_timing()` 是需要新增的辅助方法，返回指定 key 最近一次的计时值。如果难以添加，替代方案是独立用 `time.perf_counter()` 计时。

### 7.3 步结束上报

```python
perf = self.perf_metrics.compute()
for k, v in perf.items():
    stats_tracker.scalar(**{k: v})

logger.info(
    f"[Perf] throughput={perf['perf/throughput']:.0f} tok/gpu/s | "
    f"mfu={perf['perf/mfu']:.4f} | "
    f"train={perf['perf/throughput/train']:.0f} tok/gpu/s | "
    f"rollout={perf['perf/throughput/rollout']:.0f} tok/gpu/s"
)
```

## 8. actor.py 改动

当前 `_ppo_update()` 不返回 token 数。需要修改为返回 `n_tokens`，供 rl_trainer 传给 PerfMetrics。

```python
# 修改前:
def _ppo_update(self, data, ...):
    ...  # 内部已计算 n_tokens（第 271 行），但只用于 stats_tracker denominator
    return loss

# 修改后:
def _ppo_update(self, data, ...):
    ...
    return loss, n_tokens  # n_tokens 在第 271 行已经算好
```

所有调用 `_ppo_update()` 的地方需要同步更新返回值的解包方式。

## 9. 输出指标汇总

| 指标 key | 单位 | 可见位置 | 含义 |
|---|---|---|---|
| `perf/throughput` | tokens/gpu/s | SwanLab + 日志 | 端到端每 GPU 吞吐量 |
| `perf/throughput/train` | tokens/gpu/s | SwanLab + 日志 | 训练阶段每 GPU 吞吐量 |
| `perf/throughput/rollout` | tokens/gpu/s | SwanLab + 日志 | 推理阶段每 GPU 吞吐量 |
| `perf/mfu` | 比率 (0-1) | SwanLab + 日志 | 模型浮点运算利用率（仅训练阶段） |
| `perf/time_per_step` | 秒 | SwanLab + 日志 | 每步墙钟耗时 |
| `perf/total_tokens` | 数量 | SwanLab + 日志 | 每步处理的 token 总数 |

## 10. 验收测试

### 10.1 单元测试：FlopsCounter

| 用例 | 输入 | 预期 |
|---|---|---|
| A100 检测 | `device_name="NVIDIA A100-SXM4-80GB"` | `promised=312` |
| H100 检测 | `device_name="NVIDIA H100 80GB HBM3"` | `promised=989` |
| 未知 GPU | `device_name="Unknown"` | 打 warning，`promised=312`（默认） |
| Qwen3-8B FLOPs | 8.2B 参数, 1000 tokens, 1.0s | `estimated > 0`，量级约 50 TFLOPS |
| MoE FLOPs | qwen3_moe 配置, 1000 tokens | `estimated < 相同总参数量的 dense 模型` |

### 10.2 单元测试：PerfMetrics

| 用例 | 输入 | 预期 |
|---|---|---|
| 训练吞吐 | `record("train", 10000, 1.0)`, n_gpus=16 | `throughput/train = 625 tok/gpu/s` |
| 整步吞吐 | `record("rollout", 8000, 2.0)` + `record("train", 8000, 1.0)`, n_gpus=16 | `throughput = 333 tok/gpu/s` |
| MFU 范围 | 任意合理输入 | `0 < mfu < 1` |
| compute 后重置 | 连续两次 `compute()` | 第二次返回零值 |
| 零时间保护 | `record("train", 1000, 0.0)` | 不除零错误，返回 0 |

### 10.3 集成测试：训练日志验证

实际跑 10 步训练，确认：
- SwanLab 出现 `perf/throughput`、`perf/mfu` 曲线
- main.log 包含 `[Perf] throughput=... tok/gpu/s | mfu=...` 行
- 数值在合理范围：A100 上 8B 模型 → 吞吐 200-2000 tok/gpu/s，MFU 0.05-0.40

### 10.4 对比验证：与 v070 口径对齐

相同模型 + 数据 + batch size 在两个框架上运行，对比：
- `perf/throughput` 差异 < 10%（框架效率可能不同，但计算公式应一致）
- `perf/mfu` 差异 < 5%（相同 FLOPs 公式、相同 GPU 峰值）

## 11. 风险与缓解

| 风险 | 影响 | 缓解方案 |
|---|---|---|
| `last_timing()` 在 stats_tracker 中不存在 | 无法获取阶段耗时 | 替代方案：独立用 `time.perf_counter()` 包裹 |
| actor `_ppo_update` 返回值变更导致调用方报错 | 现有代码期望单返回值 | 搜索所有调用方，同步更新解包逻辑 |
| rollout 阶段 token 数不易获取 | rollout 吞吐不准确 | 从 rollout 完成后的 batch 元数据中提取 |
| FLOPs 公式与 v070 不一致 | MFU 对比无效 | 直接移植 v070 代码，保持公式完全一致 |
