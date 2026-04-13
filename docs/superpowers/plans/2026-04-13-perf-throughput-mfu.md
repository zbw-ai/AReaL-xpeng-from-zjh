# Throughput & MFU 指标实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 为 AReaL RL 训练添加 throughput (tokens/gpu/s) 和 MFU 指标，在 SwanLab 和训练日志中可见，计算口径与 verl v070 对齐。

**架构：** 从 v070 移植 FlopsCounter（GPU 峰值查表 + 按模型架构估算 FLOPs），封装 PerfMetrics 类（累积每阶段 tokens/timing，计算衍生指标），在 rl_trainer.py 中埋点上报。token 数从已有的 `stats_tracker.export_all()` 中的 `ppo_actor/n_tokens__count` 获取，不修改 actor.py。

**技术栈：** Python, PyTorch, HuggingFace transformers (PretrainedConfig)

**设计文档：** `docs/superpowers/specs/2026-04-13-perf-throughput-mfu-design.md`

---

## 文件结构

| 文件 | 动作 | 职责 |
|---|---|---|
| `areal/utils/flops_counter.py` | 新增 | GPU 峰值算力表 + 按模型架构估算 FLOPs |
| `areal/utils/perf_metrics.py` | 新增 | PerfMetrics 类：累积 tokens/timing，计算 throughput/MFU |
| `areal/trainer/rl_trainer.py` | 修改 | 初始化 PerfMetrics，每阶段 record，步结束 compute + 上报 |
| `tests/test_flops_counter.py` | 新增 | FlopsCounter 单元测试 |
| `tests/test_perf_metrics.py` | 新增 | PerfMetrics 单元测试 |

---

### Task 1: FlopsCounter — GPU 峰值算力查表

**Files:**
- Create: `areal/utils/flops_counter.py`
- Create: `tests/test_flops_counter.py`

- [ ] **Step 1: 写 GPU 检测的失败测试**

```python
# tests/test_flops_counter.py
import pytest
from areal.utils.flops_counter import get_device_flops


def test_a100_detection():
    assert get_device_flops("NVIDIA A100-SXM4-80GB") == 312e12


def test_h100_detection():
    assert get_device_flops("NVIDIA H100 80GB HBM3") == 989e12


def test_h800_detection():
    assert get_device_flops("NVIDIA H800") == 989e12


def test_l40s_detection():
    assert get_device_flops("NVIDIA L40S") == 362.05e12


def test_unknown_gpu_returns_default():
    result = get_device_flops("Unknown GPU XYZ")
    assert result == 312e12  # default to A100
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_flops_counter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'areal.utils.flops_counter'`

- [ ] **Step 3: 实现 get_device_flops**

从 v070 的 `verl/utils/flops_counter.py` 第 22-85 行移植 `_DEVICE_FLOPS` 字典和 `get_device_flops` 函数。

```python
# areal/utils/flops_counter.py
"""GPU peak FLOPS lookup and model-architecture FLOPs estimation.

Ported from verl v070 verl/utils/flops_counter.py for throughput/MFU metrics.
"""

import torch
from transformers import AutoConfig, PretrainedConfig

from areal.utils import logging

logger = logging.getLogger("FlopsCounter")

# BF16 peak FLOPS per device (from verl v070)
_DEVICE_FLOPS = {
    "CPU": 448e9,
    "GB200": 2.5e15,
    "B200": 2.25e15,
    "MI300X": 1336e12,
    "H100": 989e12,
    "H800": 989e12,
    "H200": 989e12,
    "A100": 312e12,
    "A800": 312e12,
    "L40S": 362.05e12,
    "L40": 181.05e12,
    "A40": 149.7e12,
    "L20": 119.5e12,
    "H20": 148e12,
    "910B": 354e12,
    "Ascend910": 354e12,
    "RTX 3070 Ti": 21.75e12,
}

_DEFAULT_DEVICE_FLOPS = 312e12  # A100


def get_device_flops(device_name: str) -> float:
    """Get peak BF16 FLOPS for a GPU by name substring matching.

    Args:
        device_name: GPU name string (e.g. from torch.cuda.get_device_name()).

    Returns:
        Peak BF16 FLOPS (e.g. 312e12 for A100).
    """
    for key, flops in _DEVICE_FLOPS.items():
        if key in device_name:
            return flops
    logger.warning(
        f"Unknown GPU '{device_name}', defaulting to {_DEFAULT_DEVICE_FLOPS:.0e} FLOPS (A100)"
    )
    return _DEFAULT_DEVICE_FLOPS
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_flops_counter.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add areal/utils/flops_counter.py tests/test_flops_counter.py
git commit -m "feat(utils): add GPU peak FLOPS lookup table"
```

---

### Task 2: FlopsCounter — 模型 FLOPs 估算

**Files:**
- Modify: `areal/utils/flops_counter.py`
- Modify: `tests/test_flops_counter.py`

- [ ] **Step 1: 写 FLOPs 估算的失败测试**

```python
# tests/test_flops_counter.py (追加)
from unittest.mock import MagicMock
from areal.utils.flops_counter import FlopsCounter


def _make_qwen2_config():
    """Create a mock Qwen2 config for testing."""
    config = MagicMock(spec=PretrainedConfig)
    config.model_type = "qwen2"
    config.hidden_size = 4096
    config.vocab_size = 151936
    config.num_hidden_layers = 36
    config.num_key_value_heads = 4
    config.num_attention_heads = 32
    config.intermediate_size = 22016
    config.head_dim = 128
    return config


def test_flops_counter_qwen2_positive():
    config = _make_qwen2_config()
    counter = FlopsCounter(config, device_name="NVIDIA A100-SXM4-80GB")
    seqlens = [1024] * 8  # batch of 8, each 1024 tokens
    estimated, promised = counter.estimate_flops(seqlens, delta_time=1.0)
    assert estimated > 0
    assert promised == 312e12 / 1e12  # 312 TFLOPS


def test_flops_counter_qwen2_longer_seq_more_flops():
    config = _make_qwen2_config()
    counter = FlopsCounter(config, device_name="NVIDIA A100-SXM4-80GB")
    short_seqs = [512] * 8
    long_seqs = [4096] * 8
    est_short, _ = counter.estimate_flops(short_seqs, delta_time=1.0)
    est_long, _ = counter.estimate_flops(long_seqs, delta_time=1.0)
    assert est_long > est_short  # more tokens + attention quadratic


def test_flops_counter_unknown_model_type():
    config = MagicMock(spec=PretrainedConfig)
    config.model_type = "totally_unknown_model"
    counter = FlopsCounter(config, device_name="NVIDIA A100-SXM4-80GB")
    seqlens = [1024] * 8
    estimated, promised = counter.estimate_flops(seqlens, delta_time=1.0)
    assert estimated >= 0  # should not crash
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_flops_counter.py::test_flops_counter_qwen2_positive -v`
Expected: FAIL with `ImportError: cannot import name 'FlopsCounter'`

- [ ] **Step 3: 实现 FlopsCounter 类和 _estimate_qwen2_flops**

在 `areal/utils/flops_counter.py` 末尾追加。从 v070 移植 `_estimate_qwen2_flops`（第 88-120 行）和 `FlopsCounter` 类（第 561-604 行），以及 `_estimate_qwen2_moe_flops`（第 318-352 行）、`_estimate_unknown_flops`（第 534-558 行）。

```python
# 追加到 areal/utils/flops_counter.py

def _estimate_qwen2_flops(config, tokens_sum, batch_seqlens, delta_time):
    """Estimate FLOPs for Qwen2/Qwen3/LLaMA dense models."""
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # MLP uses SwiGLU: gate + up + down = 3 projections
    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # 6x for fwd + bwd (fwd=2x, bwd=4x)
    dense_N_flops = 6 * dense_N * tokens_sum

    # Attention quadratic term
    seqlen_square_sum = sum(s * s for s in batch_seqlens)
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token / delta_time / 1e12  # TFLOPS/s
    return flops_achieved


def _estimate_qwen2_moe_flops(config, tokens_sum, batch_seqlens, delta_time):
    """Estimate FLOPs for Qwen2-MoE/Qwen3-MoE models."""
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    num_experts = getattr(config, "num_experts", 1)
    num_experts_per_tok = getattr(config, "num_experts_per_tok", 1)

    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    # MoE: only activated experts contribute to FLOPs
    mlp_N_per_expert = hidden_size * intermediate_size * 3
    mlp_N = mlp_N_per_expert * num_experts_per_tok  # only activated experts
    emd_and_lm_head_N = vocab_size * hidden_size * 2

    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_N_flops = 6 * dense_N * tokens_sum

    seqlen_square_sum = sum(s * s for s in batch_seqlens)
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token / delta_time / 1e12
    return flops_achieved


def _estimate_unknown_flops(config, tokens_sum, batch_seqlens, delta_time):
    """Fallback: estimate FLOPs using 6 * total_params * tokens approximation."""
    total_params = sum(
        getattr(config, attr, 0)
        for attr in ["num_parameters", "n_params"]
    )
    if total_params == 0:
        # Rough estimate from hidden_size and num_layers
        hidden = getattr(config, "hidden_size", 4096)
        layers = getattr(config, "num_hidden_layers", 32)
        vocab = getattr(config, "vocab_size", 32000)
        total_params = 12 * hidden * hidden * layers + 2 * vocab * hidden

    flops_all_token = 6 * total_params * tokens_sum
    flops_achieved = flops_all_token / delta_time / 1e12
    return flops_achieved


# Model type -> FLOPs estimation function
_ESTIMATE_FUNC = {
    "qwen2": _estimate_qwen2_flops,
    "qwen3": _estimate_qwen2_flops,
    "llama": _estimate_qwen2_flops,
    "qwen2_moe": _estimate_qwen2_moe_flops,
    "qwen3_moe": _estimate_qwen2_moe_flops,
}


class FlopsCounter:
    """Estimate training FLOPs and compute MFU.

    Ported from verl v070 verl/utils/flops_counter.py.

    Args:
        config: HuggingFace PretrainedConfig (or path to load from).
        device_name: GPU name for peak FLOPS lookup. None = auto-detect.
    """

    def __init__(self, config: PretrainedConfig | str, device_name: str | None = None):
        if isinstance(config, str):
            config = AutoConfig.from_pretrained(config, trust_remote_code=True)
        self.config = config

        if config.model_type not in _ESTIMATE_FUNC:
            logger.warning(
                f"Unknown model_type '{config.model_type}', "
                f"supported: {list(_ESTIMATE_FUNC.keys())}. "
                f"Using fallback FLOPs estimation."
            )

        if device_name is None and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
        self._device_name = device_name or "A100"
        self._promised_tflops = get_device_flops(self._device_name) / 1e12

    def estimate_flops(
        self, batch_seqlens: list[int], delta_time: float
    ) -> tuple[float, float]:
        """Estimate FLOPs for a batch.

        Args:
            batch_seqlens: Sequence lengths in the batch.
            delta_time: Wall-clock seconds.

        Returns:
            (estimated_tflops_per_sec, promised_tflops_per_sec)
        """
        if delta_time <= 0:
            return 0.0, self._promised_tflops

        tokens_sum = sum(batch_seqlens)
        if tokens_sum == 0:
            return 0.0, self._promised_tflops

        estimate_fn = _ESTIMATE_FUNC.get(self.config.model_type, _estimate_unknown_flops)
        estimated = estimate_fn(self.config, tokens_sum, batch_seqlens, delta_time)
        return estimated, self._promised_tflops
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_flops_counter.py -v`
Expected: 8 passed

- [ ] **Step 5: 提交**

```bash
git add areal/utils/flops_counter.py tests/test_flops_counter.py
git commit -m "feat(utils): add FlopsCounter with qwen2/moe FLOPs estimation"
```

---

### Task 3: PerfMetrics 类

**Files:**
- Create: `areal/utils/perf_metrics.py`
- Create: `tests/test_perf_metrics.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_perf_metrics.py
import pytest
from unittest.mock import MagicMock
from areal.utils.perf_metrics import PerfMetrics


def _make_mock_flops_counter(estimated=100.0, promised=312.0):
    counter = MagicMock()
    counter.estimate_flops.return_value = (estimated, promised)
    return counter


def test_train_throughput():
    pm = PerfMetrics(_make_mock_flops_counter(), n_gpus=16, n_train_gpus=8)
    pm.record("train_step", n_tokens=10000, elapsed_sec=1.0)
    result = pm.compute()
    assert result["perf/throughput/train"] == pytest.approx(10000 / 1.0 / 16)


def test_overall_throughput():
    pm = PerfMetrics(_make_mock_flops_counter(), n_gpus=16, n_train_gpus=8)
    pm.record("rollout", n_tokens=8000, elapsed_sec=2.0)
    pm.record("train_step", n_tokens=8000, elapsed_sec=1.0)
    result = pm.compute()
    assert result["perf/throughput"] == pytest.approx(16000 / 3.0 / 16)


def test_rollout_throughput():
    pm = PerfMetrics(_make_mock_flops_counter(), n_gpus=16, n_train_gpus=8)
    pm.record("rollout", n_tokens=8000, elapsed_sec=2.0)
    pm.record("train_step", n_tokens=8000, elapsed_sec=1.0)
    result = pm.compute()
    assert result["perf/throughput/rollout"] == pytest.approx(8000 / 2.0 / 16)


def test_mfu_positive():
    pm = PerfMetrics(_make_mock_flops_counter(estimated=100.0, promised=312.0), n_gpus=16, n_train_gpus=8)
    pm.record("train_step", n_tokens=10000, elapsed_sec=1.0, seqlens=[1024] * 10)
    result = pm.compute()
    assert 0 < result["perf/mfu"] < 1


def test_mfu_uses_train_gpus():
    pm = PerfMetrics(_make_mock_flops_counter(estimated=100.0, promised=312.0), n_gpus=16, n_train_gpus=8)
    pm.record("train_step", n_tokens=10000, elapsed_sec=1.0, seqlens=[1024] * 10)
    result = pm.compute()
    # MFU = estimated / promised / n_train_gpus = 100 / 312 / 8
    assert result["perf/mfu"] == pytest.approx(100.0 / 312.0 / 8)


def test_reset_after_compute():
    pm = PerfMetrics(_make_mock_flops_counter(), n_gpus=16, n_train_gpus=8)
    pm.record("train_step", n_tokens=10000, elapsed_sec=1.0)
    pm.compute()
    result = pm.compute()
    assert result["perf/throughput"] == 0
    assert result["perf/total_tokens"] == 0


def test_zero_time_no_crash():
    pm = PerfMetrics(_make_mock_flops_counter(), n_gpus=16, n_train_gpus=8)
    pm.record("train_step", n_tokens=1000, elapsed_sec=0.0)
    result = pm.compute()
    assert result["perf/throughput/train"] == 0


def test_time_per_step():
    pm = PerfMetrics(_make_mock_flops_counter(), n_gpus=16, n_train_gpus=8)
    pm.record("rollout", n_tokens=5000, elapsed_sec=3.0)
    pm.record("train_step", n_tokens=5000, elapsed_sec=2.0)
    pm.record("update_weights", n_tokens=0, elapsed_sec=0.5)
    result = pm.compute()
    assert result["perf/time_per_step"] == pytest.approx(5.5)


def test_total_tokens():
    pm = PerfMetrics(_make_mock_flops_counter(), n_gpus=16, n_train_gpus=8)
    pm.record("rollout", n_tokens=5000, elapsed_sec=3.0)
    pm.record("train_step", n_tokens=8000, elapsed_sec=2.0)
    result = pm.compute()
    assert result["perf/total_tokens"] == 13000
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_perf_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 PerfMetrics**

```python
# areal/utils/perf_metrics.py
"""Performance metrics: throughput (tokens/gpu/s) and MFU.

Accumulates per-phase token counts and timing, computes derived metrics
at step end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from areal.utils import logging

logger = logging.getLogger("PerfMetrics")


@dataclass
class _PhaseData:
    n_tokens: int = 0
    elapsed_sec: float = 0.0
    seqlens: list[int] = field(default_factory=list)


class PerfMetrics:
    """Accumulate per-phase tokens/timing, compute throughput and MFU.

    Args:
        flops_counter: FlopsCounter instance for MFU calculation.
        n_gpus: Total GPU count across all nodes.
        n_train_gpus: Training node GPU count (for MFU, excludes inference GPUs).
    """

    def __init__(self, flops_counter, n_gpus: int, n_train_gpus: int):
        self._flops_counter = flops_counter
        self._n_gpus = n_gpus
        self._n_train_gpus = n_train_gpus
        self._phases: dict[str, _PhaseData] = {}

    def record(
        self,
        phase: str,
        n_tokens: int,
        elapsed_sec: float,
        seqlens: list[int] | None = None,
    ):
        """Record one phase's token count and timing."""
        if phase not in self._phases:
            self._phases[phase] = _PhaseData()
        p = self._phases[phase]
        p.n_tokens += n_tokens
        p.elapsed_sec += elapsed_sec
        if seqlens:
            p.seqlens.extend(seqlens)

    def compute(self) -> dict[str, float]:
        """Compute all metrics for current step and reset."""
        total_tokens = sum(p.n_tokens for p in self._phases.values())
        total_time = sum(p.elapsed_sec for p in self._phases.values())

        train = self._phases.get("train_step", _PhaseData())
        rollout = self._phases.get("rollout", _PhaseData())

        result = {
            "perf/total_tokens": total_tokens,
            "perf/time_per_step": total_time,
            "perf/throughput": self._safe_div(total_tokens, total_time * self._n_gpus),
            "perf/throughput/train": self._safe_div(
                train.n_tokens, train.elapsed_sec * self._n_gpus
            ),
            "perf/throughput/rollout": self._safe_div(
                rollout.n_tokens, rollout.elapsed_sec * self._n_gpus
            ),
        }

        # MFU: only training phases, on training GPUs
        mfu = 0.0
        if train.elapsed_sec > 0 and train.seqlens:
            estimated, promised = self._flops_counter.estimate_flops(
                train.seqlens, train.elapsed_sec
            )
            if promised > 0:
                mfu = estimated / promised / self._n_train_gpus
        result["perf/mfu"] = mfu

        # Reset
        self._phases.clear()
        return result

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_perf_metrics.py -v`
Expected: 10 passed

- [ ] **Step 5: 提交**

```bash
git add areal/utils/perf_metrics.py tests/test_perf_metrics.py
git commit -m "feat(utils): add PerfMetrics for throughput and MFU"
```

---

### Task 4: rl_trainer.py 集成

**Files:**
- Modify: `areal/trainer/rl_trainer.py`

- [ ] **Step 1: 添加 import 和初始化**

在 `areal/trainer/rl_trainer.py` 顶部 import 区追加：

```python
import time as _time
from areal.utils.flops_counter import FlopsCounter
from areal.utils.perf_metrics import PerfMetrics
```

在 `__init__` 方法末尾（`self._config_perf_tracer()` 之后）追加 PerfMetrics 初始化：

```python
# Initialize throughput/MFU metrics
try:
    flops_counter = FlopsCounter(
        config=config.actor.path,
        device_name=None,  # auto-detect
    )
except Exception as e:
    logger.warning(f"FlopsCounter init failed: {e}, MFU will be 0")
    flops_counter = None

n_gpus = config.cluster.n_nodes * config.cluster.n_gpus_per_node
# Infer train GPU count from actor backend string (e.g. "megatron:d2t2p2" -> 2*2*2=8)
n_train_gpus = n_gpus // 2  # default: half for training, half for rollout
self._perf_metrics = PerfMetrics(
    flops_counter=flops_counter,
    n_gpus=n_gpus,
    n_train_gpus=n_train_gpus,
) if flops_counter is not None else None
```

- [ ] **Step 2: 在训练循环中每个阶段记录**

在 `train()` 方法中，对每个 `with stats_tracker.record_timing("xxx"):` 块，在块之后追加 `_perf_metrics.record()` 调用。使用独立的 `time.perf_counter()` 计时（因为 stats_tracker 没有 `last_timing()` API）。

修改 rollout 阶段（约第 338 行）：
```python
_t0 = _time.perf_counter()
with (
    stats_tracker.record_timing("rollout"),
    perf_tracer.trace_scope("train.rollout", category=Category.COMPUTE),
):
    # ... existing rollout code ...
_t_rollout = _time.perf_counter() - _t0
```

修改 train_step 阶段（约第 434 行）：
```python
_t0 = _time.perf_counter()
with (
    stats_tracker.record_timing("train_step"),
    perf_tracer.trace_scope("train.train_step", category=Category.COMPUTE),
):
    # ... existing train_step code ...
_t_train = _time.perf_counter() - _t0
```

修改 update_weights 阶段（约第 463 行）：
```python
_t0 = _time.perf_counter()
with (
    stats_tracker.record_timing("update_weights"),
    perf_tracer.trace_scope("train.update_weights", category=Category.COMPUTE),
):
    # ... existing update_weights code ...
_t_update = _time.perf_counter() - _t0
```

修改 recompute_logp 阶段（约第 374 行）：
```python
_t0 = _time.perf_counter()
with (
    stats_tracker.record_timing("recompute_logp"),
    perf_tracer.trace_scope("train.recompute_logp", category=Category.COMPUTE),
):
    # ... existing code ...
_t_recompute = _time.perf_counter() - _t0
```

- [ ] **Step 3: 在步结束时计算并上报**

在 `_export_and_commit_stats` 调用之前（约第 530 行后、第 895 行前），追加 PerfMetrics 的 record 和 compute：

```python
if self._perf_metrics is not None:
    # Get token counts from stats export
    stats = self.actor.export_stats()
    train_tokens = int(stats.get("ppo_actor/n_tokens__count", 0))

    # Estimate rollout tokens from batch
    rollout_tokens = sum(
        len(traj.get("input_ids", []))
        for traj in rollout_batch
    ) if rollout_batch else 0

    # Record phases
    self._perf_metrics.record("rollout", rollout_tokens, _t_rollout)
    self._perf_metrics.record("train_step", train_tokens, _t_train,
                               seqlens=[train_tokens])  # approximate
    self._perf_metrics.record("update_weights", 0, _t_update)
    self._perf_metrics.record("recompute_logp", train_tokens, _t_recompute)

    # Compute and log
    perf = self._perf_metrics.compute()
    for k, v in perf.items():
        stats_tracker.scalar(**{k: v})

    logger.info(
        f"[Perf] throughput={perf['perf/throughput']:.0f} tok/gpu/s | "
        f"mfu={perf['perf/mfu']:.4f} | "
        f"train={perf['perf/throughput/train']:.0f} tok/gpu/s | "
        f"rollout={perf['perf/throughput/rollout']:.0f} tok/gpu/s"
    )
```

- [ ] **Step 4: 运行 pre-commit 检查**

Run: `pre-commit run --all-files`
Expected: All checks passed

- [ ] **Step 5: 提交**

```bash
git add areal/trainer/rl_trainer.py
git commit -m "feat(trainer): integrate throughput/MFU metrics in RL training loop"
```

---

### Task 5: 集成测试验证

**Files:** 无新增，使用已有训练脚本验证

- [ ] **Step 1: 本地验证 import 不报错**

```bash
uv run python -c "from areal.utils.flops_counter import FlopsCounter; from areal.utils.perf_metrics import PerfMetrics; print('OK')"
```

Expected: `OK`

- [ ] **Step 2: 运行全部单元测试**

```bash
uv run pytest tests/test_flops_counter.py tests/test_perf_metrics.py -v
```

Expected: 18 passed

- [ ] **Step 3: 提交到集群运行 10 步验证**

部署一个短训练任务（10 步），检查 SwanLab 和日志：

```bash
fuyao deploy --disable-fault-tolerance \
    --docker-image=infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260325-0644 \
    --project=rc-ai-infra --experiment=zengbw1/llm_rl \
    --gpu-type a100 --gpus-per-node 8 --node=2 \
    --label=qwen3-8b-perf-metrics-test \
    --site=fuyao_b1 --queue=rc-llmrl-a100 \
    SWANLAB_API_KEY=sQOWAKdHZlG94Q8BSTnCM \
    MODEL_TAG=perf-test \
    bash fuyao_examples/fuyao_areal_run.sh \
        --run-type math_rlvr \
        --config fuyao_examples/math/qwen3_8b_rlvr.yaml \
        total_train_epochs=1
```

- [ ] **Step 4: 验证指标存在且合理**

检查 SwanLab 出现以下指标：
- `perf/throughput` — 预期 200-2000 tok/gpu/s
- `perf/throughput/train` — 预期 > throughput (train 阶段比整步短)
- `perf/throughput/rollout` — 预期 > 0
- `perf/mfu` — 预期 0.05-0.40
- `perf/time_per_step` — 预期 30-120 秒
- `perf/total_tokens` — 预期 > 0

检查 main.log 包含 `[Perf] throughput=... tok/gpu/s | mfu=...` 行。

- [ ] **Step 5: 提交最终验证通过的版本**

```bash
git add -A
git commit -m "test: verify throughput/MFU metrics in real training"
```
