# Throughput & MFU Metrics for AReaL RL Training
> Date: 2026-04-13 | Status: Approved | Author: zengbw

## 1. Problem

AReaL lacks throughput (tokens/gpu/s) and MFU (Model FLOPs Utilization) metrics. verl v070 has both, making fair throughput comparison impossible. AReaL already tracks per-phase timing (`timeperf/*`) and token counts (`ppo_actor/n_tokens`), but does not compute derived performance metrics from them.

## 2. Goal

Add throughput and MFU metrics to AReaL RL training, visible in both SwanLab real-time dashboard and training logs. Metrics must align with v070's computation methodology for fair comparison.

## 3. Non-Goals

- SFT trainer metrics (only RL trainer in scope)
- Rollout engine internal profiling (SGLang/vLLM has its own metrics)
- Refactoring existing stats_tracker or perf_tracer

## 4. Architecture

```
Training Loop (rl_trainer.py)
  │
  ├── rollout ──────→ PerfMetrics.record("rollout", tokens, time)
  ├── recompute_logp ──→ PerfMetrics.record("recompute_logp", tokens, time)
  ├── train_step ───→ PerfMetrics.record("train_step", tokens, time)
  ├── update_weights → PerfMetrics.record("update_weights", 0, time)
  │
  └── step end ─────→ PerfMetrics.compute()
                        │
                        ├── FlopsCounter.estimate_flops(seqlens, time)
                        │     └── model-arch-specific formula
                        │
                        ├── GPU peak TFLOPS lookup
                        │
                        └── Output metrics ──→ stats_tracker.scalar()  → SwanLab
                                            └→ logger.info()           → main.log
```

### New files

| File | Purpose |
|---|---|
| `areal/utils/flops_counter.py` | GPU peak TFLOPS table + per-architecture FLOPs estimation. Ported from verl v070 `verl/utils/flops_counter.py` |
| `areal/utils/perf_metrics.py` | PerfMetrics class: accumulates per-phase data, computes throughput and MFU |

### Modified files

| File | Change |
|---|---|
| `areal/trainer/rl_trainer.py` | Initialize PerfMetrics; call `record()` after each phase; call `compute()` + log at step end |
| `areal/trainer/ppo/actor.py` | Return `n_tokens` from `_ppo_update()` so rl_trainer can pass it to PerfMetrics |

## 5. FlopsCounter

Ported from `verl/utils/flops_counter.py` (~600 lines). Retains:

### 5.1 GPU Peak TFLOPS Table (BF16)

| GPU | TFLOPS | Match pattern |
|---|---|---|
| A100 | 312 | `"A100"` |
| H100 | 989 | `"H100"` |
| H800 | 989 | `"H800"` |
| L40S | 362 | `"L40S"` |

Detection: `torch.cuda.get_device_name()` → substring match → lookup. Unknown GPU logs warning, defaults to 312 (A100).

### 5.2 Model Architecture FLOPs Estimation

| Architecture | v070 function | Formula summary |
|---|---|---|
| qwen2, qwen3, llama (dense) | `_estimate_qwen2_flops` | `6 * dense_params * tokens + 6 * seqlen_sq_sum * head_dim * n_heads * n_layers` |
| qwen2_moe, qwen3_moe | `_estimate_qwen2_moe_flops` | Attention FLOPs + activated expert FLOPs (not total params) |
| deepseek_v3 | `_estimate_deepseek_v3_flops` | DeepSeek MoE specific |

Architecture detection: read `model_type` from HuggingFace `config.json`.

### 5.3 Interface

```python
class FlopsCounter:
    def __init__(self, hf_config_path: str, device_name: str | None = None):
        """Load model config, detect GPU, select FLOPs estimator."""

    def estimate_flops(
        self, batch_seqlens: list[int], delta_time: float
    ) -> tuple[float, float]:
        """Estimate FLOPs for a batch.

        Args:
            batch_seqlens: List of sequence lengths in the batch.
            delta_time: Wall-clock time in seconds.

        Returns:
            (estimated_tflops_per_sec, promised_tflops_per_sec)
        """
```

## 6. PerfMetrics

### 6.1 Interface

```python
class PerfMetrics:
    def __init__(self, flops_counter: FlopsCounter, n_gpus: int):
        """
        Args:
            flops_counter: For MFU calculation.
            n_gpus: Total GPU count across all nodes (for per-GPU throughput).
        """

    def record(self, phase: str, n_tokens: int, elapsed_sec: float,
               seqlens: list[int] | None = None):
        """Record one phase's token count and timing.

        Args:
            phase: Phase name ("rollout", "train_step", "update_weights", etc.)
            n_tokens: Total tokens processed in this phase.
            elapsed_sec: Wall-clock time for this phase.
            seqlens: Per-sequence lengths (needed for precise FLOPs with attention).
                     If None, approximated from n_tokens / n_seqs.
        """

    def compute(self) -> dict[str, float]:
        """Compute all metrics for the current step and reset accumulators.

        Returns dict with keys:
            perf/throughput          — tokens/gpu/s for entire step
            perf/throughput/train    — tokens/gpu/s for train_step phase only
            perf/throughput/rollout  — tokens/gpu/s for rollout phase only
            perf/mfu                 — MFU (training phases only, excludes rollout)
            perf/time_per_step       — total step wall-clock seconds
            perf/total_tokens        — total tokens in the step
        """
```

### 6.2 Computation Details

**Throughput:**
```
perf/throughput        = total_tokens / total_time / n_gpus
perf/throughput/train  = train_tokens / train_time / n_gpus
perf/throughput/rollout = rollout_tokens / rollout_time / n_gpus
```

Where `total_time = sum of all phase times`, `total_tokens = sum of all phase tokens`.

**MFU:**
```
estimated, promised = flops_counter.estimate_flops(train_seqlens, train_time)
perf/mfu = estimated / promised / n_train_gpus
```

MFU only counts training phases (forward + backward + optimizer). Rollout runs on separate inference GPUs (SGLang), not training GPUs, so is excluded from MFU. `n_train_gpus` = actor GPU count (e.g., 8 for one training node).

### 6.3 Token Count Sources

| Phase | Token count source | Notes |
|---|---|---|
| rollout | `sum(seq_lens)` from returned batch | Total generated tokens |
| recompute_logp | Same batch `n_tokens` | Forward pass through all tokens |
| train_step | `n_tokens` from actor `_ppo_update()` return | Tokens used in gradient computation |
| update_weights | 0 | No token processing, only communication |
| compute_advantage | 0 | CPU-side computation |

## 7. Integration in rl_trainer.py

### 7.1 Initialization (in `__init__` or `_init_engines`)

```python
from areal.utils.flops_counter import FlopsCounter
from areal.utils.perf_metrics import PerfMetrics

flops_counter = FlopsCounter(
    hf_config_path=config.actor.path,
    device_name=None,  # auto-detect
)
self.perf_metrics = PerfMetrics(
    flops_counter=flops_counter,
    n_gpus=config.cluster.n_nodes * config.cluster.n_gpus_per_node,
)
```

### 7.2 Per-Phase Recording

Modify existing `with stats_tracker.record_timing("xxx"):` blocks to also record to PerfMetrics. The timing value is extracted from stats_tracker after the context manager exits.

```python
# Example for train_step phase:
with stats_tracker.record_timing("train_step"):
    train_tokens = actor.train_step(batch)  # actor returns n_tokens

self.perf_metrics.record(
    "train_step",
    n_tokens=train_tokens,
    elapsed_sec=stats_tracker.last_timing("train_step"),
)
```

Note: `stats_tracker.last_timing()` is a new helper method that returns the last recorded timing for a key. If this is difficult to add, alternative is to wrap with `time.perf_counter()` independently.

### 7.3 Step End Logging

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

## 8. actor.py Change

Current `_ppo_update()` does not return token count. Need to return it so rl_trainer can pass to PerfMetrics.

```python
# Current:
def _ppo_update(self, data, ...):
    ...  # computes n_tokens internally for stats_tracker denominator
    return loss

# After:
def _ppo_update(self, data, ...):
    ...
    return loss, n_tokens  # n_tokens already computed at line 271
```

Callers of `_ppo_update()` need corresponding update to unpack the return value.

## 9. Output Metrics Summary

| Metric key | Unit | Where visible | Description |
|---|---|---|---|
| `perf/throughput` | tokens/gpu/s | SwanLab + log | End-to-end throughput per GPU |
| `perf/throughput/train` | tokens/gpu/s | SwanLab + log | Training phase throughput per GPU |
| `perf/throughput/rollout` | tokens/gpu/s | SwanLab + log | Rollout phase throughput per GPU |
| `perf/mfu` | ratio (0-1) | SwanLab + log | Model FLOPs Utilization (training phases only, excludes rollout) |
| `perf/time_per_step` | seconds | SwanLab + log | Wall-clock time per step |
| `perf/total_tokens` | count | SwanLab + log | Total tokens processed per step |

## 10. Verification Tests

### 10.1 Unit Test: FlopsCounter

| Case | Input | Expected |
|---|---|---|
| A100 detect | `device_name="NVIDIA A100-SXM4-80GB"` | `promised=312` |
| H100 detect | `device_name="NVIDIA H100 80GB HBM3"` | `promised=989` |
| Unknown GPU | `device_name="Unknown"` | Warning logged, `promised=312` (default) |
| Qwen3-8B FLOPs | 8.2B params, 1000 tokens, 1.0s | `estimated > 0`, order ~50 TFLOPS |
| MoE FLOPs | qwen3_moe config, 1000 tokens | `estimated < dense equivalent` |

### 10.2 Unit Test: PerfMetrics

| Case | Input | Expected |
|---|---|---|
| Train throughput | `record("train", 10000, 1.0)`, n_gpus=16 | `throughput/train = 625 tok/gpu/s` |
| Overall throughput | `record("rollout", 8000, 2.0)` + `record("train", 8000, 1.0)`, n_gpus=16 | `throughput = 333 tok/gpu/s` |
| MFU range | Any valid input | `0 < mfu < 1` |
| Reset after compute | Two consecutive `compute()` | Second returns zeros |
| Zero time guard | `record("train", 1000, 0.0)` | No division by zero, returns 0 |

### 10.3 Integration Test: Training Log Verification

Run actual training for 10 steps and verify:
- SwanLab shows `perf/throughput`, `perf/mfu` curves
- main.log contains `[Perf] throughput=... tok/gpu/s | mfu=...` lines
- Values in reasonable range: A100 8B model → throughput 200-2000 tok/gpu/s, MFU 0.05-0.40

### 10.4 Comparison Test: v070 Alignment

Same model + data + batch size on both frameworks. Compare:
- `perf/throughput` difference < 10% (framework efficiency may differ, but computation formula should agree)
- `perf/mfu` difference < 5% (same FLOPs formula, same GPU peak)

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `last_timing()` not available in stats_tracker | Cannot get phase time without code change | Alternative: independent `time.perf_counter()` wrapping |
| actor `_ppo_update` return value change breaks callers | Existing code expects single return | Search all callers, update unpacking |
| Rollout token count not easily accessible | rollout throughput inaccurate | Extract from batch metadata after rollout completes |
| FLOPs formula mismatch with v070 | MFU comparison invalid | Direct port of v070 code, same formulas verbatim |
