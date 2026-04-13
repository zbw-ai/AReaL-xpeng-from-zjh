"""Performance metrics accumulator for per-step throughput and MFU tracking.

Accumulates per-phase token counts and timing, then computes throughput and
Model FLOPS Utilization (MFU) metrics at step end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from areal.utils import logging

logger = logging.getLogger("PerfMetrics")


@dataclass
class _PhaseData:
    """Accumulated data for a single named phase within one training step."""

    n_tokens: int = 0
    elapsed_sec: float = 0.0
    seqlens: list[int] = field(default_factory=list)


class PerfMetrics:
    """Accumulates per-phase token counts and timing for one training step.

    At the end of each step, call :meth:`compute` to obtain a flat dict of
    throughput and MFU metrics.  The accumulators are reset automatically after
    each :meth:`compute` call so the instance can be reused across steps.

    Args:
        flops_counter: :class:`~areal.utils.flops_counter.FlopsCounter` instance
            that implements ``estimate_flops(seqlens, time) -> (float, float)``.
            May be ``None``; MFU will be reported as ``0.0`` in that case.
        n_gpus: Total GPU count across all nodes.  Used as the denominator for
            per-GPU throughput metrics.
        n_train_gpus: GPU count on training nodes only.  Used as the denominator
            for MFU calculation.
    """

    def __init__(
        self,
        flops_counter,
        n_gpus: int,
        n_train_gpus: int,
    ) -> None:
        self._flops_counter = flops_counter
        self._n_gpus = n_gpus
        self._n_train_gpus = n_train_gpus
        self._phases: dict[str, _PhaseData] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        phase: str,
        n_tokens: int,
        elapsed_sec: float,
        seqlens: list[int] | None = None,
    ) -> None:
        """Record one phase's token count and timing.

        Multiple calls with the same *phase* name within a single step are
        accumulated (tokens and time are summed; seqlens are extended).

        Args:
            phase: Arbitrary phase identifier (e.g. ``"train_step"``,
                ``"rollout"``).
            n_tokens: Number of tokens processed in this phase invocation.
            elapsed_sec: Wall-clock time (seconds) spent in this phase.
            seqlens: Per-sample sequence lengths.  Required for MFU
                calculation when *phase* is ``"train_step"``; ignored
                otherwise.
        """
        if phase not in self._phases:
            self._phases[phase] = _PhaseData()
        data = self._phases[phase]
        data.n_tokens += n_tokens
        data.elapsed_sec += elapsed_sec
        if seqlens:
            data.seqlens.extend(seqlens)

    def compute(self) -> dict[str, float]:
        """Compute all metrics for the current step, reset accumulators, and return.

        Returns:
            A dict with the following keys:

            * ``perf/throughput``         — total_tokens / total_time / n_gpus
            * ``perf/throughput/train``   — train_tokens / train_time / n_gpus
            * ``perf/throughput/rollout`` — rollout_tokens / rollout_time / n_gpus
            * ``perf/mfu``                — estimated_tflops / promised_tflops / n_train_gpus
            * ``perf/time_per_step``      — sum of all phase elapsed times
            * ``perf/total_tokens``       — sum of all phase token counts
        """
        phases = self._phases
        self._phases = {}  # reset immediately so any re-entrant call is clean

        # Aggregate totals
        total_tokens = sum(d.n_tokens for d in phases.values())
        total_time = sum(d.elapsed_sec for d in phases.values())

        # Per-category aggregates
        train_data = phases.get("train_step", _PhaseData())
        rollout_data = phases.get("rollout", _PhaseData())

        def _safe_throughput(tokens: int, time: float) -> float:
            if time <= 0 or self._n_gpus <= 0:
                return 0.0
            return tokens / time / self._n_gpus

        # MFU: only from train_step phase, uses n_train_gpus
        mfu = 0.0
        if (
            self._flops_counter is not None
            and train_data.seqlens
            and train_data.elapsed_sec > 0
            and self._n_train_gpus > 0
        ):
            try:
                estimated_tflops, promised_tflops = self._flops_counter.estimate_flops(
                    train_data.seqlens, train_data.elapsed_sec
                )
                if promised_tflops > 0:
                    mfu = estimated_tflops / promised_tflops / self._n_train_gpus
            except Exception:
                logger.warning("Failed to estimate FLOPs; MFU will be 0.", exc_info=True)

        return {
            "perf/throughput": _safe_throughput(total_tokens, total_time),
            "perf/throughput/train": _safe_throughput(train_data.n_tokens, train_data.elapsed_sec),
            "perf/throughput/rollout": _safe_throughput(
                rollout_data.n_tokens, rollout_data.elapsed_sec
            ),
            "perf/mfu": mfu,
            "perf/time_per_step": total_time,
            "perf/total_tokens": float(total_tokens),
        }
