"""Unit tests for areal.utils.perf_metrics.PerfMetrics."""

from unittest.mock import MagicMock

import pytest

from areal.utils.perf_metrics import PerfMetrics


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_counter():
    """FlopsCounter mock: estimate_flops returns (100.0, 312.0) TFLOPS."""
    counter = MagicMock()
    counter.estimate_flops.return_value = (100.0, 312.0)
    return counter


# ---------------------------------------------------------------------------
# Throughput tests
# ---------------------------------------------------------------------------


def test_train_throughput(mock_counter):
    pm = PerfMetrics(mock_counter, n_gpus=16, n_train_gpus=8)
    pm.record("train_step", n_tokens=10000, elapsed_sec=1.0)
    result = pm.compute()
    assert result["perf/throughput/train"] == pytest.approx(10000 / 1.0 / 16)


def test_overall_throughput(mock_counter):
    pm = PerfMetrics(mock_counter, n_gpus=16, n_train_gpus=8)
    pm.record("rollout", n_tokens=8000, elapsed_sec=2.0)
    pm.record("train_step", n_tokens=8000, elapsed_sec=1.0)
    result = pm.compute()
    assert result["perf/throughput"] == pytest.approx(16000 / 3.0 / 16)


def test_rollout_throughput(mock_counter):
    pm = PerfMetrics(mock_counter, n_gpus=16, n_train_gpus=8)
    pm.record("rollout", n_tokens=8000, elapsed_sec=2.0)
    pm.record("train_step", n_tokens=8000, elapsed_sec=1.0)
    result = pm.compute()
    assert result["perf/throughput/rollout"] == pytest.approx(8000 / 2.0 / 16)


# ---------------------------------------------------------------------------
# MFU tests
# ---------------------------------------------------------------------------


def test_mfu_uses_train_gpus(mock_counter):
    """MFU denominator must be n_train_gpus, not n_gpus."""
    pm = PerfMetrics(mock_counter, n_gpus=16, n_train_gpus=8)
    pm.record("train_step", n_tokens=1000, elapsed_sec=1.0, seqlens=[512, 488])
    result = pm.compute()
    # mock returns (100.0, 312.0); MFU = 100.0 / 312.0 / 8
    assert result["perf/mfu"] == pytest.approx(100.0 / 312.0 / 8)


def test_none_flops_counter():
    """When flops_counter is None, MFU is 0 but throughput still works."""
    pm = PerfMetrics(None, n_gpus=16, n_train_gpus=8)
    pm.record("train_step", n_tokens=10000, elapsed_sec=1.0, seqlens=[512])
    result = pm.compute()
    assert result["perf/mfu"] == pytest.approx(0.0)
    assert result["perf/throughput/train"] == pytest.approx(10000 / 1.0 / 16)


def test_mfu_zero_without_seqlens(mock_counter):
    """MFU is 0 when no seqlens are provided for train_step."""
    pm = PerfMetrics(mock_counter, n_gpus=16, n_train_gpus=8)
    pm.record("train_step", n_tokens=10000, elapsed_sec=1.0)  # no seqlens
    result = pm.compute()
    assert result["perf/mfu"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reset and edge-case tests
# ---------------------------------------------------------------------------


def test_reset_after_compute(mock_counter):
    """Second compute() after no new records returns zeros."""
    pm = PerfMetrics(mock_counter, n_gpus=16, n_train_gpus=8)
    pm.record("train_step", n_tokens=10000, elapsed_sec=1.0)
    pm.compute()  # first compute; resets accumulators
    result = pm.compute()  # second compute; nothing recorded
    assert result["perf/throughput"] == pytest.approx(0.0)
    assert result["perf/throughput/train"] == pytest.approx(0.0)
    assert result["perf/throughput/rollout"] == pytest.approx(0.0)
    assert result["perf/mfu"] == pytest.approx(0.0)
    assert result["perf/time_per_step"] == pytest.approx(0.0)
    assert result["perf/total_tokens"] == pytest.approx(0.0)


def test_zero_time_no_crash(mock_counter):
    """elapsed_sec=0.0 must not raise; throughput should be 0."""
    pm = PerfMetrics(mock_counter, n_gpus=16, n_train_gpus=8)
    pm.record("train_step", n_tokens=10000, elapsed_sec=0.0)
    result = pm.compute()
    assert result["perf/throughput/train"] == pytest.approx(0.0)
    assert result["perf/mfu"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Aggregate metric tests
# ---------------------------------------------------------------------------


def test_time_per_step(mock_counter):
    """perf/time_per_step is the sum of all phase elapsed times."""
    pm = PerfMetrics(mock_counter, n_gpus=16, n_train_gpus=8)
    pm.record("rollout", n_tokens=5000, elapsed_sec=2.5)
    pm.record("train_step", n_tokens=5000, elapsed_sec=1.5)
    result = pm.compute()
    assert result["perf/time_per_step"] == pytest.approx(4.0)


def test_total_tokens(mock_counter):
    """perf/total_tokens is the sum of all phase token counts."""
    pm = PerfMetrics(mock_counter, n_gpus=16, n_train_gpus=8)
    pm.record("rollout", n_tokens=6000, elapsed_sec=1.0)
    pm.record("train_step", n_tokens=4000, elapsed_sec=1.0)
    result = pm.compute()
    assert result["perf/total_tokens"] == pytest.approx(10000.0)
