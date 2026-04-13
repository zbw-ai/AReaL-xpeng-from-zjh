"""Unit tests for areal.utils.flops_counter.

Tests cover:
- Task 1: get_device_flops() GPU name -> peak FLOPS lookup
- Task 2: FlopsCounter class and per-architecture FLOPs estimators
"""

from unittest.mock import MagicMock

import pytest

from areal.utils.flops_counter import (
    FlopsCounter,
    _DEFAULT_DEVICE_FLOPS,
    _estimate_qwen2_flops,
    _estimate_qwen2_moe_flops,
    _estimate_unknown_flops,
    get_device_flops,
)


# ---------------------------------------------------------------------------
# Task 1: get_device_flops()
# ---------------------------------------------------------------------------


class TestGetDeviceFlops:
    def test_a100_exact(self):
        """A100 substring matches and returns correct peak FLOPS."""
        assert get_device_flops("NVIDIA A100 SXM4 80GB") == pytest.approx(312e12)

    def test_a100_short(self):
        """Short 'A100' string is matched."""
        assert get_device_flops("A100") == pytest.approx(312e12)

    def test_h100(self):
        """H100 returns 989 TFLOPS."""
        assert get_device_flops("NVIDIA H100 SXM5 80GB") == pytest.approx(989e12)

    def test_h800(self):
        """H800 returns 989 TFLOPS (same as H100)."""
        assert get_device_flops("H800") == pytest.approx(989e12)

    def test_l40s_not_l40(self):
        """L40S is matched before L40 (order-sensitive)."""
        assert get_device_flops("NVIDIA L40S") == pytest.approx(362.05e12)

    def test_l40_not_l40s(self):
        """Plain L40 (without S) is matched correctly."""
        assert get_device_flops("NVIDIA L40 48GB") == pytest.approx(181.05e12)

    def test_unknown_gpu_returns_default(self):
        """Completely unknown GPU returns A100 default (312 TFLOPS)."""
        assert get_device_flops("FutureGPU X9000") == pytest.approx(_DEFAULT_DEVICE_FLOPS)

    def test_unknown_gpu_logs_warning(self, caplog):
        """Unknown GPU emits a warning log."""
        import logging

        with caplog.at_level(logging.WARNING, logger="FlopsCounter"):
            get_device_flops("MyUnknownGPU 999")
        assert any("Unknown GPU" in r.message for r in caplog.records)

    def test_h200(self):
        """H200 returns 989 TFLOPS."""
        assert get_device_flops("NVIDIA H200") == pytest.approx(989e12)

    def test_gb200(self):
        """GB200 returns 2.5 PFLOPS."""
        assert get_device_flops("GB200") == pytest.approx(2.5e15)


# ---------------------------------------------------------------------------
# Helpers – build mock PretrainedConfig objects
# ---------------------------------------------------------------------------


def _make_qwen2_config(
    hidden_size: int = 2048,
    intermediate_size: int = 5504,
    num_hidden_layers: int = 24,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 8,
    vocab_size: int = 32000,
    model_type: str = "qwen2",
) -> MagicMock:
    """Create a minimal mock PretrainedConfig for a Qwen2/LLaMA-style model."""
    cfg = MagicMock(spec=["hidden_size", "intermediate_size", "num_hidden_layers",
                          "num_attention_heads", "num_key_value_heads", "vocab_size",
                          "model_type"])
    cfg.hidden_size = hidden_size
    cfg.intermediate_size = intermediate_size
    cfg.num_hidden_layers = num_hidden_layers
    cfg.num_attention_heads = num_attention_heads
    cfg.num_key_value_heads = num_key_value_heads
    cfg.vocab_size = vocab_size
    cfg.model_type = model_type
    return cfg


def _make_qwen2_moe_config(
    hidden_size: int = 2048,
    moe_intermediate_size: int = 1408,
    intermediate_size: int = 5504,
    num_hidden_layers: int = 24,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 8,
    vocab_size: int = 32000,
    num_experts_per_tok: int = 4,
    model_type: str = "qwen2_moe",
) -> MagicMock:
    """Create a minimal mock PretrainedConfig for a Qwen2-MoE-style model."""
    cfg = MagicMock(spec=["hidden_size", "moe_intermediate_size", "intermediate_size",
                          "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                          "vocab_size", "num_experts_per_tok", "model_type"])
    cfg.hidden_size = hidden_size
    cfg.moe_intermediate_size = moe_intermediate_size
    cfg.intermediate_size = intermediate_size
    cfg.num_hidden_layers = num_hidden_layers
    cfg.num_attention_heads = num_attention_heads
    cfg.num_key_value_heads = num_key_value_heads
    cfg.vocab_size = vocab_size
    cfg.num_experts_per_tok = num_experts_per_tok
    cfg.model_type = model_type
    return cfg


# ---------------------------------------------------------------------------
# Task 2: internal FLOPs estimators
# ---------------------------------------------------------------------------


class TestEstimateQwen2Flops:
    def _cfg(self, **kwargs):
        return _make_qwen2_config(**kwargs)

    def test_returns_positive(self):
        """FLOPs estimation returns a positive value for valid inputs."""
        cfg = self._cfg()
        result = _estimate_qwen2_flops(cfg, tokens_sum=1024, batch_seqlens=[512, 512], delta_time=1.0)
        assert result > 0

    def test_longer_seqs_more_flops(self):
        """Longer sequences produce more FLOPs (quadratic attention term)."""
        cfg = self._cfg()
        short = _estimate_qwen2_flops(cfg, tokens_sum=256, batch_seqlens=[256], delta_time=1.0)
        long_ = _estimate_qwen2_flops(cfg, tokens_sum=1024, batch_seqlens=[1024], delta_time=1.0)
        assert long_ > short

    def test_more_tokens_more_flops(self):
        """More tokens in the batch produce more FLOPs."""
        cfg = self._cfg()
        small = _estimate_qwen2_flops(cfg, tokens_sum=128, batch_seqlens=[128], delta_time=1.0)
        large = _estimate_qwen2_flops(cfg, tokens_sum=512, batch_seqlens=[512], delta_time=1.0)
        assert large > small

    def test_faster_time_higher_tflops(self):
        """Halving delta_time doubles the reported TFLOPS/s."""
        cfg = self._cfg()
        slow = _estimate_qwen2_flops(cfg, tokens_sum=512, batch_seqlens=[512], delta_time=2.0)
        fast = _estimate_qwen2_flops(cfg, tokens_sum=512, batch_seqlens=[512], delta_time=1.0)
        assert fast == pytest.approx(slow * 2, rel=1e-6)


class TestEstimateQwen2MoeFlops:
    def _cfg(self, **kwargs):
        return _make_qwen2_moe_config(**kwargs)

    def test_returns_positive(self):
        """MoE FLOPs estimation returns a positive value for valid inputs."""
        cfg = self._cfg()
        result = _estimate_qwen2_moe_flops(cfg, tokens_sum=512, batch_seqlens=[512], delta_time=1.0)
        assert result > 0

    def test_moe_less_than_dense_same_hidden(self):
        """MoE activated FLOPs < dense FLOPs when moe_intermediate_size < dense intermediate_size."""
        dense_cfg = _make_qwen2_config(intermediate_size=5504)
        moe_cfg = _make_qwen2_moe_config(
            moe_intermediate_size=1408, num_experts_per_tok=2, intermediate_size=5504
        )
        dense_flops = _estimate_qwen2_flops(dense_cfg, tokens_sum=512, batch_seqlens=[512], delta_time=1.0)
        moe_flops = _estimate_qwen2_moe_flops(moe_cfg, tokens_sum=512, batch_seqlens=[512], delta_time=1.0)
        # 2 experts * 1408 << 1 * 5504, so MoE is cheaper
        assert moe_flops < dense_flops

    def test_more_active_experts_more_flops(self):
        """Activating more experts increases FLOPs proportionally."""
        cfg2 = _make_qwen2_moe_config(num_experts_per_tok=2)
        cfg4 = _make_qwen2_moe_config(num_experts_per_tok=4)
        flops2 = _estimate_qwen2_moe_flops(cfg2, tokens_sum=512, batch_seqlens=[512], delta_time=1.0)
        flops4 = _estimate_qwen2_moe_flops(cfg4, tokens_sum=512, batch_seqlens=[512], delta_time=1.0)
        assert flops4 > flops2


class TestEstimateUnknownFlops:
    def test_returns_positive(self):
        """Fallback estimator returns a positive value."""
        cfg = MagicMock()
        cfg.hidden_size = 4096
        cfg.num_hidden_layers = 32
        cfg.vocab_size = 32000
        result = _estimate_unknown_flops(cfg, tokens_sum=512, batch_seqlens=[512], delta_time=1.0)
        assert result > 0


# ---------------------------------------------------------------------------
# Task 2: FlopsCounter class
# ---------------------------------------------------------------------------


class TestFlopsCounter:
    def _make_counter(self, model_type: str = "qwen2", device_name: str = "A100") -> FlopsCounter:
        """Build a FlopsCounter with a mock config and explicit device name."""
        cfg = _make_qwen2_config(model_type=model_type)
        counter = FlopsCounter.__new__(FlopsCounter)
        # Bypass __init__ to avoid AutoConfig.from_pretrained calls
        from areal.utils.flops_counter import _ESTIMATE_FUNC, _estimate_unknown_flops

        counter._config = cfg
        counter._promised_tflops = get_device_flops(device_name) / 1e12
        model_type_key = getattr(cfg, "model_type", "unknown")
        counter._estimate_fn = _ESTIMATE_FUNC.get(model_type_key, _estimate_unknown_flops)
        return counter

    def test_estimate_flops_returns_positive(self):
        """estimate_flops returns positive achieved TFLOPS for valid input."""
        counter = self._make_counter()
        tflops, peak = counter.estimate_flops(batch_seqlens=[512, 256], delta_time=1.0)
        assert tflops > 0
        assert peak > 0

    def test_longer_seqs_more_tflops(self):
        """Longer batch sequences produce higher estimated TFLOPS."""
        counter = self._make_counter()
        short_tflops, _ = counter.estimate_flops(batch_seqlens=[128], delta_time=1.0)
        long_tflops, _ = counter.estimate_flops(batch_seqlens=[2048], delta_time=1.0)
        assert long_tflops > short_tflops

    def test_promised_tflops_matches_device(self):
        """promised_tflops matches the known A100 peak."""
        counter = self._make_counter(device_name="A100")
        _, peak = counter.estimate_flops(batch_seqlens=[512], delta_time=1.0)
        assert peak == pytest.approx(312.0)  # 312 TFLOPS for A100

    def test_promised_tflops_h100(self):
        """promised_tflops matches the known H100 peak."""
        counter = self._make_counter(device_name="H100")
        _, peak = counter.estimate_flops(batch_seqlens=[512], delta_time=1.0)
        assert peak == pytest.approx(989.0)

    def test_zero_delta_time_returns_zero(self):
        """Non-positive delta_time returns (0.0, promised_tflops) gracefully."""
        counter = self._make_counter()
        tflops, peak = counter.estimate_flops(batch_seqlens=[512], delta_time=0.0)
        assert tflops == 0.0
        assert peak > 0

        tflops_neg, _ = counter.estimate_flops(batch_seqlens=[512], delta_time=-1.0)
        assert tflops_neg == 0.0

    def test_empty_seqlens_returns_zero(self):
        """Empty batch_seqlens returns (0.0, promised_tflops) gracefully."""
        counter = self._make_counter()
        tflops, peak = counter.estimate_flops(batch_seqlens=[], delta_time=1.0)
        assert tflops == 0.0
        assert peak > 0

    def test_unknown_model_type_does_not_crash(self):
        """An unknown model type falls back gracefully without raising."""
        counter = self._make_counter(model_type="future_arch")
        # _make_counter already sets up the fallback fn; just verify it runs
        tflops, peak = counter.estimate_flops(batch_seqlens=[512], delta_time=1.0)
        assert tflops >= 0.0
        assert peak > 0

    def test_unknown_model_type_logs_warning(self, caplog):
        """Constructing FlopsCounter with an unknown model_type logs a warning."""
        import logging

        cfg = _make_qwen2_config(model_type="future_arch_xyz")
        with caplog.at_level(logging.WARNING, logger="FlopsCounter"):
            counter = FlopsCounter(cfg, device_name="A100")
        assert any("Unknown model type" in r.message for r in caplog.records)
        # Should still work
        tflops, _ = counter.estimate_flops(batch_seqlens=[256], delta_time=1.0)
        assert tflops > 0

    def test_accepts_pretrained_config_directly(self):
        """FlopsCounter accepts a PretrainedConfig object directly."""
        cfg = _make_qwen2_config()
        counter = FlopsCounter(cfg, device_name="A100")
        tflops, peak = counter.estimate_flops(batch_seqlens=[512], delta_time=1.0)
        assert tflops > 0
        assert peak == pytest.approx(312.0)

    def test_moe_model_type(self):
        """FlopsCounter works correctly for qwen2_moe model type."""
        cfg = _make_qwen2_moe_config()
        counter = FlopsCounter(cfg, device_name="H100")
        tflops, peak = counter.estimate_flops(batch_seqlens=[512, 256], delta_time=2.0)
        assert tflops > 0
        assert peak == pytest.approx(989.0)
