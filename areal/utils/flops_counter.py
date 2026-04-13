"""GPU peak FLOPS lookup and model FLOPs estimation utilities.

Ported and adapted from verl v070's verl/utils/flops_counter.py.
"""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig

from areal.utils import logging

logger = logging.getLogger("FlopsCounter")

# BF16 peak TFLOPS per GPU device type.
# Values are in FLOPS (not TFLOPS), keyed by substring match against device name.
_DEVICE_FLOPS: dict[str, float] = {
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

# Default FLOPS when GPU is unknown (A100 baseline)
_DEFAULT_DEVICE_FLOPS = 312e12


def get_device_flops(device_name: str) -> float:
    """Return BF16 peak FLOPS for the given GPU device name.

    Performs substring matching against the known device table.  The match
    order follows the declaration order of ``_DEVICE_FLOPS``; longer/more
    specific names (e.g. ``"L40S"``) must come before shorter prefixes
    (``"L40"``) in the table to avoid false matches.

    Args:
        device_name: GPU name string (e.g. ``"NVIDIA H100 SXM5 80GB"``).

    Returns:
        BF16 peak FLOPS as a float.  Falls back to ``_DEFAULT_DEVICE_FLOPS``
        (A100, 312 TFLOPS) for unknown devices with a warning.
    """
    for key, flops in _DEVICE_FLOPS.items():
        if key in device_name:
            return flops

    logger.warning(
        "Unknown GPU device '%s'; defaulting to A100 peak FLOPS (312 TFLOPS).",
        device_name,
    )
    return _DEFAULT_DEVICE_FLOPS


# ---------------------------------------------------------------------------
# Internal FLOPs estimation helpers
# ---------------------------------------------------------------------------


def _estimate_qwen2_flops(
    config: PretrainedConfig,
    tokens_sum: int,
    batch_seqlens: list[int],
    delta_time: float,
) -> float:
    """Estimate TFLOPS/s for dense Qwen2/Qwen3/LLaMA models.

    Args:
        config: HuggingFace ``PretrainedConfig`` with model architecture fields.
        tokens_sum: Total number of tokens in the batch.
        batch_seqlens: Per-sample sequence lengths.
        delta_time: Wall-clock time in seconds for the measured operation.

    Returns:
        Estimated throughput in TFLOPS/s.
    """
    hidden_size: int = config.hidden_size
    intermediate_size: int = config.intermediate_size
    num_hidden_layers: int = config.num_hidden_layers
    num_attention_heads: int = config.num_attention_heads
    num_key_value_heads: int = getattr(config, "num_key_value_heads", num_attention_heads)
    vocab_size: int = config.vocab_size
    head_dim: int = hidden_size // num_attention_heads

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # Parameter counts (weight matrices only; biases ignored per convention)
    mlp_n = hidden_size * intermediate_size * 3  # SwiGLU: gate, up, down
    attn_linear_n = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emb_and_lm_head_n = vocab_size * hidden_size * 2

    dense_n = (mlp_n + attn_linear_n) * num_hidden_layers + emb_and_lm_head_n

    # Linear-layer FLOPs: 2 multiply-adds per weight, forward + backward = 6×
    dense_n_flops = 6 * dense_n * tokens_sum

    # Attention QK^T and AV FLOPs (quadratic in sequence length)
    seqlen_square_sum = sum(s * s for s in batch_seqlens)
    attn_qkv_flops = (
        6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
    )

    total_flops = dense_n_flops + attn_qkv_flops
    return total_flops / delta_time / 1e12


def _estimate_qwen2_moe_flops(
    config: PretrainedConfig,
    tokens_sum: int,
    batch_seqlens: list[int],
    delta_time: float,
) -> float:
    """Estimate TFLOPS/s for MoE Qwen2-MoE/Qwen3-MoE models.

    Uses ``num_experts_per_tok`` (activated experts) rather than total expert
    count for the MLP FLOPs, reflecting actual compute during a forward pass.

    Args:
        config: HuggingFace ``PretrainedConfig`` with MoE architecture fields.
        tokens_sum: Total number of tokens in the batch.
        batch_seqlens: Per-sample sequence lengths.
        delta_time: Wall-clock time in seconds for the measured operation.

    Returns:
        Estimated throughput in TFLOPS/s.
    """
    hidden_size: int = config.hidden_size
    # For MoE models the MLP intermediate is per-expert
    moe_intermediate_size: int = getattr(
        config, "moe_intermediate_size", config.intermediate_size
    )
    num_hidden_layers: int = config.num_hidden_layers
    num_attention_heads: int = config.num_attention_heads
    num_key_value_heads: int = getattr(config, "num_key_value_heads", num_attention_heads)
    vocab_size: int = config.vocab_size
    head_dim: int = hidden_size // num_attention_heads
    num_experts_per_tok: int = getattr(config, "num_experts_per_tok", 1)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # Only count activated experts (num_experts_per_tok) for MLP FLOPs
    mlp_n = hidden_size * moe_intermediate_size * 3 * num_experts_per_tok
    attn_linear_n = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emb_and_lm_head_n = vocab_size * hidden_size * 2

    dense_n = (mlp_n + attn_linear_n) * num_hidden_layers + emb_and_lm_head_n

    dense_n_flops = 6 * dense_n * tokens_sum

    seqlen_square_sum = sum(s * s for s in batch_seqlens)
    attn_qkv_flops = (
        6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
    )

    total_flops = dense_n_flops + attn_qkv_flops
    return total_flops / delta_time / 1e12


def _estimate_unknown_flops(
    config: PretrainedConfig,
    tokens_sum: int,
    batch_seqlens: list[int],
    delta_time: float,
) -> float:
    """Fallback FLOPs estimator using a rough parameter count heuristic.

    Estimates total parameters from ``hidden_size``, ``num_hidden_layers``,
    and ``vocab_size`` when a model-type-specific estimator is unavailable.

    Args:
        config: HuggingFace ``PretrainedConfig``.
        tokens_sum: Total number of tokens in the batch.
        batch_seqlens: Per-sample sequence lengths (unused in this estimator).
        delta_time: Wall-clock time in seconds for the measured operation.

    Returns:
        Estimated throughput in TFLOPS/s.
    """
    hidden_size: int = getattr(config, "hidden_size", 4096)
    num_hidden_layers: int = getattr(config, "num_hidden_layers", 32)
    vocab_size: int = getattr(config, "vocab_size", 32000)

    # Rough estimate: 12 * hidden² * layers + 2 * vocab * hidden (embeddings)
    estimated_total_params = (
        12 * hidden_size * hidden_size * num_hidden_layers + 2 * vocab_size * hidden_size
    )

    total_flops = 6 * estimated_total_params * tokens_sum
    return total_flops / delta_time / 1e12


# Map config.model_type → FLOPs estimator function
_ESTIMATE_FUNC = {
    "qwen2": _estimate_qwen2_flops,
    "qwen3": _estimate_qwen2_flops,
    "llama": _estimate_qwen2_flops,
    "qwen2_moe": _estimate_qwen2_moe_flops,
    "qwen3_moe": _estimate_qwen2_moe_flops,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class FlopsCounter:
    """Estimates model FLOPS utilisation (MFU) during training or inference.

    Given a HuggingFace model config and the GPU device name, ``FlopsCounter``
    can estimate the achieved TFLOPS/s for a batch and compare it against the
    hardware's theoretical peak TFLOPS.

    Example::

        counter = FlopsCounter("Qwen/Qwen2-7B", device_name="NVIDIA H100")
        tflops, peak = counter.estimate_flops([512, 256, 1024], delta_time=1.5)
        mfu = tflops / peak

    Args:
        config: A ``PretrainedConfig`` instance **or** a model name/path string
            (passed to ``AutoConfig.from_pretrained``).
        device_name: GPU device name used for peak FLOPS lookup.  When
            ``None``, tries ``torch.cuda.get_device_name(0)``; falls back to
            ``"Unknown GPU"`` if CUDA is unavailable.
    """

    def __init__(
        self,
        config: PretrainedConfig | str,
        device_name: str | None = None,
    ) -> None:
        if isinstance(config, str):
            config = AutoConfig.from_pretrained(config, trust_remote_code=True)
        self._config = config

        if device_name is None:
            try:
                import torch

                device_name = torch.cuda.get_device_name(0)
            except Exception:
                device_name = "Unknown GPU"

        model_type: str = getattr(config, "model_type", "unknown")
        if model_type not in _ESTIMATE_FUNC:
            logger.warning(
                "Unknown model type '%s'; using fallback FLOPs estimator.",
                model_type,
            )
        self._estimate_fn = _ESTIMATE_FUNC.get(model_type, _estimate_unknown_flops)
        self._promised_tflops: float = get_device_flops(device_name) / 1e12

    def estimate_flops(
        self,
        batch_seqlens: list[int],
        delta_time: float,
    ) -> tuple[float, float]:
        """Estimate achieved TFLOPS/s for a batch and return hardware peak.

        Args:
            batch_seqlens: List of sequence lengths for each sample in the batch.
            delta_time: Wall-clock time (seconds) taken for the operation.

        Returns:
            ``(estimated_tflops_per_sec, promised_tflops_per_sec)`` where the
            first element is the model's achieved throughput and the second is
            the device's theoretical peak BF16 throughput.  Both values are in
            TFLOPS/s.  Returns ``(0.0, promised_tflops)`` when ``delta_time``
            is non-positive or ``batch_seqlens`` is empty.
        """
        if delta_time <= 0 or not batch_seqlens:
            return 0.0, self._promised_tflops

        tokens_sum = sum(batch_seqlens)
        estimated = self._estimate_fn(
            self._config, tokens_sum, batch_seqlens, delta_time
        )
        return estimated, self._promised_tflops
