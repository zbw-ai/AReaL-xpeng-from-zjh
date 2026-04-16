"""Verify Qwen3.5 MoE state dict adapter roundtrip (HF → Archon → HF)."""

import pytest
import torch
from types import SimpleNamespace

CUDA_AVAILABLE = torch.cuda.is_available()


def _make_mock_hf_config():
    return SimpleNamespace(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1024,
        head_dim=64,
        intermediate_size=512,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        partial_rotary_factor=0.25,
        max_position_embeddings=4096,
        eos_token_id=0,
        tie_word_embeddings=False,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        attention_bias=False,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=512,
        num_shared_experts=2,
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_state_dict_roundtrip():
    from areal.experimental.models.archon.qwen3_5.model.args import Qwen3_5ModelArgs
    from areal.experimental.models.archon.qwen3_5.model.model import Qwen3_5Model
    from areal.experimental.models.archon.qwen3_5.model.state_dict_adapter import (
        Qwen3_5StateDictAdapter,
    )

    hf_config = _make_mock_hf_config()
    model_args = Qwen3_5ModelArgs.from_hf_config(hf_config)
    model = Qwen3_5Model(model_args).to("cuda", dtype=torch.bfloat16)

    # Get Archon state dict
    archon_sd = model.state_dict()

    # Convert to HF format
    adapter = Qwen3_5StateDictAdapter(hf_config)
    hf_sd = adapter.to_hf(archon_sd)

    # Convert back to Archon format
    archon_sd_rt = adapter.from_hf(hf_sd)

    # Verify roundtrip
    for key in archon_sd:
        assert key in archon_sd_rt, f"Missing key after roundtrip: {key}"
        torch.testing.assert_close(
            archon_sd[key].float(),
            archon_sd_rt[key].float(),
            atol=0,
            rtol=0,
            msg=f"Mismatch for {key}",
        )

    print(f"Roundtrip OK: {len(archon_sd)} keys verified")
