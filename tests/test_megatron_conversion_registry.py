import pytest

pytest.importorskip("megatron.core")

from areal.engine.megatron_utils.megatron import (
    _resolve_conversion_fn,
    convert_qwen2_to_hf,
    convert_qwen3_5_to_hf,
)


def test_resolve_conversion_fn_prefers_qwen3_5_over_qwen3_substring():
    fn = _resolve_conversion_fn("qwen3_5_moe")
    assert fn is convert_qwen3_5_to_hf


def test_resolve_conversion_fn_exact_match_has_priority():
    fn = _resolve_conversion_fn("qwen3")
    assert fn is convert_qwen2_to_hf
