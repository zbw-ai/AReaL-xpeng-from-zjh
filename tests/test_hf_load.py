import pytest
import torch

pytest.importorskip("mbridge")

from areal.models.mcore.hf_load import _merge_qkv_weights, _slice_generic_weight


def test_slice_generic_weight_multi_slices_tp_shard():
    # Simulate two HF tensors that should be concatenated along dim 0, then TP sliced.
    first = torch.arange(24, dtype=torch.float32).view(6, 4)
    second = torch.arange(24, 48, dtype=torch.float32).view(6, 4)

    loaded = _slice_generic_weight(
        mcore_param_shape=[6, 4],
        hf_weights_safe_slice=[first, second],
        tp_rank=1,
        tp_size=2,
        mcore_weights_name="decoder.layers.0.some.weight",
    )

    torch.testing.assert_close(loaded, second)


def test_slice_generic_weight_multi_slices_no_tp():
    # Simulate a non-sharded local parameter composed from two HF slices.
    first = torch.arange(8, dtype=torch.float32).view(2, 4)
    second = torch.arange(8, 16, dtype=torch.float32).view(2, 4)

    loaded = _slice_generic_weight(
        mcore_param_shape=[2, 8],
        hf_weights_safe_slice=[first, second],
        tp_rank=0,
        tp_size=1,
        mcore_weights_name="decoder.layers.0.some.weight",
    )

    expected = torch.cat([first, second], dim=1)
    torch.testing.assert_close(loaded, expected)


def test_slice_generic_weight_multi_slices_raises_when_not_mergeable():
    first = torch.ones((2, 3), dtype=torch.float32)
    second = torch.ones((4, 5), dtype=torch.float32)

    with pytest.raises(ValueError, match="Cannot infer how to merge generic HF weights"):
        _slice_generic_weight(
            mcore_param_shape=[2, 3],
            hf_weights_safe_slice=[first, second],
            tp_rank=0,
            tp_size=1,
            mcore_weights_name="decoder.layers.0.some.weight",
        )


def test_merge_qkv_weights_supports_num_kv_heads_attr():
    class _Cfg:
        hidden_size = 8
        num_attention_heads = 4
        num_kv_heads = 2
        head_dim = 2

    q = torch.randn(8, 8)
    k = torch.randn(4, 8)
    v = torch.randn(4, 8)
    loaded = _merge_qkv_weights(
        hf_config=_Cfg(),
        mcore_weights_name="module.module.decoder.layers.0.self_attention.linear_qkv.weight",
        hf_weights_safe_slice=[q, k, v],
        tp_rank=0,
        tp_size=2,
    )
    assert list(loaded.shape) == [8, 8]
