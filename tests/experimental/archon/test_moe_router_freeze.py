"""Test MoE router freezing during RL training."""

import torch


def test_freeze_moe_router_basic():
    """Test that freeze_moe_router disables gradient for router weights."""
    from areal.experimental.engine.archon_engine import freeze_moe_router

    model = torch.nn.Module()
    layer = torch.nn.Module()
    moe = torch.nn.Module()
    router = torch.nn.Module()
    router.gate = torch.nn.Linear(64, 8, bias=False)
    moe.router = router
    moe.experts_w1 = torch.nn.Parameter(torch.randn(8, 64, 32))
    layer.moe = moe
    model.layers = torch.nn.ModuleDict({"0": layer})

    assert router.gate.weight.requires_grad is True

    count = freeze_moe_router(model)

    assert router.gate.weight.requires_grad is False
    assert moe.experts_w1.requires_grad is True
    assert count == 1


def test_freeze_moe_router_no_moe():
    """Test that freeze_moe_router is a no-op for dense models."""
    from areal.experimental.engine.archon_engine import freeze_moe_router

    model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.feed_forward = torch.nn.Linear(64, 64)
    model.layers = torch.nn.ModuleDict({"0": layer})

    count = freeze_moe_router(model)
    assert count == 0
    assert layer.feed_forward.weight.requires_grad is True
