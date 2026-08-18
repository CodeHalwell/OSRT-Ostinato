"""Tests for recursion + MoE health monitoring."""

import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM
from osrt.monitoring import loop_depth_probe, moe_health, summarize


def _tiny_model(**over):
    cfg = OSRTConfig(
        dim=128, heads=4, head_dim=32,
        vocab_size=256, real_vocab_size=256,
        num_blocks=2, recursive_loops=3,
        num_routed_experts=8, top_k_experts=2,
        expert_hidden=64, shared_expert_hidden=64,
        max_position_embeddings=64,
        aux_loop_loss_weight=0.05,
        **over,
    )
    return OSRTForCausalLM(cfg)


def test_moe_health_shapes_and_keys():
    model = _tiny_model()
    ids = torch.randint(0, 256, (2, 16))
    model(ids)  # populate routing buffers
    h = moe_health(model)
    assert h.num_experts == 8
    assert len(h.load_entropy) == 2  # blocks
    assert len(h.load_entropy[0]) == 3  # loops
    # entropy is a normalized [0, 1] quantity
    for blk in h.load_entropy:
        for e in blk:
            assert 0.0 <= e <= 1.0001


def test_loop_depth_probe_returns_per_loop_ce():
    model = _tiny_model()
    ids = torch.randint(0, 256, (2, 16))
    labels = ids.clone()
    d = loop_depth_probe(model, ids, labels)
    # one CE per loop (n-1 intermediate + final)
    assert len(d.per_loop_ce) == 3
    assert all(c > 0 for c in d.per_loop_ce)


def test_summarize_flattens_for_logging():
    model = _tiny_model()
    ids = torch.randint(0, 256, (2, 16))
    labels = ids.clone()
    flat, msg = summarize(model, ids, labels)
    assert any(k.startswith("moe/") for k in flat)
    assert any(k.startswith("loop_depth/") for k in flat)
    assert "moe/collapsing" in flat
    assert isinstance(msg, str) and msg


def test_detects_dead_experts_when_router_forced():
    """Force the router to one expert -> monitor must flag collapse."""
    model = _tiny_model()
    with torch.no_grad():
        # crank one expert's router logit so routing collapses onto it
        for block in model.model.blocks:
            block.moe.router.weight.zero_()
            block.moe.router.weight[0] += 50.0
            block.moe.bias_enabled = False  # don't let the balance bias rescue it
    ids = torch.randint(0, 256, (4, 16))
    model(ids)
    h = moe_health(model)
    assert h.collapsing, "monitor failed to flag a forced single-expert collapse"


def test_balance_loss_normalises_by_actual_loops_under_dropout():
    """Regression for the loop-dropout normalization bug.

    When loop_dropout_prob fires and shortens the loop chain, the router
    balance / z / seq-balance losses must be divided by the ACTUAL number
    of loops run, not the configured depth. Otherwise the regularizer is
    under-weighted exactly on the stochastic-depth batches that need it
    most.

    Uses the default recursive_loops=3, num_blocks=2. With dropout=1.0 and
    min_loops=2, loops_run is sampled from {2, 3}; the actual MoE-layer
    count is therefore in {4, 6}. The buggy denominator was always 6.
    """
    import random

    model = _tiny_model(
        loop_dropout_prob=1.0,
        loop_dropout_min_loops=2,
    )
    model.train()  # dropout only fires in train mode
    ids = torch.randint(0, 256, (2, 16))
    labels = ids.clone()

    # Across many seeded runs, the normalization ratio must sometimes
    # be 4 (loops_run=2). If only 6 ever appears, the denominator is
    # still bolted to the configured depth and the bug is back.
    ratios = set()
    for seed in range(20):
        random.seed(seed)
        torch.manual_seed(seed)
        model(input_ids=ids, labels=labels)
        raw = model.last_balance_loss
        norm = model.last_balance_loss_normalised
        assert raw is not None and norm is not None
        ratios.add(round(float(raw) / float(norm), 1))

    assert ratios.issubset({4.0, 6.0}), (
        f"ratios outside expected {{4.0, 6.0}}: {sorted(ratios)}"
    )
    assert 4.0 in ratios, (
        f"no run produced ratio 4.0 (loops_run=2) — denominator not "
        f"adapting to shortened loops. Saw ratios: {sorted(ratios)}"
    )
