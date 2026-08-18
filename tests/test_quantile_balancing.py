"""Quantile Balancing (Kimi K3) — the aux-loss-free balance controller v7 requires.

The heuristic controller nudges each expert's bias by ±gamma in proportion to
its load error. Its step size was tuned at E=8; at v7's E=28 it has 3.5x as
many biases to move with 3.5x less load signal per expert, and a dead expert
costs 3.6% of block capacity (roadmap §14.6).

QB instead sets each bias directly from the router-score quantile matching the
expert's target load, in one shot, with no update rate / EMA / clamp target.
"""

import pytest
import torch

from osrt.config import OSRTConfig
from osrt.model import MoELayer


def _cfg(**over):
    base = dict(
        dim=64, heads=4, head_dim=16, num_kv_heads=2,
        vocab_size=256, real_vocab_size=256, num_blocks=1, recursive_loops=2,
        num_routed_experts=8, top_k_experts=2, expert_hidden=32,
        shared_expert_hidden=32, max_position_embeddings=64,
        router_affinity="sqrt_softplus",
    )
    base.update(over)
    return OSRTConfig(**base)


def _moe(**over):
    return MoELayer(_cfg(**over), block_idx=0)


def _feed(moe, skew=None, n=512, loop=0):
    """Accumulate balance stats from one skewed batch of router scores."""
    torch.manual_seed(0)
    x = torch.randn(n, moe.num_routed)
    if skew is not None:
        x[:, skew] += 3.0            # make one expert dominate
    affinity = torch.sqrt(torch.nn.functional.softplus(x))
    top_idx = affinity.topk(moe.top_k, dim=-1).indices
    moe._accumulate_balance_counts(top_idx, loop, score=affinity)
    return affinity


# ── config plumbing ──────────────────────────────────────────────────────

def test_default_is_heuristic_and_preset_is_quantile():
    assert _cfg().router_balance_mode == "heuristic"
    from osrt.presets import OSRT_V7
    assert OSRT_V7["router_balance_mode"] == "quantile"


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="router_balance_mode"):
        _cfg(router_balance_mode="ema")


# ── the load-equalising property ─────────────────────────────────────────

def test_overloaded_expert_gets_the_lowest_bias():
    moe = _moe(router_balance_mode="quantile")
    _feed(moe, skew=3)
    moe.apply_balance_update()
    bias = moe.router_balance_bias[0]
    assert bias.argmin().item() == 3, (
        f"expert 3 was over-selected, so it must receive the lowest bias; got {bias}")


def test_biases_are_centred_so_they_cannot_drift():
    moe = _moe(router_balance_mode="quantile")
    _feed(moe, skew=1)
    moe.apply_balance_update()
    assert moe.router_balance_bias[0].sum().abs().item() < 1e-4


def test_quantile_balance_reduces_load_imbalance():
    """The whole point: applying the bias must flatten realised top-k load."""
    moe = _moe(router_balance_mode="quantile")
    affinity = _feed(moe, skew=5)

    def spread(bias):
        idx = (affinity + bias).topk(moe.top_k, dim=-1).indices
        counts = torch.bincount(idx.reshape(-1), minlength=moe.num_routed)
        return (counts.float() / counts.sum()).std().item()

    before = spread(torch.zeros(moe.num_routed))
    moe.apply_balance_update()
    after = spread(moe.router_balance_bias[0])
    assert after < before, f"QB did not flatten load: {before:.4f} -> {after:.4f}"


def test_balanced_input_produces_near_zero_bias():
    moe = _moe(router_balance_mode="quantile")
    _feed(moe, skew=None)
    moe.apply_balance_update()
    assert moe.router_balance_bias[0].abs().max().item() < 0.5


# ── determinism and state hygiene ────────────────────────────────────────

def test_update_is_deterministic_and_stateless_across_repeats():
    """Same evidence twice -> same bias. The heuristic controller would keep
    integrating; QB is a one-shot solve, so a repeat must reproduce itself."""
    a, b = _moe(router_balance_mode="quantile"), _moe(router_balance_mode="quantile")
    _feed(a, skew=2)
    a.apply_balance_update()
    _feed(b, skew=2)
    b.apply_balance_update()
    assert torch.allclose(a.router_balance_bias, b.router_balance_bias)

    first = a.router_balance_bias.clone()
    _feed(a, skew=2)
    a.apply_balance_update()
    assert torch.allclose(a.router_balance_bias, first), \
        "QB must be a solve, not an integrator"


def test_histogram_clears_after_update():
    moe = _moe(router_balance_mode="quantile")
    _feed(moe, skew=0)
    assert moe.qb_hist.sum().item() > 0
    moe.apply_balance_update()
    assert moe.qb_hist.sum().item() == 0
    assert moe.qb_token_count.sum().item() == 0


def test_no_update_without_evidence():
    moe = _moe(router_balance_mode="quantile")
    moe.apply_balance_update()
    assert torch.count_nonzero(moe.router_balance_bias).item() == 0


def test_per_loop_biases_are_independent():
    moe = _moe(router_balance_mode="quantile")
    _feed(moe, skew=1, loop=0)
    _feed(moe, skew=6, loop=1)
    moe.apply_balance_update()
    assert moe.router_balance_bias[0].argmin().item() == 1
    assert moe.router_balance_bias[1].argmin().item() == 6


def test_heuristic_path_untouched_by_the_new_mode():
    moe = _moe(router_balance_mode="heuristic")
    _feed(moe, skew=4)
    moe.apply_balance_update()
    assert moe.router_balance_bias[0].argmin().item() == 4
    assert moe.qb_hist.sum().item() == 0, "heuristic mode must not build histograms"
