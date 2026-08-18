"""Tests for the passive weight EMA used as a GRPO evaluation observer.

The EMA is only trustworthy if three things hold, and none is obvious by
inspection: the recursion matches the closed form, the live model is never
mutated, and EVERY persistent state_dict entry is covered. The last one is
OSRT-specific — `router_balance_bias` and `gumbel_tau` are mutable persistent
buffers read at eval time, so averaging parameters alone would pair averaged
weights with the latest router state and produce a hybrid model.
"""
from __future__ import annotations

import torch
from torch import nn

from osrt.grpo_train import ema_init, ema_update, ema_weight_of_init


class _Tiny(nn.Module):
    """Stands in for OSRT's shape: params PLUS mutable persistent buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3, bias=True)
        # mirrors router_balance_bias (mutated in training, read at eval)
        self.register_buffer("router_balance_bias", torch.zeros(6, 8))
        self.register_buffer("gumbel_tau", torch.tensor(1.0))
        # mirrors rope tables (constant) and a non-persistent buffer
        self.register_buffer("rope_cos", torch.ones(2, 2))
        self.register_buffer("scratch", torch.zeros(2), persistent=False)


def test_ema_covers_every_persistent_state_dict_entry():
    m = _Tiny()
    ema = ema_init(m)
    assert set(ema) == set(m.state_dict())
    # the mutable buffers must be present, not just parameters
    assert "router_balance_bias" in ema
    assert "gumbel_tau" in ema
    # non-persistent buffers are absent from state_dict, so absent here too
    assert "scratch" not in ema


def test_ema_shadow_is_fp32_even_from_a_bf16_model():
    m = _Tiny().to(torch.bfloat16)
    ema = ema_init(m)
    assert all(v.dtype is torch.float32 for v in ema.values()), (
        "bf16 has 8 mantissa bits and cannot accumulate 1%-weighted updates"
    )


def test_ema_matches_closed_form_for_a_constant_target():
    """With the live weights held constant at w, after n updates from e0 the
    shadow is exactly  w + (e0 - w) * decay**n."""
    m = _Tiny()
    decay, n = 0.9, 25
    with torch.no_grad():
        m.lin.weight.fill_(2.0)
    ema = ema_init(m)
    with torch.no_grad():
        ema["lin.weight"].fill_(0.0)          # e0 = 0, target w = 2
    for _ in range(n):
        ema_update(ema, m, decay)
    expected = 2.0 + (0.0 - 2.0) * decay ** n
    assert torch.allclose(ema["lin.weight"],
                          torch.full_like(ema["lin.weight"], expected),
                          atol=1e-6), ema["lin.weight"].flatten()[0]


def test_ema_update_does_not_mutate_the_model():
    """The observer must be passive: theta and its buffers stay untouched."""
    m = _Tiny()
    with torch.no_grad():
        m.router_balance_bias.normal_()
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    ema = ema_init(m)
    with torch.no_grad():                     # make the shadow differ from live
        for v in ema.values():
            v.add_(1.0)
    for _ in range(5):
        ema_update(ema, m, 0.99)
    after = m.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"model tensor {k} was mutated"


def test_ema_tracks_a_moving_target_and_lags_it():
    """A moving target: the shadow must move toward it but stay behind."""
    m = _Tiny()
    with torch.no_grad():
        m.lin.weight.fill_(0.0)
    ema = ema_init(m)
    for i in range(1, 21):
        with torch.no_grad():
            m.lin.weight.fill_(float(i))      # live races ahead
        ema_update(ema, m, 0.99)
    live = float(m.lin.weight.detach().flatten()[0])
    shadow = float(ema["lin.weight"].flatten()[0])
    assert 0.0 < shadow < live, (shadow, live)


def test_ema_update_rejects_a_key_mismatch():
    m = _Tiny()
    ema = ema_init(m)
    ema["not_a_real_key"] = torch.zeros(1)
    try:
        ema_update(ema, m, 0.99)
    except KeyError:
        return
    raise AssertionError("a key absent from the model should raise")


def test_residual_weight_on_init_is_reported_honestly():
    """Guards against crediting an early EMA win to averaging."""
    assert ema_weight_of_init(0.99, 0) == 1.0
    assert abs(ema_weight_of_init(0.99, 50) - 0.605) < 0.01
    assert abs(ema_weight_of_init(0.99, 100) - 0.366) < 0.01
    # at 900 steps the base has essentially washed out
    assert ema_weight_of_init(0.99, 900) < 1e-3
