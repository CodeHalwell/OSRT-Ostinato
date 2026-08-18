"""WSD (warmup-stable-decay) LR schedule — roadmap item 0.2.

The reason v7 switches off cosine is economic, not aesthetic. This project
trains in drip-funded chunks; under cosine every extension either re-warms or
reshapes the curve mid-run, and §2.2 records the token budget as the plan's
weakest link. WSD holds LR flat through the trunk so a run can stop and resume
anywhere at no cost, and only the release branch pays the decay.
"""

from osrt.train import get_lr
from osrt.train_config import PretrainConfig


def _cfg(**over):
    c = PretrainConfig()
    for k, v in over.items():
        setattr(c, k, v)
    return c


def test_default_schedule_is_wsd():
    assert PretrainConfig().lr_schedule == "wsd"


def test_warmup_is_linear_and_shared_by_both_schedules():
    for sched in ("wsd", "cosine"):
        c = _cfg(lr_schedule=sched, warmup_steps=100, peak_lr=1e-3)
        assert get_lr(0, c) == 0.0
        assert abs(get_lr(50, c) - 5e-4) < 1e-12
        assert abs(get_lr(100, c) - 1e-3) < 1e-9


def test_stable_phase_is_flat_at_peak():
    c = _cfg(lr_schedule="wsd", warmup_steps=100, total_steps=1000,
             wsd_decay_frac=0.2, peak_lr=1e-3, min_lr=1e-4)
    # decay starts at 800; everything between warmup and there is flat
    for step in (100, 300, 500, 799):
        assert abs(get_lr(step, c) - 1e-3) < 1e-12, f"not flat at {step}"


def test_decay_is_linear_to_min_lr_and_terminates_there():
    c = _cfg(lr_schedule="wsd", warmup_steps=100, total_steps=1000,
             wsd_decay_frac=0.2, peak_lr=1e-3, min_lr=1e-4)
    assert abs(get_lr(900, c) - 5.5e-4) < 1e-9      # halfway down
    assert abs(get_lr(1000, c) - 1e-4) < 1e-12
    assert abs(get_lr(1500, c) - 1e-4) < 1e-12      # clamped past the end


def test_resuming_mid_trunk_costs_nothing():
    """The property the switch exists for: two adjacent steps in the stable
    phase have identical LR, so a stop/resume there is free."""
    c = _cfg(lr_schedule="wsd", warmup_steps=100, total_steps=10_000,
             wsd_decay_frac=0.2, peak_lr=6e-4)
    assert get_lr(2_000, c) == get_lr(2_001, c) == 6e-4


def test_cosine_still_available_for_historical_runs():
    c = _cfg(lr_schedule="cosine", warmup_steps=100, total_steps=1000,
             peak_lr=1e-3, min_lr=1e-4)
    mid = get_lr(550, c)
    assert 1e-4 < mid < 1e-3
    assert get_lr(200, c) > get_lr(800, c)          # monotone decreasing
