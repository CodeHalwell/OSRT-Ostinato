"""--cooldown-steps schedule arithmetic and the checkpoint-soup average."""

import torch

from osrt.soup import average_state_dicts
from osrt.train import get_lr
from osrt.train_config import MidtrainConfig
from osrt.train_main import cooldown_schedule


def test_cooldown_holds_peak_then_decays_to_min():
    cfg = MidtrainConfig()
    cfg.total_steps, cfg.wsd_decay_frac = cooldown_schedule(13_500, 1_000)
    assert cfg.total_steps == 14_500
    assert get_lr(13_500, cfg) == cfg.peak_lr
    assert get_lr(13_499, cfg) == cfg.peak_lr
    assert (
        abs(get_lr(14_000, cfg) - (cfg.peak_lr + cfg.min_lr) / 2) < cfg.peak_lr * 0.01
    )
    assert abs(get_lr(14_500, cfg) - cfg.min_lr) < 1e-12


def test_average_state_dicts_means_floats_and_keeps_buffers():
    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor([1])}
    b = {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor([2])}
    out = average_state_dicts([a, b])
    assert torch.equal(out["w"], torch.tensor([2.0, 4.0]))
    assert torch.equal(out["n"], torch.tensor([2]))
