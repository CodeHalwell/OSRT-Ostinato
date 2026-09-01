"""v7 launch-readiness: the invariants a run must satisfy before compute is spent.

Each test is a thing that, when wrong, produces a run that LOOKS normal — a
truncated budget, a silently swapped tokenizer, a resume that splices two
experiments. That is the expensive failure class, so these fail closed.
"""
from __future__ import annotations

import math

import pytest
import torch
from transformers import AutoTokenizer

from osrt.model import OSRTForCausalLM
from osrt.presets import OSRT_V7, build_v7_config
from osrt.tokenizer_contract import validate_tokenizer_contract
from osrt.train import (
    _set_param_group_lrs,
    assert_no_resume_drift,
    get_lr,
    save_checkpoint,
)
from osrt.train_config import PretrainConfig, V7SanityConfig

COMMITTED_PHYSICAL = 968_468_355     # roadmap §16.5, compute_budget.py


# ── the shape ────────────────────────────────────────────────────────────

def test_v7_preset_is_the_committed_shape():
    cfg = build_v7_config()
    shape = (cfg.num_routed_experts, cfg.top_k_experts, cfg.expert_hidden)
    assert shape == (28, 4, 2112)
    assert (cfg.vocab_size, cfg.real_vocab_size) == (49_280, 49_184)
    assert cfg.situ_glu is True
    assert cfg.router_balance_mode == "quantile"
    assert cfg.router_seq_balance_loss_coeff == 1e-4
    assert cfg.use_hra is False                # E1: off for pretraining
    assert cfg.shared_expert_hidden == 3840    # HRA params reinvested exactly
    assert "use_mhc" not in OSRT_V7          # removed, not merely disabled
    with torch.device("meta"):
        n = sum(p.numel() for p in OSRTForCausalLM(cfg).parameters())
    assert n == COMMITTED_PHYSICAL


# ── the training recipe is internally consistent ─────────────────────────

def test_phases_partition_total_steps_exactly():
    """v6 shipped total_steps=3,500 beside a phase table ending at 300,000."""
    cfg = PretrainConfig()
    ends = [p["end"] for p in cfg.phases.values()]
    starts = [p["start"] for p in cfg.phases.values()]
    assert starts[0] == 0 and ends[-1] == cfg.total_steps
    assert starts[1:] == ends[:-1], "phases must be contiguous"


def test_default_budget_is_about_one_chinchilla_on_active():
    cfg = PretrainConfig()
    tokens = cfg.total_tokens()
    assert 5.0e9 < tokens < 5.6e9, f"budget drifted: {tokens/1e9:.2f}B"


def test_wsd_decay_is_aligned_with_the_anneal_phase():
    """LR decay and high-quality data both belong to the branch (§7.5.2)."""
    cfg = PretrainConfig()
    decay_start = int(cfg.total_steps * (1 - cfg.wsd_decay_frac))
    assert decay_start == cfg.phases["anneal"]["start"]


def test_changing_total_steps_rederives_phases():
    cfg = PretrainConfig()
    cfg.total_steps = 1_000
    assert [p["end"] for p in cfg.phases.values()][-1] == 1_000
    assert sum(p["end"] - p["start"] for p in cfg.phases.values()) == 1_000


def test_config_instances_do_not_share_phase_state():
    a, b = PretrainConfig(), PretrainConfig()
    a.total_steps = 500
    assert b.phases["anneal"]["end"] == PretrainConfig().total_steps


def test_validate_rejects_impossible_recipes():
    c = PretrainConfig()
    c.total_steps = 200                      # sanity-style, warmup still 400
    with pytest.raises(ValueError, match="warmup_steps"):
        c.validate()
    d = PretrainConfig()
    d.phases["anneal"]["frac"] = 0.5         # fractions no longer sum to 1
    with pytest.raises(ValueError, match="sum to 1.0"):
        d._resolve_phases()


def test_sanity_config_validates_and_is_capped():
    v = V7SanityConfig()
    v.validate()
    assert v.total_steps <= 100
    assert v.save_final_checkpoint is False


# ── the LR schedule ──────────────────────────────────────────────────────

def test_wsd_schedule_boundaries():
    cfg = PretrainConfig()
    cfg.total_steps, cfg.warmup_steps = 1_000, 100
    decay_start = int(cfg.total_steps * (1 - cfg.wsd_decay_frac))
    assert get_lr(0, cfg) == 0.0
    assert math.isclose(get_lr(cfg.warmup_steps, cfg), cfg.peak_lr)
    assert get_lr(decay_start - 1, cfg) == cfg.peak_lr
    assert get_lr(decay_start, cfg) == cfg.peak_lr
    decay_len = cfg.total_steps - decay_start
    assert math.isclose(get_lr(decay_start + decay_len // 2, cfg),
                        (cfg.peak_lr + cfg.min_lr) / 2, rel_tol=1e-6)
    assert math.isclose(get_lr(cfg.total_steps, cfg), cfg.min_lr)


def test_schedule_respects_per_group_peak_and_floor():
    cfg = PretrainConfig()

    class _Opt:
        param_groups = [{"lr": 0.0, "_peak_lr": 2.0, "_min_lr": 0.2},
                        {"lr": 0.0, "_peak_lr": 1.0, "_min_lr": 0.1}]

    opt = _Opt()
    _set_param_group_lrs(opt, cfg.total_steps, cfg)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.2)
    assert opt.param_groups[1]["lr"] == pytest.approx(0.1)


# ── the tokenizer ────────────────────────────────────────────────────────

def test_shipped_tokenizer_satisfies_the_contract():
    validate_tokenizer_contract(AutoTokenizer.from_pretrained("tokenizer"))


def test_wrong_vocab_is_refused():
    tok = AutoTokenizer.from_pretrained("tokenizer")
    with pytest.raises(ValueError, match="vocab size is 49184, expected 65536"):
        validate_tokenizer_contract(tok, expected_vocab_size=65_536)


# ── resume fails closed ──────────────────────────────────────────────────

def _ckpt(tmp_path, model_cfg, train_cfg):
    m = torch.nn.Linear(2, 2)
    o = torch.optim.SGD(m.parameters(), lr=0.1)
    path = str(tmp_path / "osrt_step_7.pt")
    save_checkpoint(m, o, 7, path, model_config=model_cfg, train_cfg=train_cfg)
    return torch.load(path, map_location="cpu", weights_only=False), path


def test_resume_accepts_an_exact_match(tmp_path):
    mc, tc = build_v7_config(), PretrainConfig()
    ck, path = _ckpt(tmp_path, mc, tc)
    assert_no_resume_drift(ck, model_config=mc, train_cfg=tc, path=path)


def test_resume_rejects_model_shape_drift(tmp_path):
    ck, path = _ckpt(tmp_path, build_v7_config(), PretrainConfig())
    other = build_v7_config(vocab_size=65_536, real_vocab_size=65_536)
    with pytest.raises(RuntimeError, match="MODEL SHAPE drift"):
        assert_no_resume_drift(ck, model_config=other, path=path)


def test_resume_rejects_same_shape_recipe_drift(tmp_path):
    """Same weights would load fine — that is exactly why this must fail."""
    mc = build_v7_config()
    ck, path = _ckpt(tmp_path, mc, PretrainConfig())
    tc = PretrainConfig()
    tc.peak_lr = 1e-9
    with pytest.raises(RuntimeError, match="TRAINING RECIPE drift"):
        assert_no_resume_drift(ck, model_config=mc, train_cfg=tc, path=path)


# ── early-stop thresholds scale with the expert count ────────────────────

def _thresholds(E, K):
    """Mirror _check_early_stop_criteria's resolution exactly."""
    cfg = PretrainConfig()
    ln_e = math.log(E)
    return {
        "target_pte": ln_e - cfg.per_token_entropy_drop_frac * ln_e,
        "min_raw_max": cfg.raw_max_prob_frac_of_topk / K,
        "min_margin": cfg.top_margin_frac_of_topk / K,
        "min_marginal": cfg.marginal_entropy_frac * ln_e,
        "min_prebias_marginal": cfg.prebias_marginal_entropy_frac * ln_e,
        "min_prebias_expert": cfg.prebias_expert_fraction_of_uniform / E,
    }


def test_relative_thresholds_reproduce_the_v6_absolutes_at_e8_top2():
    """The fractions were chosen so v6's tuned numbers fall out exactly."""
    t = _thresholds(E=8, K=2)
    assert t["target_pte"] == pytest.approx(2.079 - 0.55, abs=0.01)
    assert t["min_raw_max"] == pytest.approx(0.30)
    assert t["min_margin"] == pytest.approx(0.10)
    assert t["min_marginal"] == pytest.approx(1.80, abs=0.01)
    assert t["min_prebias_marginal"] == pytest.approx(1.55, abs=0.01)
    assert t["min_prebias_expert"] == pytest.approx(0.01)


def test_relative_thresholds_scale_sanely_to_e28_top4():
    """At v7's shape the old absolutes were wrong in both directions."""
    t = _thresholds(E=28, K=4)
    # raw_max: 0.30 would have been 8.4x uniform and killed a healthy top-4
    # router. 0.15 is 0.6 of the 1/4 a sharpened top-4 pick sits near.
    assert t["min_raw_max"] == pytest.approx(0.15)
    # marginal entropy: 1.80 was 87% of ln 8; at E=28 that same 87% is 2.89,
    # so a router sitting at the OLD 1.80 (54% of ln 28) is now caught.
    assert t["min_marginal"] == pytest.approx(0.866 * math.log(28), abs=0.01)
    assert 1.80 < t["min_marginal"]
    # entropy must drop the same FRACTION of its ceiling, not the same nats
    assert t["target_pte"] == pytest.approx(math.log(28) * (1 - 0.265), abs=0.01)
    assert t["min_prebias_expert"] == pytest.approx(0.08 / 28)


def _healthy_summary(**over):
    """A summary every non-loop criterion passes on; loop keys as given."""
    base = {
        "per_token_H": 1.0, "raw_max": 0.3, "top_margin": 0.05,
        "marginal_H": 3.3, "prebias_marginal_H": 3.3,
        "prebias_expert_min": 0.03, "bias_abs_max": 0.0,
        "loop_update_norm_min": 0.5, "loop_hidden_norm_ratio": 2.0,
    }
    base.update(over)
    return base


def test_loop_guards_are_wired_into_the_early_stop_check():
    from osrt.presets import build_v7_config
    from osrt.train import _check_early_stop_criteria
    from osrt.train_config import PretrainConfig

    cfg, mcfg = PretrainConfig(), build_v7_config()
    ok = _check_early_stop_criteria(1000, _healthy_summary(), cfg, mcfg)
    assert not any(f.startswith("loop_") for f in ok)

    dead = _check_early_stop_criteria(
        1000, _healthy_summary(loop_update_norm_min=1e-5), cfg, mcfg)
    assert any(f.startswith("loop_update_norm_min") for f in dead)

    blown = _check_early_stop_criteria(
        1000, _healthy_summary(loop_hidden_norm_ratio=600.0), cfg, mcfg)
    assert any(f.startswith("loop_hidden_norm_ratio") for f in blown)


def test_hidden_norm_ratio_measures_within_loop_growth_not_embedding():
    """Shapes lifted from the 2026-09-02 ladder W&B norms (18 effective layers,
    norm_loop resets to unit RMS = 3.6e3 at each loop boundary)."""
    from osrt.train import _hidden_norm_ratio

    def avg(norms):
        return {f"loop/hidden_norm_l{i}": v for i, v in enumerate(norms)}

    nohra = [82.8, 7.57e5, 1.06e6] + [3.59e3, 6.5e5, 9.6e5] * 5
    a = [75.0, 7.05e5, 3.67e7] + [3.62e3, 8.0e5, 3.5e7] * 5
    g4 = [73.2, 9.89e5, 4.82e7, 2.72e9] + [3.6e3, 1.2e6, 5e7, 3e9] * 4
    assert 1.0 < _hidden_norm_ratio(avg(nohra)) < 2.0     # residual composition intact
    assert _hidden_norm_ratio(avg(a)) > 50                # blocks overwrite the stream
    assert _hidden_norm_ratio(avg(g4)) > 1000
    # deepest/embedding would have read ~1e4 on ALL of them, nohra included
    assert _hidden_norm_ratio(avg([82.8, 7.57e5])) == 1.0  # too few layers: neutral

