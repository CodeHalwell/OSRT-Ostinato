"""Regression tests for the training harness fixes of 2026-09-30.

Each test pins a behaviour that the review found silently wrong:
the applied LR schedule, the resume drift guard and schedule tags, the
continuous health checks, the final-checkpoint alias, and the Muon telemetry.
They exercise the real functions in `osrt.train` / `osrt.muon` on CPU.
"""

from __future__ import annotations

import math
import os
import shutil

import pytest
import torch

from osrt.config import OSRTConfig
from osrt.muon import HybridMuonAdamW, Muon
from osrt.train import (
    _alias_checkpoint,
    _average_moe_snapshots,
    _check_early_stop_criteria,
    _health_scope,
    _model_shape_metadata,
    _nonfinite_stop_reason,
    _optimizer_lr_tags,
    _reset_router_balance_accumulators,
    _set_param_group_lrs,
    _stamp_schedule_tags,
    _training_recipe_metadata,
    _update_health_streaks,
    assert_no_resume_drift,
    get_lr,
)
from osrt.train_config import PretrainConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_DIR = os.path.join(REPO, "tokenizer")


class _Opt:
    """Just enough optimizer surface for the schedule (param_groups)."""

    def __init__(self, tags: list[tuple[float, float]]) -> None:
        self.param_groups = [
            {"lr": 0.0, "_peak_lr": peak, "_min_lr": floor} for peak, floor in tags
        ]


# ── LR schedule: what is APPLIED must be what get_lr says ────────────────

def test_applied_schedule_is_flat_through_the_wsd_stable_phase():
    cfg = PretrainConfig()                      # lr_schedule == "wsd"
    opt = _Opt([(cfg.peak_lr, cfg.min_lr), (cfg.muon_lr, cfg.muon_min_lr)])
    decay_start = int(cfg.total_steps * (1 - cfg.wsd_decay_frac))
    for step in (cfg.warmup_steps, 2_000, 9_000, decay_start - 1):
        logged = _set_param_group_lrs(opt, step, cfg)
        assert logged == pytest.approx(cfg.peak_lr)
        assert opt.param_groups[0]["lr"] == pytest.approx(cfg.peak_lr)
        assert opt.param_groups[1]["lr"] == pytest.approx(cfg.muon_lr)


def test_applied_schedule_tracks_get_lr_per_group_in_decay_and_warmup():
    cfg = PretrainConfig()
    opt = _Opt([(cfg.peak_lr, cfg.min_lr), (cfg.muon_lr, cfg.muon_min_lr)])
    span_adamw = cfg.peak_lr - cfg.min_lr
    span_muon = cfg.muon_lr - cfg.muon_min_lr
    for step in (100, 300, 16_000, 16_650, 17_500, cfg.total_steps):
        logged = _set_param_group_lrs(opt, step, cfg)
        base = get_lr(step, cfg)
        assert logged == pytest.approx(base)
        assert opt.param_groups[0]["lr"] == pytest.approx(base)
        if step < cfg.warmup_steps:
            expect_muon = cfg.muon_lr * base / cfg.peak_lr
        else:
            frac = (base - cfg.min_lr) / span_adamw
            expect_muon = cfg.muon_min_lr + span_muon * frac
        assert opt.param_groups[1]["lr"] == pytest.approx(expect_muon)


def test_cosine_schedule_is_still_available_and_applied():
    cfg = PretrainConfig(lr_schedule="cosine")
    opt = _Opt([(cfg.peak_lr, cfg.min_lr)])
    lrs = [_set_param_group_lrs(opt, s, cfg) for s in (500, 5_000, 12_000, 17_999)]
    assert lrs == sorted(lrs, reverse=True)      # monotone decreasing
    assert lrs[1] < cfg.peak_lr * 0.95           # cosine has left the peak by 5k
    assert _set_param_group_lrs(opt, 5_000, cfg) == pytest.approx(get_lr(5_000, cfg))


# ── Schedule tags survive a checkpoint load only because we re-stamp them ──

def _hybrid(muon_lr: float) -> HybridMuonAdamW:
    w = torch.nn.Parameter(torch.randn(8, 8))
    b = torch.nn.Parameter(torch.zeros(8))
    muon = Muon([w], lr=muon_lr)
    adamw = torch.optim.AdamW([{"params": [b], "weight_decay": 0.0}], lr=6e-4)
    return HybridMuonAdamW(muon, adamw)


def test_stamp_schedule_tags_overrides_values_restored_by_load_state_dict():
    cfg_old = PretrainConfig(muon_lr=0.02, muon_min_lr=2e-3)
    cfg_new = PretrainConfig(muon_lr=0.015, muon_min_lr=1.5e-3)
    session1 = _hybrid(0.02)
    _stamp_schedule_tags(session1, cfg_old)
    state = session1.state_dict()

    session2 = _hybrid(0.015)
    _stamp_schedule_tags(session2, cfg_new)
    session2.load_state_dict(state)
    # torch restores every param-group key except `params`, so the OLD tag
    # is back — this is the bug the re-stamp fixes.
    assert session2.muon.param_groups[0]["_peak_lr"] == 0.02
    _stamp_schedule_tags(session2, cfg_new)
    assert session2.muon.param_groups[0]["_peak_lr"] == 0.015
    assert session2.muon.param_groups[0]["_min_lr"] == 1.5e-3
    assert session2.adamw.param_groups[0]["_peak_lr"] == cfg_new.peak_lr


# ── Resume drift guard ───────────────────────────────────────────────────

def _ckpt_for(cfg: PretrainConfig) -> dict:
    return {
        "training_recipe": _training_recipe_metadata(cfg, TOKENIZER_DIR),
        "optimizer_state_dict": {
            "muon": {"param_groups": [{"_peak_lr": cfg.muon_lr}]},
            "adamw": {"param_groups": [{"_peak_lr": cfg.peak_lr}]},
        },
    }


def test_recipe_metadata_carries_phase_plan_and_tokenizer_digest():
    meta = _training_recipe_metadata(PretrainConfig(), TOKENIZER_DIR)
    assert "phase_plan" in meta and "tokenizer_sha256" in meta
    assert len(meta["tokenizer_sha256"]) == 64
    for key in ("muon_lr", "muon_min_lr", "optimizer_name", "lr_schedule",
                "router_gumbel_tau_init", "router_gumbel_anneal_steps"):
        assert key in meta
    # operational knobs and the (rescalable) micro-batch shape are NOT recipe
    assert "dataloader_num_workers" not in meta
    assert "batch_size" not in meta


def test_drift_guard_catches_muon_lr_and_data_plan_changes(monkeypatch):
    monkeypatch.delenv("OSRT_ALLOW_RECIPE_DRIFT", raising=False)
    base = PretrainConfig()
    ckpt = _ckpt_for(base)
    # identical config: passes
    assert_no_resume_drift(ckpt, train_cfg=PretrainConfig(),
                           tokenizer_name=TOKENIZER_DIR)
    # Muon LR change: caught (via the recipe key)
    with pytest.raises(RuntimeError, match="muon_lr"):
        assert_no_resume_drift(ckpt, train_cfg=PretrainConfig(muon_lr=1e-3),
                               tokenizer_name=TOKENIZER_DIR)
    # data plan change (seq_len of one phase): caught via the phase plan
    changed = PretrainConfig()
    changed.phases["knowledge"]["seq_len"] = 1024
    with pytest.raises(RuntimeError, match="phase_plan"):
        assert_no_resume_drift(ckpt, train_cfg=changed, tokenizer_name=TOKENIZER_DIR)
    # micro-batch rescale at constant tokens/step: allowed
    rescaled = PretrainConfig()
    rescaled.scale_micro_batches(0.5)
    assert_no_resume_drift(ckpt, train_cfg=rescaled, tokenizer_name=TOKENIZER_DIR)


def test_drift_guard_reads_lr_tags_from_old_checkpoints(monkeypatch):
    """A checkpoint written before muon_lr joined the recipe keys still
    carries the LR inside the optimizer state — the guard reads it there."""
    monkeypatch.delenv("OSRT_ALLOW_RECIPE_DRIFT", raising=False)
    old = {
        "training_recipe": {"total_steps": 18_000, "peak_lr": 6e-4},
        "optimizer_state_dict": {
            "muon": {"param_groups": [{"_peak_lr": 0.02}]},
            "adamw": {"param_groups": [{"_peak_lr": 6e-4}]},
        },
    }
    assert _optimizer_lr_tags(old["optimizer_state_dict"]) == {
        "muon_lr": 0.02, "peak_lr": 6e-4}
    with pytest.raises(RuntimeError, match="OPTIMIZER LR"):
        assert_no_resume_drift(old, train_cfg=PretrainConfig(muon_lr=3e-3))
    assert_no_resume_drift(old, train_cfg=PretrainConfig(muon_lr=0.02))


def test_drift_guard_treats_a_missing_gate_flag_as_the_old_true(monkeypatch):
    """Checkpoints from before `router_bias_in_gates` existed were trained with
    the balance bias inside the gating weights (today's True); resuming one
    under the new default (False) must fail closed. (Codex review on PR #2.)"""
    monkeypatch.delenv("OSRT_ALLOW_RECIPE_DRIFT", raising=False)
    kw = dict(num_routed_experts=28, top_k_experts=4, router_balance_mode="quantile")
    current = OSRTConfig(**kw)
    assert current.router_bias_in_gates is False
    old_shape = {k: v for k, v in _model_shape_metadata(current).items()
                 if k != "router_bias_in_gates"}
    for old in ({"model_shape": old_shape}, {}):          # pre-flag, pre-metadata
        with pytest.raises(RuntimeError, match="router_bias_in_gates"):
            assert_no_resume_drift(old, model_config=current)
        # continuing the run as it was trained
        assert_no_resume_drift(old, model_config=OSRTConfig(
            **kw, router_bias_in_gates=True))
        # with neither the bias nor Gumbel in play the two settings agree
        assert_no_resume_drift(old, model_config=OSRTConfig(
            **kw, router_balance_bias_enabled=False))
    with pytest.raises(RuntimeError, match="router_bias_in_gates"):
        assert_no_resume_drift({}, model_config=OSRTConfig(
            **kw, router_balance_bias_enabled=False, router_gumbel_tau_init=1.0))
    # Gumbel is driven by the TRAINING config's schedule, not the model
    # config's field (which normally stays 0.0): with the bias off, the
    # default PretrainConfig schedule (0.5 -> 0 over 4,000 steps) still means
    # the legacy checkpoint's gates were noised. (Codex review on PR #2.)
    nobias = OSRTConfig(**kw, router_balance_bias_enabled=False)
    with pytest.raises(RuntimeError, match="router_bias_in_gates"):
        assert_no_resume_drift({"model_shape": old_shape}, model_config=nobias,
                               train_cfg=PretrainConfig())
    with pytest.raises(RuntimeError, match="router_bias_in_gates"):
        assert_no_resume_drift({"model_shape": old_shape, "step": 100},
                               model_config=nobias, train_cfg=PretrainConfig())
    # ... unless the schedule is silent, or had already annealed to zero at
    # the checkpoint's step, in which case the remaining steps agree
    assert_no_resume_drift({"model_shape": old_shape}, model_config=nobias,
                           train_cfg=PretrainConfig(router_gumbel_tau_init=0.0))
    assert_no_resume_drift({"model_shape": old_shape, "step": 10_000},
                           model_config=nobias, train_cfg=PretrainConfig())
    # a checkpoint stamped by this code carries the flag and is compared as-is
    new = {"model_shape": _model_shape_metadata(current)}
    assert_no_resume_drift(new, model_config=current)
    with pytest.raises(RuntimeError, match="router_bias_in_gates"):
        assert_no_resume_drift(new, model_config=OSRTConfig(
            **kw, router_bias_in_gates=True))
    # the deliberate-change escape hatch still applies
    monkeypatch.setenv("OSRT_ALLOW_RECIPE_DRIFT", "1")
    assert_no_resume_drift({"model_shape": old_shape}, model_config=current)


def test_drift_guard_env_override_downgrades_to_warning(monkeypatch, capsys):
    monkeypatch.setenv("OSRT_ALLOW_RECIPE_DRIFT", "1")
    ckpt = _ckpt_for(PretrainConfig())
    assert_no_resume_drift(ckpt, train_cfg=PretrainConfig(muon_lr=1e-3),
                           tokenizer_name=TOKENIZER_DIR)
    assert "WARNING (OSRT_ALLOW_RECIPE_DRIFT=1)" in capsys.readouterr().out


# ── Health checks run continuously, with patience ─────────────────────────

def test_health_scope_before_gate_after_gate_and_disabled():
    cfg = PretrainConfig()
    gate, warm = cfg.early_stop_check_step, cfg.warmup_steps
    assert _health_scope(0, cfg) is None
    assert _health_scope(warm - 1, cfg) is None
    assert _health_scope(warm, cfg) == "loop"
    assert _health_scope(gate - 50, cfg) == "loop"
    assert _health_scope(gate, cfg) == "full"
    assert _health_scope(gate + 50, cfg) == "full"
    assert _health_scope(cfg.total_steps - 1, cfg) == "full"
    off = PretrainConfig(continuous_health_checks=False)
    assert _health_scope(gate + 50, off) is None
    assert _health_scope(gate, off) == "full"      # the one-shot gate stays


def _healthy_summary(E: int = 28) -> dict:
    ln_e = math.log(E)
    return {
        "per_token_H": 0.4 * ln_e, "clean_per_token_H": 0.4 * ln_e,
        "raw_max": 0.6, "clean_raw_max": 0.6,
        "top_margin": 0.3, "clean_top_margin": 0.3,
        "marginal_H": 0.95 * ln_e, "clean_marginal_H": 0.95 * ln_e,
        "prebias_marginal_H": 0.9 * ln_e, "prebias_expert_min": 0.8 / E,
        "bias_abs_max": 0.1,
        "loop_update_norm_min": 0.05, "loop_hidden_norm_ratio": 1.4,
    }


def test_loop_scope_ignores_router_criteria_but_full_scope_does_not():
    cfg = PretrainConfig()
    mcfg = OSRTConfig(num_routed_experts=28, top_k_experts=4,
                      router_balance_mode="quantile")
    unsharp = _healthy_summary()
    unsharp["clean_per_token_H"] = math.log(28)   # router not sharpened yet
    assert _check_early_stop_criteria(1000, unsharp, cfg, mcfg, scope="loop") == []
    full = _check_early_stop_criteria(1000, unsharp, cfg, mcfg, scope="full")
    assert any("per_token_entropy" in f for f in full)
    collapsed = _healthy_summary()
    collapsed["loop_update_norm_min"] = 1e-5
    collapsed["loop_hidden_norm_ratio"] = 500.0
    loop_f = _check_early_stop_criteria(1000, collapsed, cfg, mcfg, scope="loop")
    assert len(loop_f) == 2
    with pytest.raises(ValueError):
        _check_early_stop_criteria(1000, collapsed, cfg, mcfg, scope="router")


def test_nonfinite_batches_retry_until_a_streak_or_total_cap():
    """A skipped batch retries the same step; only a streak or a run total
    ends the run. (Codex review on PR #2: a skipped step used to fall
    through to checkpointing and `step += 1`.)"""
    cfg = PretrainConfig()
    assert cfg.max_consecutive_nonfinite_steps == 5
    assert cfg.max_total_nonfinite_batches == 50
    assert _nonfinite_stop_reason(1, 1, cfg) is None
    assert _nonfinite_stop_reason(4, 30, cfg) is None
    assert "consecutive" in _nonfinite_stop_reason(5, 5, cfg)
    assert "over the run" in _nonfinite_stop_reason(1, 50, cfg)
    no_total = PretrainConfig(max_total_nonfinite_batches=0)
    assert _nonfinite_stop_reason(1, 10_000, no_total) is None
    assert "consecutive" in _nonfinite_stop_reason(5, 10_000, no_total)


def test_health_patience_counts_the_same_criterion_not_any_failure():
    """Three unrelated one-off failures must not add up to a stop; one metric
    failing `patience` checks in a row must. (Copilot review on PR #2.)"""
    streaks: dict[str, int] = {}
    _update_health_streaks(streaks, ["clean_raw_max_prob 0.10 < 0.20 (x)"])
    _update_health_streaks(streaks, ["loop_update_norm_min 1.0e-05 < 1.0e-03 (y)"])
    keys = _update_health_streaks(streaks, ["clean_top_margin 0.00 < 0.10 (z)"])
    assert keys == ["clean_top_margin"]
    assert streaks == {"clean_top_margin": 1}          # the blips did not stack
    for _ in range(3):
        _update_health_streaks(streaks, [
            "loop_update_norm_min 1.0e-05 < 1.0e-03 (y)",
            "clean_raw_max_prob 0.10 < 0.20 (x)" if _ == 1 else
            "prebias_expert_min 0.0010 < 0.0071 (w)",
        ])
    assert streaks["loop_update_norm_min"] == 3        # the persisting one
    assert max(v for k, v in streaks.items() if k != "loop_update_norm_min") == 1
    _update_health_streaks(streaks, [])
    assert streaks == {}


def test_averaged_snapshots_emit_hidden_norm_ratio_metric():
    snaps = [
        {"loop/hidden_norm_l0": 0.02, "loop/hidden_norm_l1": 1.0,
         "loop/hidden_norm_l2": 3.0, "loop/hidden_norm_l3": 2.0,
         "moe/dead_experts_total": 0},
        {"loop/hidden_norm_l0": 0.02, "loop/hidden_norm_l1": 1.0,
         "loop/hidden_norm_l2": 5.0, "loop/hidden_norm_l3": 2.0,
         "moe/dead_experts_total": 1},
    ]
    avg, summary = _average_moe_snapshots(snaps)
    assert avg["loop/hidden_norm_ratio"] == pytest.approx(4.0)   # max(1,4,2)/1
    assert summary["loop_hidden_norm_ratio"] == pytest.approx(4.0)
    assert avg["moe/dead_experts_total"] == pytest.approx(0.5)


# ── Checkpoint alias and accumulator reset ───────────────────────────────

def test_alias_checkpoint_exposes_final_under_step_name(tmp_path):
    src = tmp_path / "osrt_final.pt"
    src.write_bytes(b"weights")
    dst = tmp_path / "osrt_step_18000.pt"
    _alias_checkpoint(str(src), str(dst))
    assert dst.read_bytes() == b"weights"
    _alias_checkpoint(str(src), str(dst))       # idempotent
    assert dst.read_bytes() == b"weights"


def test_alias_checkpoint_copy_fallback_publishes_atomically(tmp_path, monkeypatch):
    """On a volume that rejects hard links the alias is COPIED; the step name
    must not exist until the copy is complete, or the sync daemon can upload a
    truncated alias and mark it pushed. (Codex review on PR #2.)"""
    src = tmp_path / "osrt_final.pt"
    src.write_bytes(b"weights")
    dst = tmp_path / "osrt_step_18000.pt"
    seen: dict = {}
    real_copy = shutil.copyfile

    def no_link(*a, **k):
        raise OSError("hard links not supported here")

    def spy_copy(s_, d_, *a, **k):
        seen["target"] = os.path.basename(d_)
        seen["final_visible_during_copy"] = dst.exists()
        return real_copy(s_, d_, *a, **k)

    monkeypatch.setattr(os, "link", no_link)
    monkeypatch.setattr(shutil, "copyfile", spy_copy)
    _alias_checkpoint(str(src), str(dst))
    assert dst.read_bytes() == b"weights"
    assert seen["final_visible_during_copy"] is False
    # the interim name matches neither the resume glob nor the sync regex
    assert seen["target"] == "osrt_step_18000.pt.tmp"
    assert not seen["target"].endswith(".pt")
    assert not (tmp_path / "osrt_step_18000.pt.tmp").exists()


def test_reset_router_balance_accumulators_zeroes_per_step_stats():
    from osrt.model import OSRTForCausalLM
    cfg = OSRTConfig(
        dim=64, heads=2, head_dim=32, vocab_size=128, real_vocab_size=128,
        num_blocks=1, recursive_loops=2, num_routed_experts=4, top_k_experts=2,
        expert_hidden=64, shared_expert_hidden=64, use_hra=False,
        router_balance_mode="quantile", max_position_embeddings=32,
    )
    model = OSRTForCausalLM(cfg).train()
    model(torch.randint(0, 128, (2, 8)))
    moe = model.model.blocks[0].moe
    assert moe.balance_total_accum.sum() > 0 and moe.qb_token_count.sum() > 0
    _reset_router_balance_accumulators(model)
    for name in ("balance_count_accum", "balance_total_accum", "qb_hist",
                 "qb_token_count"):
        assert getattr(moe, name).abs().sum() == 0


# ── Muon telemetry ───────────────────────────────────────────────────────

def test_muon_ortho_err_is_per_block_under_per_head_muon():
    """Per-head blocks are orthogonalised independently and are not mutually
    orthogonal, so the Gram of the reassembled matrix reads ~0.8-0.9 for a
    perfectly converged update. The reported residual must be the per-block
    one, or the RUNBOOK's 'rising ortho_err' reading is uninterpretable."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(96, 128))
    opt = Muon([{"params": [w], "head_dim": 16}], lr=1e-3, ns_steps=8,
               ns_stable_steps=2, update_rms=0.18)
    w.grad = torch.randn_like(w)
    opt.collect_ortho_error = True
    opt.step()
    assert opt.last_stats["muon/ortho_err"] < 0.05
    assert {"muon/update_rms_pre", "muon/update_rms_post"} <= set(opt.last_stats)


def test_muon_stats_present_every_step_and_ortho_only_when_asked():
    w = torch.nn.Parameter(torch.randn(48, 24))
    opt = Muon([w], lr=1e-3)
    w.grad = torch.randn_like(w)
    opt.step()
    assert set(opt.last_stats) == {"muon/update_rms_pre", "muon/update_rms_post"}
    opt.collect_ortho_error = True
    w.grad = torch.randn_like(w)
    opt.step()
    assert "muon/ortho_err" in opt.last_stats
    opt.collect_ortho_error = False
    w.grad = torch.randn_like(w)
    opt.step()
    assert "muon/ortho_err" not in opt.last_stats   # never a stale carry-over
