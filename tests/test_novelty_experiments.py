"""The instruments behind roadmap §18 — the three experiments OSRT can own.

E1 needs an HRA-off model that is genuinely adapter-free and still trains.
E2 needs the Muon telemetry to populate. E3 needs the loop-count recommender
to require all three signals, not one.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM
from osrt.muon import Muon
from osrt.presets import LADDER_ARMS, build_config

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from recommend_loop_count import recommend  # noqa: E402

_TINY = dict(dim=64, heads=4, head_dim=16, num_kv_heads=2, vocab_size=256,
             real_vocab_size=256, num_blocks=2, recursive_loops=3,
             num_routed_experts=4, top_k_experts=2, expert_hidden=32,
             shared_expert_hidden=32, max_position_embeddings=32)


# ── E1: adapters off ─────────────────────────────────────────────────────

def test_use_hra_false_has_zero_adapter_params_and_trains():
    m = OSRTForCausalLM(OSRTConfig(**_TINY, use_hra=False, adapter_rank=8))
    assert sum(p.numel() for n, p in m.named_parameters() if "adapter" in n) == 0
    assert m.model.adapter_scale == 0.0
    ids = torch.randint(0, 256, (2, 8))
    out = m(ids, labels=ids.clone())
    out.loss.backward()
    assert torch.isfinite(out.loss)
    m.eval()
    assert m.generate(ids[:, :4], max_new_tokens=2, temperature=0.0).shape == (2, 6)


def test_use_hra_true_is_unchanged():
    m = OSRTForCausalLM(OSRTConfig(**_TINY, use_hra=True, adapter_rank=8))
    n = sum(p.numel() for nm, p in m.named_parameters() if "adapter" in nm)
    # blocks*loops pairs, each an (a, b) pair of dim*rank
    assert n == 2 * 3 * 2 * (64 * 8)


def _flop_eq(arm):
    cfg = build_config(arm, expert_orthogonal_init=False)
    with torch.device("meta"):
        m = OSRTForCausalLM(cfg)
    tot = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for n, p in m.named_parameters()
              if "embedding" in n or "lm_head" in n)
    rt = sum(p.numel() for n, p in m.named_parameters() if ".experts." in n)
    mtp = sum(p.numel() for n, p in m.named_parameters() if "mtp" in n)
    act = tot - mtp - int(rt * (1 - cfg.top_k_experts / cfg.num_routed_experts))
    return tot, emb + (act - emb) * cfg.recursive_loops


def test_nohra_is_an_iso_compute_ablation():
    """Adapters are all-active params: dropping them drops FLOPs. The arm
    reinvests them into the shared expert so the ONLY difference is the
    mechanism, not the budget. Both totals and FLOP-eq must match `a`."""
    a, n = LADDER_ARMS["a"], LADDER_ARMS["nohra"]
    diff = {k for k in set(a) | set(n) if a.get(k) != n.get(k)}
    assert diff == {"use_hra", "shared_expert_hidden"}, diff
    (ta, fa), (tn, fn) = _flop_eq(a), _flop_eq(n)
    assert ta == tn, f"total drifted: {ta:,} vs {tn:,}"
    assert fa == fn, f"compute drifted: {fa:,} vs {fn:,}"


def test_g4_arm_holds_total_and_compute_within_two_percent():
    g4 = LADDER_ARMS["g4"]
    assert (g4["num_blocks"], g4["recursive_loops"]) == (4, 5)
    assert g4["expert_hidden"] % 64 == 0      # survives the tensor-core round-up
    (ta, fa), (tg, fg) = _flop_eq(LADDER_ARMS["a"]), _flop_eq(g4)
    assert abs(tg / ta - 1) < 0.02 and abs(fg / fa - 1) < 0.02


# ── E2: Muon telemetry ───────────────────────────────────────────────────

def test_muon_reports_update_rms_and_ortho_error():
    w = torch.nn.Parameter(torch.randn(48, 24))
    opt = Muon([w], lr=0.02, ns_steps=8, ns_stable_steps=2, update_rms=0.18)
    w.grad = torch.randn_like(w)
    opt.step()
    assert set(opt.last_stats) == {"muon/update_rms_pre", "muon/update_rms_post"}
    opt.collect_ortho_error = True
    w.grad = torch.randn_like(w)
    opt.step()
    assert "muon/ortho_err" in opt.last_stats
    # 8 fast + 2 stabilising NS iterations should leave the step near-orthogonal
    assert opt.last_stats["muon/ortho_err"] < 0.05


# ── E3: the recommender must require ALL THREE signals ───────────────────

def _report(kv_move, upd, ent):
    return {"config": {"num_loops": len(upd)},
            "blocks": [{"block": 0, "kv_move_size": kv_move,
                        "loop_update_norm": upd, "route_marginal_entropy": ent}]}


def test_recommender_trims_when_all_three_signals_are_idle():
    # loops 3 and 4 (0-indexed) idle on every signal -> keep 3
    rep = _report(kv_move=[0.3, 0.2, 0.01, 0.01],
                  upd=[1.0, 0.8, 0.5, 0.005, 0.004],
                  ent=[2.0, 1.9, 1.8, 1.79, 1.79])
    out = recommend(rep, kv_tol=0.08, upd_tol=0.02, ent_tol=0.05)
    assert out["recommended_loops"] == 3
    assert out["decode_latency_saving"] == 0.4


def test_recommender_refuses_when_only_cka_is_flat():
    """The v6 failure mode: latent stopped rotating, layer kept writing."""
    rep = _report(kv_move=[0.3, 0.2, 0.01, 0.01],
                  upd=[1.0, 0.8, 0.5, 0.45, 0.40],      # still writing
                  ent=[2.0, 1.9, 1.8, 1.79, 1.79])
    out = recommend(rep, kv_tol=0.08, upd_tol=0.02, ent_tol=0.05)
    assert out["recommended_loops"] == 5, \
        "CKA alone must not justify trimming"


def test_recommender_on_the_real_v6_probe_says_do_not_trim():
    path = REPO / "research" / "probe_cross_loop_kv_general.json"
    out = recommend(json.load(open(path)), kv_tol=0.08, upd_tol=0.02, ent_tol=0.05)
    assert out["recommended_loops"] == out["trained_loops"] == 6


def test_recommender_cli_runs():
    script = REPO / "scripts" / "recommend_loop_count.py"
    data = REPO / "research" / "probe_cross_loop_kv_general.json"
    r = subprocess.run([sys.executable, str(script), str(data)],
                       capture_output=True, text=True, check=True)
    assert "recommend running 6" in r.stdout
