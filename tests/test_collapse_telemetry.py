"""Collapse telemetry for the recursive loops + MoE.

The MoE side already logs per-block/per-loop entropy, expert max/min, drop
rate, and prebias router stats (see _collect_moe_metrics). This adds the
missing recursive-loop signal — the per-effective-layer residual update
magnitude ||Δx|| / ||x|| — which reveals loops collapsing to no-ops, plus an
explicit dead-expert count.

The loop telemetry is gated on telemetry_enabled (like the MoE telemetry) so
it never runs on normal (compiled, fullgraph) steps — verified separately by
the graph-break probe.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_model import tiny_config  # noqa: E402

from osrt.model import OSRTForCausalLM  # noqa: E402
from osrt.train import _collect_moe_metrics  # noqa: E402


def _model_and_input(**over):
    torch.manual_seed(0)
    cfg = tiny_config(**over)
    model = OSRTForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    return cfg, model, x


def test_loop_update_norms_populated_when_telemetry_on():
    cfg, model, x = _model_and_input()
    model.set_moe_telemetry(True)
    model(input_ids=x)
    base = model.model
    n_eff = cfg.num_blocks * cfg.recursive_loops
    assert hasattr(base, "last_loop_update_norm")
    assert len(base.last_loop_update_norm) == n_eff
    assert all(v >= 0.0 for v in base.last_loop_update_norm)
    # Something must change the residual stream.
    assert any(v > 0.0 for v in base.last_loop_update_norm)


def test_loop_hidden_norms_populated():
    cfg, model, x = _model_and_input()
    model.set_moe_telemetry(True)
    model(input_ids=x)
    base = model.model
    assert len(base.last_loop_hidden_norm) == cfg.num_blocks * cfg.recursive_loops
    assert all(v > 0.0 for v in base.last_loop_hidden_norm)


def test_loop_telemetry_skipped_when_off():
    cfg, model, x = _model_and_input()
    model.set_moe_telemetry(False)
    base = model.model
    n_eff = cfg.num_blocks * cfg.recursive_loops
    base.last_loop_update_norm = [-1.0] * n_eff  # sentinel
    model(input_ids=x)
    # Telemetry off → the hook must not touch the list (keeps the fast path clean).
    assert all(v == -1.0 for v in base.last_loop_update_norm)


def test_collect_emits_loop_metrics():
    cfg, model, x = _model_and_input()
    model.set_moe_telemetry(True)
    model(input_ids=x)
    metrics, summary = _collect_moe_metrics(model)
    # per-effective-layer curves
    assert any(k.startswith("loop/update_norm_l") for k in metrics)
    # collapse-at-a-glance aggregates
    assert "loop/update_norm_mean" in metrics
    assert "loop/update_norm_min" in metrics
    assert "loop/update_norm_last" in metrics  # deepest loop; ~0 ⇒ no-op
    assert "loop_update_norm_min" in summary


def test_collect_emits_dead_expert_count():
    cfg, model, x = _model_and_input()
    model.set_moe_telemetry(True)
    model(input_ids=x)
    metrics, summary = _collect_moe_metrics(model)
    assert "moe/dead_experts_total" in metrics
    assert metrics["moe/dead_experts_total"] >= 0
    assert "dead_experts_total" in summary
