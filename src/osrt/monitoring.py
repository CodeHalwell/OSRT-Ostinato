"""Recursion + MoE health monitoring — the v5 blind spot, instrumented.

v5 lost months to two failure modes discovered late (see docs/LEARNINGS.md):
  * MoE collapse  — routing funnels to ~2 of N experts; the rest go dead.
  * Loop collapse — one recursive loop does ~all the work; depth is wasted.

Both are silent: training loss still falls. This module surfaces them as
explicit per-(block, loop) metrics and boolean alarms, designed to be logged
at EVERY training stage so you can watch for them from step 1.

Two entry points:
  * moe_health(model)               — zero extra compute; reads buffers the
                                       model already populated on its last forward.
  * loop_depth_probe(model, x, y)   — one extra forward; per-loop CE ("Test 3").
  * summarize(...)                  — both, flattened for wandb + a human summary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# ── thresholds (tune per scale; defaults are deliberately conservative) ──
DEAD_EXPERT_FRAC = 0.5      # expert is "dead" below 0.5x its uniform share
MIN_LOAD_ENTROPY = 0.55     # normalized load entropy below this = collapsing
MAX_EXPERT_SHARE = 0.45     # one expert taking >45% of tokens = collapsing
MAX_LOOP_GAIN_SHARE = 0.70  # one loop doing >70% of CE reduction = loop collapse
MIN_LATE_LOOP_GAIN = 0.01   # a late loop adding <1% relative = wasted depth


def _entropy(p: torch.Tensor) -> float:
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum())


def _base(model):
    """Return the inner OSRTModel regardless of wrapper."""
    return model.model if hasattr(model, "model") else model


@dataclass
class MoEHealth:
    # per [block][loop]
    load_entropy: list[list[float]] = field(default_factory=list)   # normalized 0..1
    max_share: list[list[float]] = field(default_factory=list)
    dead_experts: list[list[int]] = field(default_factory=list)
    drop_rate: list[list[float]] = field(default_factory=list)
    num_experts: int = 0
    collapsing: bool = False
    summary: str = ""


def moe_health(model) -> MoEHealth:
    """Per-(block, loop) expert-load health from the model's last forward.

    Uses the CLEAN routing fractions (deployed behavior, no Gumbel/bias noise)
    when available — that's the distribution that actually matters at inference.
    """
    base = _base(model)
    h = MoEHealth()
    worst_msgs: list[str] = []
    for b, block in enumerate(base.blocks):
        moe = block.moe
        h.num_experts = moe.num_routed
        uniform = 1.0 / moe.num_routed
        fracs = (getattr(moe, "last_clean_expert_fraction", None)
                 or moe.last_expert_fraction)
        drops = getattr(moe, "last_drop_rate", [0.0] * moe.num_loops)
        be, bm, bd, bdr = [], [], [], []
        for loop in range(moe.num_loops):
            load = torch.tensor(fracs[loop], dtype=torch.float32)
            if load.sum() > 0:
                load = load / load.sum()
            ent = _entropy(load) / math.log(moe.num_routed)
            mx = float(load.max())
            dead = int((load < DEAD_EXPERT_FRAC * uniform).sum())
            be.append(ent)
            bm.append(mx)
            bd.append(dead)
            bdr.append(float(drops[loop]))
            if (ent < MIN_LOAD_ENTROPY or mx > MAX_EXPERT_SHARE
                    or dead >= moe.num_routed // 2):
                h.collapsing = True
                worst_msgs.append(
                    f"block{b}/loop{loop}: entropy={ent:.2f} max={mx:.2f} dead={dead}"
                )
        h.load_entropy.append(be)
        h.max_share.append(bm)
        h.dead_experts.append(bd)
        h.drop_rate.append(bdr)
    h.summary = (
        "MoE COLLAPSE RISK — " + "; ".join(worst_msgs[:3]) if h.collapsing
        else "MoE balanced across experts and loops"
    )
    return h


@dataclass
class LoopDepth:
    per_loop_ce: list[float] = field(default_factory=list)  # CE at end of each loop
    marginal_gain: list[float] = field(default_factory=list)  # CE reduction per loop
    # fraction of total reduction
    gain_share: list[float] = field(default_factory=list)
    total_reduction: float = 0.0
    collapsing: bool = False
    summary: str = ""


@torch.no_grad()
def loop_depth_probe(model, input_ids, labels) -> LoopDepth:
    """Per-loop CE via the tied LM head ("Test 3"). Healthy recursion reduces
    CE monotonically and spreads the reduction across loops; collapse shows up
    as one loop doing ~all the work (or late loops adding nothing)."""
    was_training = model.training
    model.train()  # capture_aux requires training mode + aux_loop_loss_weight>0
    model(input_ids, labels=labels)
    if not was_training:
        model.eval()

    # Guard against an unpopulated attribute (aux loop disabled, or a model
    # that never set it) — iterating None would raise.
    per_loop_losses = getattr(model, "last_per_loop_aux_losses", None) or []
    per_loop = [float(x) for x in per_loop_losses]  # loops 0..n-2
    final = float(model.last_task_loss)
    d = LoopDepth(per_loop_ce=per_loop + [final])
    if len(d.per_loop_ce) < 2:
        d.summary = "depth probe unavailable (need aux_loop_loss_weight>0)"
        return d

    ce = d.per_loop_ce
    d.marginal_gain = [ce[i] - ce[i + 1] for i in range(len(ce) - 1)]
    d.total_reduction = ce[0] - ce[-1]
    if d.total_reduction > 1e-6:
        d.gain_share = [g / d.total_reduction for g in d.marginal_gain]
        max_share = max(d.gain_share)
        late_wasted = any(s < MIN_LATE_LOOP_GAIN
                          for s in d.gain_share[len(d.gain_share) // 2:])
        if max_share > MAX_LOOP_GAIN_SHARE or late_wasted:
            d.collapsing = True
    else:
        d.collapsing = True  # no reduction across depth at all

    d.summary = (
        f"LOOP DEPTH UNDERUSED — gain shares {[round(s, 2) for s in d.gain_share]}"
        if d.collapsing
        else f"depth utilized — CE {ce[0]:.2f}->{ce[-1]:.2f} "
             f"spread across {len(ce)} loops"
    )
    return d


def summarize(model, input_ids=None, labels=None) -> tuple[dict, str]:
    """Flatten MoE + loop-depth metrics for wandb/tensorboard, plus a human
    one-liner. Call after a training step; pass (input_ids, labels) to also run
    the depth probe."""
    flat: dict[str, float] = {}
    h = moe_health(model)
    for b in range(len(h.load_entropy)):
        for loop in range(len(h.load_entropy[b])):
            flat[f"moe/b{b}/loop{loop}/load_entropy"] = h.load_entropy[b][loop]
            flat[f"moe/b{b}/loop{loop}/max_share"] = h.max_share[b][loop]
            flat[f"moe/b{b}/loop{loop}/dead_experts"] = h.dead_experts[b][loop]
            flat[f"moe/b{b}/loop{loop}/drop_rate"] = h.drop_rate[b][loop]
    flat["moe/collapsing"] = float(h.collapsing)
    lines = [h.summary]

    if input_ids is not None and labels is not None:
        d = loop_depth_probe(model, input_ids, labels)
        for i, ce in enumerate(d.per_loop_ce):
            flat[f"loop_depth/ce/loop{i}"] = ce
        for i, s in enumerate(d.gain_share):
            flat[f"loop_depth/gain_share/loop{i}"] = s
        flat["loop_depth/total_reduction"] = d.total_reduction
        flat["loop_depth/collapsing"] = float(d.collapsing)
        lines.append(d.summary)

    return flat, " | ".join(lines)
