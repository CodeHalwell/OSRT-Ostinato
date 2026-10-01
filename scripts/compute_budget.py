"""Canonical parameter / active-param budget for osrt configs.

Replaces the hand-derived tables in README.md / docs/ARCHITECTURE.md with numbers
generated from the real model on a meta device (no memory allocated).

IMPORTANT: by default this reports the canonical preset OSRT_V7, which
is the config that actually trains. Do NOT trust loose CLI overrides to
reproduce the preset — the CLI only exposes a handful of knobs and everything
else falls back to OSRTConfig defaults (e.g. num_kv_heads=None => MHA, not the
preset's GQA-8, and mtp_heads=0). A partial override silently builds a
different model. Prefer the default (whole preset) run.

Usage:
    PYTHONPATH=src python scripts/compute_budget.py        # canonical preset
    PYTHONPATH=src python scripts/compute_budget.py --solve 600e6   # hit a target
"""

from __future__ import annotations

import argparse

import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM
from osrt.presets import OSRT_V7

# Map a parameter name to a budget category. Ordered: first match wins.
_CATEGORIES = [
    # loop_embeddings before the tied embedding: "embedding" is a substring of
    # "loop_embeddings", and first match wins.
    ("loop_emb", lambda n: "loop_emb" in n),
    ("embedding", lambda n: "embedding" in n or "lm_head" in n),
    ("attention", lambda n: any(k in n for k in (
        "q_proj", "kv_down", "v_from_k", "out_proj",
        "norm_q", "norm_k", "norm_attn"))),
    ("shared_expert", lambda n: "shared_expert" in n),
    ("routed_experts", lambda n: ".experts." in n),
    ("router", lambda n: "router" in n or "moe_gate" in n),
    ("adapters", lambda n: "adapter" in n),
    ("mtp_heads", lambda n: "mtp" in n),
    ("norms_misc", lambda n: True),  # catch-all
]


def _categorize(name: str) -> str:
    for cat, pred in _CATEGORIES:
        if pred(name):
            return cat
    return "norms_misc"


def budget(cfg: OSRTConfig) -> dict[str, int]:
    """Return per-category physical param counts (meta device, no allocation).

    Also records the expert hidden size the model ACTUALLY built under
    `cats["_h_routed_actual"]` / `["_h_shared_actual"]` (model.py rounds
    `expert_hidden` up to a multiple of 64; the config keeps the requested
    value). `report` echoes the built value — CLAUDE.md tells readers to trust
    that echo, so it must not repeat the config."""
    cfg.expert_orthogonal_init = False  # QR can't run on meta tensors
    with torch.device("meta"):
        model = OSRTForCausalLM(cfg)
    cats: dict[str, int] = {}
    for name, p in model.named_parameters():
        cats[_categorize(name)] = cats.get(_categorize(name), 0) + p.numel()
    moe = model.model.blocks[0].moe
    cats["_h_routed_actual"] = moe.experts[0].w_gate.out_features
    cats["_h_shared_actual"] = moe.shared_expert.w_gate.out_features
    return cats


def _counts(cats: dict[str, int]) -> dict[str, int]:
    return {k: v for k, v in cats.items() if not k.startswith("_")}


def active_per_token(cats: dict[str, int], cfg: OSRTConfig) -> int:
    """Active params per token at INFERENCE: routed experts scaled by
    top_k / num_routed; MTP heads excluded (training-time only, dropped at
    deploy); everything else fully active (embedding counted full because the
    tied LM head touches the whole matrix)."""
    sparse_frac = cfg.top_k_experts / cfg.num_routed_experts
    active = 0
    for cat, n in _counts(cats).items():
        if cat == "routed_experts":
            active += int(n * sparse_frac)
        elif cat == "mtp_heads":
            continue  # training-only, not part of the inference forward
        else:
            active += n
    return active


def report(cfg: OSRTConfig) -> tuple[int, int]:
    cats = budget(cfg)
    counts = _counts(cats)
    total = sum(counts.values())
    active = active_per_token(cats, cfg)
    h_routed, h_shared = cats["_h_routed_actual"], cats["_h_shared_actual"]
    rounded = ""
    if h_routed != cfg.expert_hidden or h_shared != cfg.shared_expert_hidden:
        rounded = (f" (requested h_routed={cfg.expert_hidden} "
                   f"h_shared={cfg.shared_expert_hidden}; model.py rounds up "
                   "to a multiple of 64)")
    print(
        f"cfg: dim={cfg.dim} vocab={cfg.vocab_size} blocks={cfg.num_blocks} "
        f"loops={cfg.recursive_loops} kv_heads={cfg.num_kv_heads} "
        f"experts={cfg.num_routed_experts} top_k={cfg.top_k_experts} "
        f"h_routed={h_routed} h_shared={h_shared} "
        f"rank={cfg.adapter_rank} mtp={cfg.mtp_heads}"
        f"{' hra=off' if not cfg.use_hra else ''}{rounded}"
    )
    print("-" * 64)
    for cat in ("embedding", "attention", "shared_expert",
                "routed_experts", "router", "adapters", "mtp_heads",
                "loop_emb", "norms_misc"):
        if cat in counts:
            print(f"  {cat:<16} {counts[cat]:>14,}")
    print("-" * 64)
    print(f"  {'TOTAL PHYSICAL':<16} {total:>14,}  (~{total/1e6:.0f}M)")
    print(f"  {'ACTIVE / TOKEN':<16} {active:>14,}  (~{active/1e6:.0f}M, "
          f"{100*active/total:.1f}% of physical, inference — excl. MTP)")
    return total, active


def solve_expert_hidden(target: int, base: OSRTConfig, step: int = 128) -> int:
    """Smallest expert_hidden (multiple of `step`) whose total >= target."""
    h = step
    while True:
        cfg = OSRTConfig(**{**base.to_dict(), "expert_hidden": h,
                                "expert_orthogonal_init": False})
        if sum(_counts(budget(cfg)).values()) >= target:
            return h
        h += step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", type=float, default=None,
                    help="target physical params; widens expert_hidden to hit it")
    ap.add_argument(
        "--arm", default=None,
        help="report a G3a ladder arm from osrt.presets.LADDER_ARMS instead of "
             "the trunk preset (a, b, c, dense, hra, g4)",
    )
    ap.add_argument(
        "--override", nargs="*", default=[],
        help="key=value overrides on top of the canonical preset, e.g. "
             "--override expert_hidden=4096 num_routed_experts=12. Use with "
             "care: the preset is the config that actually trains.",
    )
    args = ap.parse_args()

    # Start from the REAL canonical preset, not loose defaults. This is the
    # config that trains; reporting anything else silently misleads (the
    # old CLI fell back to MHA + no-MTP defaults and over-reported by ~6M).
    preset = dict(OSRT_V7)
    if args.arm is not None:
        from osrt.presets import LADDER_ARMS
        if args.arm not in LADDER_ARMS:
            raise SystemExit(f"unknown arm {args.arm!r}; one of {list(LADDER_ARMS)}")
        preset = dict(LADDER_ARMS[args.arm])
        print(f"ladder arm {args.arm!r}")
    for kv in args.override:
        k, _, v = kv.partition("=")
        # int-ify where possible, else leave as string/bool
        if v.lower() in ("true", "false"):
            preset[k] = v.lower() == "true"
        else:
            try:
                preset[k] = int(v)
            except ValueError:
                preset[k] = v
    base = OSRTConfig(**preset)

    if args.solve:
        h = solve_expert_hidden(int(args.solve), base)
        print(f"=> expert_hidden={h} hits target {args.solve:.0f}\n")
        base = OSRTConfig(**{**base.to_dict(), "expert_hidden": h})
    report(base)


if __name__ == "__main__":
    main()
