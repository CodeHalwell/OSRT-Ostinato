"""Canonical model preset for OSRT.

`OSRT_V7` is the committed v7 shape — see
`docs/specs/2026-08-11-v7-roadmap.md` §14 (shape), §16 (tokenizer) and §12.3
(mHC removed).

**No parameter counts appear in this file, or in any name.** The v6 lineage
carried four mutually inconsistent stale counts simultaneously; regenerate with
`scripts/compute_budget.py`, which instantiates the real model on a `meta`
device and is the only trusted source.

Assumption on record (roadmap §14.8), still open at gate G3a:

    v7 assumes the compute-optimal token requirement tracks ACTIVE parameters,
    not total. This follows from C = 6ND with N set by active FLOPs. It is
    FALSIFIED if loss-per-token degrades as total params rise at fixed active,
    or if the trunk run trails a smaller control at matched tokens.
"""

from __future__ import annotations

from osrt.config import OSRTConfig

OSRT_V7: dict = dict(
    dim=1536,
    heads=24,
    head_dim=64,
    num_kv_heads=8,             # GQA 24/8 + MLA-style compressed-latent KV cache
    # SmolLM2 base (49,152) + 32 OSRT special tokens = 49,184 real, padded to
    # a multiple of 128 for tensor cores. Chosen at G2 for single-digit number
    # tokenization: 100% context consistency and true place-value alignment,
    # where the v6 65,536 BPE made 1-3 digit numbers ATOMIC (roadmap §16).
    vocab_size=49280,
    real_vocab_size=49184,
    num_blocks=3,
    recursive_loops=6,          # 3 x 6 = 18 effective layers
    # Fine-grained re-grain (§14.3). Iso-active with 14 x h4224 top-2, but
    # satisfies the "more, smaller experts" requirement rather than reversing
    # it. h2112 = 33 x 64, so it survives the model.py tensor-core round-up.
    num_routed_experts=28,
    top_k_experts=4,            # 4/28 = 14.3% density
    expert_hidden=2112,
    shared_expert_hidden=2816,
    adapter_rank=256,           # real HRA capacity (NOT LoRA-style 16)
    adapter_alpha=256.0,        # match rank so scale = 1.0
    swiglu_clamp=10.0,          # DeepSeek-style SwiGLU stability clamp
    # Attention sink DROPPED in v6 and stays dropped. The manual sink path
    # materialises a (B,H,S,S) score matrix — measured OOM (>85GB) at batch 2,
    # seq 8192 — against 35.9GB through flash SDPA, with no demonstrated
    # benefit.
    attention_sink=False,
    # Grouped-GEMM MoE dispatch: removes the per-expert .nonzero(), the only
    # torch.compile graph break, so the model compiles fullgraph. NOTE: this
    # was validated at E=8/h3840; gate G7 re-benchmarks it at E=28/h2112 and
    # settles whether the expert path gets FP8/NVFP4 kernels on Blackwell.
    moe_grouped_gemm=True,
    aux_loop_loss_weight=0.05,  # on from step 1 — anti loop-collapse
    # Multi-Token Prediction. The head COUNT is deliberately NOT slimmed to 1:
    # roadmap §15 shows DeepSeek ran MTP-1 in production only because static
    # multi-token drafters degrade aggregate throughput under HIGH CONCURRENCY,
    # a constraint absent at the batch-1 decode this model targets. More draft
    # positions plus a lightweight sequential head is the right shape here.
    mtp_heads=2,
    mtp_loss_weight=0.3,
    router_aux_loss_coeff=0.10,
    router_z_loss_coeff=1e-3,
    # Quantile Balancing — REQUIRED at v7's granularity (roadmap §14.6).
    router_balance_mode="quantile",
    router_balance_bias_enabled=True,
    # sqrt(softplus) routing affinity: the balance bias steers TOP-K selection
    # on the non-negative affinity; gating weights renormalise the selected
    # balanced affinities. NOTE: §14.6 makes Quantile Balancing REQUIRED for

    router_affinity="sqrt_softplus",
    max_position_embeddings=4096,
)


def build_config(preset: dict = OSRT_V7, **overrides) -> OSRTConfig:
    """Build an OSRTConfig from a preset, with optional overrides."""
    return OSRTConfig(**{**preset, **overrides})


# ── G3a ladder ────────────────────────────────────────────────────────────
# The one gate that blocks the trunk run (roadmap §14.7): does the
# compute-optimal token requirement track ACTIVE parameters or TOTAL?
#
# Design: hold active fixed, sweep total by varying the expert COUNT at fixed
# top_k x expert_hidden. Every arm therefore does identical per-token compute
# and differs only in how much sparse capacity sits behind the router. If
# loss-per-token is flat across the sweep, §14.8's assumption holds and the
# committed 968M shape is safe; if it degrades with total, re-price before the
# trunk run.
#
# The arms bracket v7's own ratio (3.68x): 1.83x / 2.98x / 5.26x.
#
# CAVEAT: the tied embedding is 41% of active here against 28.8% in v7 — the
# vocab is fixed while dim shrinks, so it cannot be matched exactly at ladder
# scale. Read the arms against each OTHER, not against v7's absolute numbers.
_LADDER_BASE: dict = dict(
    dim=1024, heads=16, head_dim=64, num_kv_heads=4,
    vocab_size=49280, real_vocab_size=49184,
    num_blocks=3, recursive_loops=6,
    top_k_experts=4, expert_hidden=1056, shared_expert_hidden=1920,
    adapter_rank=192, adapter_alpha=192.0,
    swiglu_clamp=10.0, attention_sink=False, moe_grouped_gemm=True,
    aux_loop_loss_weight=0.05, mtp_heads=2, mtp_loss_weight=0.3,
    router_aux_loss_coeff=0.10, router_z_loss_coeff=1e-3,
    router_balance_bias_enabled=True, router_balance_mode="quantile",
    router_affinity="sqrt_softplus", max_position_embeddings=2048,
)

LADDER_ARMS: dict[str, dict] = {
    # name: experts. Active is constant; only total moves.
    "a": {**_LADDER_BASE, "num_routed_experts": 14},   # 225M total, 1.83x
    "b": {**_LADDER_BASE, "num_routed_experts": 28},   # 365M total, 2.98x
    "c": {**_LADDER_BASE, "num_routed_experts": 56},   # 646M total, 5.26x
}
