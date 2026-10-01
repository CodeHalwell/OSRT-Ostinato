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
from osrt.tokenizer_contract import OSTINATO_SPECIAL_TOKEN_IDS as _TOK

# Structural token ids of the v7 tokenizer (tokenizer/, SmolLM2 base + 32 OSRT
# specials at 49,152+). OSRTConfig's own defaults are the v6 ids (0..13), which
# in THIS tokenizer are SmolLM2's <|endoftext|>, <|im_start|>, <gh_stars>,
# <jupyter_start>...; a model built from the preset alone used to inherit them
# (generate() then stopped on <|im_end|>). The FIM ids are the three slots
# between <|unknown|> and <|think|> in scripts/build_tokenizer_v7.py.
assert _TOK["<|think|>"] == _TOK["<|unknown|>"] + 4, "tokenizer contract moved"
OSRT_V7_TOKEN_IDS: dict = dict(
    bos_token_id=_TOK["<|begin_of_text|>"],
    eos_token_id=_TOK["<|end_of_text|>"],
    pad_token_id=_TOK["<|padding|>"],
    unk_token_id=_TOK["<|unknown|>"],
    fim_prefix_id=_TOK["<|unknown|>"] + 1,
    fim_middle_id=_TOK["<|unknown|>"] + 2,
    fim_suffix_id=_TOK["<|unknown|>"] + 3,
    think_open_id=_TOK["<|think|>"],
    think_close_id=_TOK["<|/think|>"],
    answer_open_id=_TOK["<|answer|>"],
    answer_close_id=_TOK["<|/answer|>"],
    user_token_id=_TOK["<|user|>"],
    assistant_token_id=_TOK["<|assistant|>"],
    system_token_id=_TOK["<|system|>"],
)

OSRT_V7: dict = dict(
    **OSRT_V7_TOKEN_IDS,
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
    # E1 (roadmap §18.1, 2026-09-02): HRA is OFF for pretraining. The adapter
    # on the raw residual was un-normalised feedback that broke residual
    # composition inside every loop; the no-HRA ladder arm led by 1.5 nats.
    # Its 14,155,776 all-active params (18 x 2 x 1536 x 256) go into the
    # shared expert instead: 3 blocks x 3 x 1536 x (3840 - 2816) is exactly
    # the same count, so total, active and per-token FLOPs are unchanged.
    # HRA is attached to the frozen base as the post-training adapter; the
    # rank/alpha below size that adapter (now on the normalised input).
    use_hra=False,
    shared_expert_hidden=3840,
    adapter_rank=256,           # post-training adapter capacity (NOT LoRA-style 16)
    adapter_alpha=256.0,        # match rank so scale = 1.0
    # SiTU-GLU (graduated, §14.1): param-free smooth cap that REPLACES
    # SwiGLU + the hard clamp — no parameter change, so the budget numbers
    # are untouched and a checkpoint loads under either setting. The clamp
    # value is kept as the fallback for the G3 ladder's SiTU-vs-clamp A/B.
    situ_glu=True,
    swiglu_clamp=10.0,          # DeepSeek-style clamp; inert while situ_glu=True
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
    # Per-sequence balance loss (§14.1 router line; V4 runs exactly this).
    # OSRTConfig ships 0.0 so v6-reproduction runs are untouched.
    router_seq_balance_loss_coeff=1e-4,
    # Quantile Balancing — REQUIRED at v7's granularity (roadmap §14.6): at
    # E=28 one dead expert costs 3.6% of block capacity, and the ±γ heuristic
    # controller was tuned at E=8.
    router_balance_mode="quantile",
    router_balance_bias_enabled=True,
    # sqrt(softplus) routing affinity: the balance bias steers TOP-K selection
    # on the non-negative affinity; gating weights renormalise the selected
    # balanced affinities.
    router_affinity="sqrt_softplus",
    # 8192, not 4096: the anneal phase trains at seq 8192 (train_config) and
    # RoPE tables are built to this length, so 4096 would index out of range
    # in the last 15% of the run. Deployment context is still 4K (§8, "RoPE @
    # 4096 ctx, 8K-capable"); this is the capability ceiling, not the target.
    max_position_embeddings=8192,
)


def build_v7_config(**overrides) -> OSRTConfig:
    """The committed v7 shape. Alias kept so both repos read the same."""
    return build_config(OSRT_V7, **overrides)


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
# committed shape is safe; if it degrades with total, re-price before the
# trunk run.
#
# The base is the TRUNK RECIPE at a smaller dim: everything not listed here
# (SiTU-GLU, seq-balance, Quantile Balancing, sqrt-softplus affinity, MTP,
# HRA off with the shared-expert reinvestment, the token ids) is inherited
# from OSRT_V7, so an arm explains the trunk rather than a different model.
# Until 2026-09-30 the base was a separate dict that had drifted (HRA on,
# SiTU off, no seq-balance) and asked for expert_hidden=1056, which
# model.py rounds up to 1088 — 1088 is now stated. Counts per arm come from
# `scripts/compute_budget.py --arm <name>`, not from comments here.
#
# CAVEAT: the tied embedding is a larger share of active here than in v7 —
# the vocab is fixed while dim shrinks, so it cannot be matched exactly at
# ladder scale. Read the arms against each OTHER, not against v7's numbers.
_LADDER_BASE: dict = {
    **OSRT_V7,
    "dim": 1024, "heads": 16, "head_dim": 64, "num_kv_heads": 4,
    "top_k_experts": 4, "expert_hidden": 1088,
    # HRA-off reinvestment at this dim: 1920 + the adapters' all-active
    # params (18 x 2 x 1024 x 192 = 3 x 3 x 1024 x 768) -> 2688.
    "shared_expert_hidden": 2688,
    "adapter_rank": 192, "adapter_alpha": 192.0,
    "max_position_embeddings": 2048,
}

LADDER_ARMS: dict[str, dict] = {
    # name: experts. Active is constant; only total moves.
    "a": {**_LADDER_BASE, "num_routed_experts": 14},
    "b": {**_LADDER_BASE, "num_routed_experts": 28},
    "c": {**_LADDER_BASE, "num_routed_experts": 56},
    # DENSE CONTROL at matched active compute: top-4 of 4 means every expert
    # is always on, so total == active and there is no sparsity. Required
    # because Krajewski et al. (roadmap §17.4) find MoE needs LONGER training
    # than dense before it pulls ahead, and v7's 5.3B-token budget is a third
    # of their fitted minimum — without this arm, a "sparsity hurts" result
    # and a "we are before the crossover" result look identical.
    "dense": {**_LADDER_BASE, "num_routed_experts": 4, "dense_control": True},
    # ── E1 (roadmap §18.1): do per-loop adapters add anything over grouping?
    # Arm a WITH the HRA adapters and the shared expert shrunk back by the
    # same all-active parameter count (2688 -> 1920), so a and hra match on
    # total and FLOP-equivalent. E1 concluded (2026-09-02) that the no-HRA
    # arm led by 1.5 nats; this arm is kept so the result stays reproducible.
    "hra": {**_LADDER_BASE, "num_routed_experts": 14, "use_hra": True,
            "shared_expert_hidden": 1920},
    # ── G4 with MoEUT's G=4 prior (roadmap §17.2): 4 blocks x 5 loops.
    # A fourth block carries its own attention + shared expert, so total and
    # compute cannot BOTH be held when block count changes. This shape holds
    # both to within 2% of arm a (total +1.1%, FLOP-eq +0.2%, re-solved
    # 2026-09-30 for the reinvested shared expert) at 20 effective layers vs
    # a's 18: more distinct blocks, slightly smaller experts
    # (tests/test_novelty_experiments pins the 2% and the 64-multiple).
    "g4": {**_LADDER_BASE, "num_blocks": 4, "recursive_loops": 5,
           "num_routed_experts": 12, "expert_hidden": 896},
}
