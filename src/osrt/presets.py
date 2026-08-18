"""Canonical model presets for osrt.

Generated/validated by `scripts/compute_budget.py`. The headline preset
`OSRT_605M_A288M` is the locked target for the $350 build:

    8 routed experts (Mixtral-style, top-2 = 25% routing density), shared
    expert shrunk and routed experts widened so the model is ~607M physical
    / ~288M active (47.5%). 8 experts (not the original 12) trades sparsity
    for per-token capacity: each token sees a larger fraction of the routed
    knowledge base, less risk of expert under-utilization at this scale.

    Hitting both physical AND a tighter active number (e.g. 269M) with
    rank=256 HRA all-active is infeasible without dropping shared expert
    width below ~1k — so we accept 288M active as the floor.
"""

from __future__ import annotations

from osrt.config import OSRTConfig

# Locked $350-run target. See compute_budget.py: ~606.8M physical / 288.3M active.
OSRT_605M_A288M: dict = dict(
    dim=1536,
    heads=24,
    head_dim=64,
    num_kv_heads=8,            # GQA 24/8 + MLA-style compressed-latent KV cache
    vocab_size=65536,
    real_vocab_size=65536,
    num_blocks=3,
    recursive_loops=6,
    num_routed_experts=8,       # less sparse than 12 → more capacity per token
    top_k_experts=2,            # 2/8 = 25% routing density
    expert_hidden=3840,         # solved via compute_budget.py at rank=256
    shared_expert_hidden=2816,  # trial-and-error to fit budget; revisit at GPU phase
    adapter_rank=256,           # real HRA capacity (NOT LoRA-style 16)
    adapter_alpha=256.0,        # match rank so scale = 1.0
    # Manifold-Constrained Hyper-Connections (ARCHITECTURE.md §8): 4-channel
    # residual stream, Birkhoff/Sinkhorn doubly-stochastic mixing. Enabled for
    # GPU-phase testing — CPU pre-flight (scripts/ablate_features.py) showed
    # gradient amplification + NaN under sustained training, needs profiling on
    # real hardware to see if it's a CPU-precision artifact or a real bug.
    # See ARCHITECTURE.md §8 + LEARNINGS.md when v6 GPU runs begin.
    use_mhc=True,
    n_hc=4,
    mhc_sinkhorn_iters=20,
    swiglu_clamp=10.0,         # DeepSeek-style SwiGLU stability clamp (§7.8)
    # Attention sink DROPPED (was True). The manual sink path materialises a
    # (B,H,S,S) score matrix; at the seq-8192 instruction phase that is ~12GB
    # recomputed in the checkpointed backward — measured OOM (>85GB) at batch 2.
    # attention_sink=False routes through F.scaled_dot_product_attention (flash),
    # which never builds the score matrix: the SAME footprint fits at 35.9GB.
    # v5's proven path; scales to every phase. The sink had no demonstrated
    # benefit (kept only because it happened to fit at seq 2048).
    attention_sink=False,
    # B4: grouped-GEMM MoE dispatch. Removes the per-expert .nonzero() — the
    # only torch.compile graph break — so the model compiles fullgraph.
    # Validated on H100: loss tracks the loop path, dropless, ~9-12% faster
    # steady-state (gated by gradient-checkpointing recompute). Weights are
    # identical to the loop path, so checkpoints load under either dispatch.
    moe_grouped_gemm=True,
    # lean-v6 training stack (all already supported by the v5 model code)
    aux_loop_loss_weight=0.05,   # on from step 1 — anti loop-collapse
    # Multi-Token Prediction heads (§9.3, §11.4): 2 extra heads predicting +2/+3
    # from the final hidden state. Training-time only (droppable at deploy);
    # densifies the signal à la DeepSeek-V3/V4. beta = 0.3 per §11.4.
    mtp_heads=2,
    mtp_loss_weight=0.3,
    router_aux_loss_coeff=0.10,  # v5-proven balance pressure
    router_z_loss_coeff=1e-3,
    router_balance_bias_enabled=True,
    # sqrt(softplus) routing affinity (ARCHITECTURE.md §7.4, §16.3): the
    # balance bias steers TOP-K selection on the non-negative affinity, gating
    # weights renormalise the selected balanced affinities. Hash routing stays
    # off (hash_routing_blocks default 0) — reserved for later A/B testing.
    router_affinity="sqrt_softplus",
    max_position_embeddings=4096,
)


def build_config(preset: dict = OSRT_605M_A288M, **overrides) -> OSRTConfig:
    """Build a OSRTConfig from a preset, with optional overrides."""
    return OSRTConfig(**{**preset, **overrides})


# Back-compat alias — old code (app.py, sft_train.py) imports OSRT_605M_A279M.
# Resolve to the corrected preset until call sites are updated. The "A279M"
# number was based on a stale compute_budget run; the locked active count is
# 288M after restoring HRA rank=256.
OSRT_605M_A279M = OSRT_605M_A288M
