"""OSRT — Mixtral-style MoE without dense FFN.

Architecture changes from v4:
  - Dense FFN removed. Shared expert (hidden=4096) replaces it.
  - 8 routed experts × hidden=2048 (was 11 × 1024).
  - Top-2 softmax routing (Mixtral-style), renormalised gates.
  - Switch balance loss: num_experts * sum(f_i * p_i).
    NO importance loss (it enforced uniformity and killed v4's router).
    Uses a DeepSeek-style balance-bias controller plus annealed Gumbel top-k
    noise to keep experts alive while the router learns token preferences.
  - Capacity factor 2.0, tokens exceeding capacity skip that expert's branch.
  - Orthogonal per-expert initialisation breaks symmetry at step 0.

Kept from v4:
  - Recursive weight sharing (3 physical blocks × 6 loops).
  - Per-pass low-rank adapters (residual, not LoRA).
  - Causal attention with RoPE.
  - KV cache.
  - Loop embeddings for per-loop routing preferences.
  - HuggingFace PreTrainedModel compatibility.

Default config (measured on the actual model):
  Physical params      : 362,720,259 (~363M, LM head tied with embedding)
  Active / token (body): ~192M       (shared expert + 2 of 8 routed + attn + embed)
  Block applications    : 18          (num_blocks × recursive_loops)
"""

import math
import random
from contextlib import contextmanager
from typing import NamedTuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from osrt.config import OSRTConfig
from osrt.fused_ce import fused_linear_cross_entropy

# ── RoPE ────────────────────────────────────────────────────────────────


def compute_rope_freqs(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    scaling: dict | None = None,
) -> tuple[Tensor, Tensor]:
    """Pre-compute RoPE cos/sin tensors. Shape: (1, seq_len, 1, dim)."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE requires even dimension, got dim={dim}")

    effective_theta = theta
    if scaling is not None:
        stype = scaling.get("type", "").lower()
        factor = float(scaling.get("factor", 1.0))
        if stype == "ntk" and factor > 1.0:
            effective_theta = theta * (factor ** (dim / (dim - 2)))

    freqs = 1.0 / (
        effective_theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2]
            / dim
        )
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # RoPE buffers are stored in fp32 for stable precompute, but attention runs
    # under bf16 autocast. Cast here so q/k do not get promoted back to fp32.
    if cos.dtype != x.dtype or cos.device != x.device:
        cos = cos.to(device=x.device, dtype=x.dtype)
    if sin.dtype != x.dtype or sin.device != x.device:
        sin = sin.to(device=x.device, dtype=x.dtype)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rot * sin


# ── Expert FFN ──────────────────────────────────────────────────────────


def _glu_combine(
    gate: Tensor,
    up: Tensor,
    *,
    clamp: float | None,
    situ: bool,
    b_gate: float,
    b_up: float,
) -> Tensor:
    """Combine the gate and up branches of a (Swi)GLU expert.

    Shared by the eager `ExpertFFN.forward` and the grouped-GEMM path so the
    two stay bit-identical.

    Default (`situ=False`): SwiGLU — `F.silu(gate) * up` — with the optional
    hard `clamp` on the pre-activations. Unchanged from the original path.

    `situ=True`: SiTU-GLU (Kimi K3 §2.3.2). Smoothly caps the linear factor of
    the Swish gate and the up branch:

        gate' = b_gate * tanh(gate / b_gate) * sigmoid(gate)   # capped Swish
        up'   = b_up   * tanh(up   / b_up)
        out   = gate' * up'                                    # |out| <= b_gate*b_up

    It matches SwiGLU to first order near the origin, bounds large activations,
    and — unlike the hard clamp — keeps nonzero gradients past the cap. Adds no
    parameters, so it REPLACES the hard clamp rather than stacking with it.
    """
    if situ:
        gate = b_gate * torch.tanh(gate / b_gate) * torch.sigmoid(gate)
        up = b_up * torch.tanh(up / b_up)
        return gate * up
    if clamp is not None:
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    return F.silu(gate) * up


# Quantile-Balancing histogram resolution. These are numerical parameters, not
# tuning knobs: the range brackets both router-score conventions and the bin
# count sets threshold precision (16/1000 of the range).
QB_BINS = 512
QB_LO = -8.0
QB_HI = 8.0


class ExpertFFN(nn.Module):
    """SwiGLU / SiTU-GLU feed-forward. Used for both shared and routed experts."""

    def __init__(
        self,
        dim: int,
        hidden: int,
        clamp: float | None = None,
        *,
        situ: bool = False,
        situ_beta_gate: float = 4.0,
        situ_beta_up: float = 25.0,
    ) -> None:
        super().__init__()
        hidden = 64 * ((hidden + 63) // 64)  # TC-align
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        # Optional SwiGLU stability clamp (ARCHITECTURE.md §7.8): bound the
        # gate (max) and up (both sides) pre-activations so a single extreme
        # activation can't blow up the product. None → no clamp (no-op for a
        # healthy model; the bound just caps tails).
        self.clamp = clamp
        # SiTU-GLU (Kimi K3 §2.3.2): param-free smooth cap; when on it replaces
        # SwiGLU + the hard clamp. See `_glu_combine`.
        self.situ = situ
        self.situ_beta_gate = situ_beta_gate
        self.situ_beta_up = situ_beta_up

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        return self.w_down(_glu_combine(
            gate, up, clamp=self.clamp, situ=self.situ,
            b_gate=self.situ_beta_gate, b_up=self.situ_beta_up,
        ))


def orthogonal_expert_init(expert: ExpertFFN, seed: int, gain: float = 1.0) -> None:
    """Initialise an expert's projections with orthogonal columns.

    Ensures experts start in different feature subspaces so gradients
    push them in distinct directions. Uses QR decomposition of a random
    matrix (deterministic given seed).

    w_gate, w_up: (hidden, dim) — columns span a subspace of R^hidden
    w_down: (dim, hidden) — rows span a subspace of R^dim

    `gain` scales the resulting matrix to match standard init variance
    (roughly 1/sqrt(fan_in) for nn.Linear).
    """
    gen = torch.Generator(device=expert.w_gate.weight.device)
    gen.manual_seed(seed)
    with torch.no_grad():
        for lin in (expert.w_gate, expert.w_up, expert.w_down):
            w = lin.weight  # (out, in)
            rows, cols = w.shape
            # Generate random matrix with same dtype/device
            rand = torch.randn(
                max(rows, cols), min(rows, cols),
                generator=gen, device=w.device, dtype=w.dtype,
            )
            q, _ = torch.linalg.qr(rand)
            # q is orthonormal along its shorter axis. After slicing:
            #   rows >= cols (fat): q has orthonormal columns; element
            #     std = 1/sqrt(rows), NOT 1/sqrt(cols). This was the
            #     previous bug — for w_gate/w_up (hidden > dim) the
            #     weights came out ~13% under the claimed fan_in std.
            #   rows <  cols (tall, w_down): q has orthonormal rows
            #     after transpose; element std = 1/sqrt(cols) already.
            q = q[:rows, :cols] if rows >= cols else q[:cols, :rows].T
            # Target: std = gain / sqrt(fan_in) where fan_in = cols.
            # q's native element std is 1/sqrt(max(rows, cols)), so
            # rescale by sqrt(max(rows, cols) / cols) * gain.
            scale = gain * math.sqrt(max(rows, cols) / cols)
            w.copy_(q * scale)


# ── MoE Layer (Switch-style) ────────────────────────────────────────────


def _ref_grouped_mm(a: Tensor, b: Tensor, offs: Tensor) -> Tensor:
    """Reference (CPU-safe) grouped matmul matching torch._grouped_mm.

    a:    (M, K) tokens already sorted into contiguous per-expert spans.
    b:    (G, K, N) per-expert weight matrices.
    offs: (G,) int cumulative END offsets into M (offs[-1] == M); expert g is
          rows [prev:offs[g]] @ b[g].
    Returns (M, N).

    torch._grouped_mm's CUDA kernel is the production path (fused, no graph
    break). Its CPU *backward* is broken in torch 2.10, so this loop-of-matmuls
    is used on CPU (forward+backward both correct) and as the parity oracle for
    the kernel. The Python loop over G groups graph-breaks under compile — fine,
    it only ever runs on CPU/tests; CUDA uses the fused kernel.
    """
    outs = []
    lo = 0
    for g in range(b.shape[0]):
        hi = int(offs[g])
        outs.append(a[lo:hi] @ b[g])
        lo = hi
    return torch.cat(outs, dim=0)


class MoELayer(nn.Module):
    """Mixtral-style MoE: top-k (default 2) softmax routing, capacity-limited.

    Key differences from v4's MoELayer:
      - Switch balance loss: N * sum(f_i * p_i) — minimises at uniform without
        enforcing it on router probs.
      - No importance/z loss and no soft warmup/blend — sparse routing from
        step 0.
      - Optional persistent balance bias directly controls expert load.
      - Optional training-only Gumbel top-k noise, annealed by the trainer.
      - Dropped tokens (exceeded per-expert capacity) skip that expert's branch
        for this batch.
      - Orthogonal expert init (per-expert QR decomposition).
      - Top-k gates are renormalised so they sum to 1 — router decisions
        don't down-weight the MoE output just because k > 1.
    """

    def __init__(
        self, config: OSRTConfig, moe_seed: int = 0, block_idx: int = 0,
    ) -> None:
        super().__init__()
        self.dim = config.dim
        self.num_routed = config.num_routed_experts
        self.top_k = config.top_k_experts
        # B4: grouped-GEMM dispatch (vs the per-expert .nonzero() loop).
        self.grouped_gemm = getattr(config, "moe_grouped_gemm", False)
        self.expert_hidden = config.expert_hidden
        self.capacity_factor = config.router_capacity_factor
        self.num_loops = config.recursive_loops
        # Save seed for deferred orthogonal init (applied after post_init).
        self._moe_seed = moe_seed
        self._orthogonal_init_requested = config.expert_orthogonal_init
        # Hash routing (ARCHITECTURE.md §7.5): this physical block uses
        # deterministic top-1 hash routing instead of the learned router iff
        # block_idx < hash_routing_blocks. Hard switch, decided at construction.
        self.block_idx = block_idx
        self.use_hash_routing = block_idx < config.hash_routing_blocks

        # Shared expert: always active, larger hidden than routed experts.
        # Replaces v4's parallel dense FFN.
        clamp = getattr(config, "swiglu_clamp", None)
        situ_kw = {
            "situ": getattr(config, "situ_glu", False),
            "situ_beta_gate": getattr(config, "situ_beta_gate", 4.0),
            "situ_beta_up": getattr(config, "situ_beta_up", 25.0),
        }
        self.shared_expert = ExpertFFN(
            config.dim, config.shared_expert_hidden, clamp=clamp, **situ_kw,
        )

        # Routed experts
        self.experts = nn.ModuleList([
            ExpertFFN(config.dim, config.expert_hidden, clamp=clamp, **situ_kw)
            for _ in range(self.num_routed)
        ])

        # Router: projects (hidden + loop_emb) → num_routed logits
        self.loop_embeddings = nn.Embedding(config.recursive_loops, config.dim)
        self.loop_embeddings._osrt_init_std = config.loop_embedding_init_std
        self.router = nn.Linear(config.dim, self.num_routed, bias=False)
        # Affinity transform: "softmax" (historical) or "sqrt_softplus"
        # (DeepSeek-V4). See forward() for how each maps logits -> routing.
        self.router_affinity = config.router_affinity
        self.register_buffer(
            "gumbel_tau",
            torch.tensor(config.router_gumbel_tau_init, dtype=torch.float32),
        )

        # Per-loop, per-expert additive load-balancing bias. This is part of
        # the routing mechanism, not an optimizer parameter: it is applied in
        # train/eval selection and saved in checkpoints, then updated once per
        # optimizer step by the training loop via apply_balance_update().
        #
        # Capacity is enforced per MoE call, so loop-specific load imbalance
        # must be corrected per loop. A single block-level bias can look
        # balanced in aggregate while individual loop calls overflow.
        self.balance_mode = config.router_balance_mode
        self.bias_enabled = config.router_balance_bias_enabled
        self.bias_update_rate = config.router_balance_bias_update_rate
        self.bias_ema_rate = config.router_balance_bias_ema_rate
        self.bias_max = config.router_balance_bias_max
        self.register_buffer(
            "router_balance_bias",
            torch.zeros(self.num_loops, self.num_routed, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "balance_count_accum",
            torch.zeros(self.num_loops, self.num_routed, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "balance_total_accum",
            torch.zeros(self.num_loops, dtype=torch.float32),
            persistent=False,
        )
        # ── Quantile Balancing (Kimi K3) ────────────────────────────────
        # Histogram of the PRE-BIAS router score per (loop, expert). The
        # update sets each expert's bias from the score quantile matching its
        # target load, so every expert ends up with the same share of its own
        # distribution above the selection threshold. Deterministic: no update
        # rate, no EMA, no clamp target to tune. QB_LO/HI bracket both score
        # conventions (sqrt(softplus) affinity >= 0, and raw logits).
        self.qb_bins = QB_BINS
        self.qb_lo, self.qb_hi = QB_LO, QB_HI
        self.register_buffer(
            "qb_hist",
            torch.zeros(self.num_loops, self.num_routed, QB_BINS,
                        dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "qb_token_count",
            torch.zeros(self.num_loops, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expert_ema_fraction",
            torch.full(
                (self.num_loops, self.num_routed),
                1.0 / self.num_routed,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.balance_accum_enabled = True

        # When False, MoELayer.forward skips the ~21 .item()/.tolist()
        # calls in the telemetry block (one per stat × multiple stats).
        # On CUDA each .item() forces synchronisation; with 18 effective
        # MoE applications per forward this adds up. The training loop
        # flips this off on non-logging steps via
        # OSRTForCausalLM.set_moe_telemetry(False). Default True so
        # downstream consumers (monitoring, test_monitoring)
        # keep working without explicit opt-in.
        self.telemetry_enabled: bool = True

        # NOTE: orthogonal expert init is NOT applied here because HF's
        # post_init() walks the module tree and calls _init_weights on every
        # nn.Linear, which would stomp the orthogonal weights. Apply via
        # apply_orthogonal_init() after post_init() has finished.

        # Per-layer losses (set during forward, read by wrapper).
        # balance_loss   — Switch global imbalance, scaled by aux coeff.
        # z_loss         — (logsumexp router_logits)^2; bounds magnitude.
        # seq_balance_loss — Per-sequence Switch; opt-in long-context safety.
        self.balance_loss: Tensor | None = None
        self.z_loss: Tensor | None = None
        self.seq_balance_loss: Tensor | None = None

        # Telemetry — plain Python lists, zero cost.
        # per_token_entropy: mean_token entropy of softmax (real sharpness signal).
        # marginal_entropy: entropy of batch-mean p vector (balance proxy).
        # assignment_entropy: entropy of hard-assignment fractions f.
        # raw_max_prob: mean top-1 softmax prob BEFORE renormalisation (router
        #   confidence — uniform 1/E means no opinion, >1/E means preferences).
        # top_margin: mean (p_rank0 - p_rank1) (confidence gap).
        # drop_rate: fraction of (token, rank) pairs dropped by capacity cap.
        self.last_per_token_entropy: list[float] = [0.0] * config.recursive_loops
        self.last_marginal_entropy: list[float] = [0.0] * config.recursive_loops
        self.last_assignment_entropy: list[float] = [0.0] * config.recursive_loops
        self.last_clean_per_token_entropy: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_clean_marginal_entropy: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_clean_assignment_entropy: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_expert_fraction: list[list[float]] = [
            [0.0] * self.num_routed for _ in range(config.recursive_loops)
        ]
        self.last_clean_expert_fraction: list[list[float]] = [
            [0.0] * self.num_routed for _ in range(config.recursive_loops)
        ]
        self.last_drop_rate: list[float] = [0.0] * config.recursive_loops
        self.last_raw_max_prob: list[float] = [0.0] * config.recursive_loops
        self.last_top_margin: list[float] = [0.0] * config.recursive_loops
        self.last_prebias_per_token_entropy: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_prebias_marginal_entropy: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_prebias_assignment_entropy: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_prebias_expert_fraction: list[list[float]] = [
            [0.0] * self.num_routed for _ in range(config.recursive_loops)
        ]
        self.last_prebias_raw_max_prob: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_prebias_top_margin: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_clean_raw_max_prob: list[float] = [0.0] * config.recursive_loops
        self.last_clean_top_margin: list[float] = [0.0] * config.recursive_loops

    def apply_orthogonal_init(self) -> None:
        """Apply orthogonal per-expert init. MUST be called after HF post_init()."""
        if not self._orthogonal_init_requested:
            return
        # Skip on the meta device: HF from_pretrained builds the skeleton on
        # meta (torch.Generator(device=meta) would raise), then loads the real
        # weights — which overwrite this init anyway. Only fresh construction on
        # a real device needs the orthogonal init.
        if self.experts[0].w_gate.weight.is_meta:
            return
        # nn.ModuleList iterates as Module (generic); we know ours contains
        # ExpertFFN instances because we just constructed them.
        for ei, expert in enumerate(self.experts):
            assert isinstance(expert, ExpertFFN)
            orthogonal_expert_init(
                expert, seed=self._moe_seed * 1000 + ei, gain=1.0,
            )

    @torch._dynamo.disable
    @torch.no_grad()
    def _accumulate_balance_counts(
        self, top_idx: Tensor, loop_idx: int, score: Tensor | None = None,
    ) -> None:
        """Accumulate load statistics for the bias controller.

        `score` is the PRE-BIAS per-expert router score (N, E) and is required
        only by Quantile Balancing, which needs the distribution rather than
        just the realised counts.
        """
        if not self.bias_enabled:
            return
        flat = top_idx.reshape(-1)
        counts = torch.bincount(flat, minlength=self.num_routed).float()
        self.balance_count_accum[loop_idx].add_(counts)
        self.balance_total_accum[loop_idx].add_(counts.sum())

        if self.balance_mode == "quantile" and score is not None:
            width = (self.qb_hi - self.qb_lo) / self.qb_bins
            idx = ((score.detach().float() - self.qb_lo) / width).long()
            idx = idx.clamp_(0, self.qb_bins - 1).t()          # (E, N)
            self.qb_hist[loop_idx].scatter_add_(
                1, idx, torch.ones_like(idx, dtype=torch.float32),
            )
            self.qb_token_count[loop_idx].add_(float(score.shape[0]))

    @torch.no_grad()
    def _apply_quantile_balance(self) -> None:
        """Quantile Balancing (Kimi K3): set each expert's bias from the
        router-score quantile that matches its target load.

        For a target selection fraction p = top_k / num_routed, find for every
        expert the score threshold t_e with exactly p of ITS OWN score mass
        above it, then set bias_e = mean(t) - t_e. Every expert then presents
        the same fraction of its distribution above the common selection
        threshold, so load equalises in one shot rather than by nudging.

        Deterministic and hyperparameter-free: no update rate, no EMA, no
        clamp target. Contrast the heuristic controller, whose +/-gamma step
        was tuned at E=8 and which at E=28 must move 3.5x as many biases with
        3.5x less load signal per expert.
        """
        active = self.qb_token_count > 0
        if not active.any():
            return
        p = self.top_k / self.num_routed
        width = (self.qb_hi - self.qb_lo) / self.qb_bins

        hist = self.qb_hist[active]                       # (A, E, B)
        target = p * self.qb_token_count[active].view(-1, 1)   # (A, 1)
        # Mass at or above each bin. Non-increasing in bin index, so the count
        # of bins meeting the target is exactly the highest qualifying index+1.
        cum_above = hist.flip(-1).cumsum(-1).flip(-1)      # (A, E, B)
        idx = (cum_above >= target.unsqueeze(-1)).sum(-1) - 1   # (A, E)
        idx = idx.clamp_(0, self.qb_bins - 1)
        thresh = self.qb_lo + (idx.float() + 0.5) * width  # (A, E)

        # Centre so the biases sum to zero: only differences steer top-k, and
        # an uncentred set would drift without bound across updates.
        bias = thresh.mean(dim=-1, keepdim=True) - thresh
        self.router_balance_bias[active] = bias.clamp(
            -self.bias_max, self.bias_max,
        )
        self.qb_hist.zero_()
        self.qb_token_count.zero_()

    @torch.no_grad()
    def apply_balance_update(self) -> None:
        """Update per-expert routing bias once from accumulated clean load."""
        if not self.bias_enabled:
            return
        if self.balance_mode == "quantile":
            self._apply_quantile_balance()
            self.balance_count_accum.zero_()
            self.balance_total_accum.zero_()
            return
        active = self.balance_total_accum > 0
        if not active.any():
            return

        current_frac = self.expert_ema_fraction.clone()
        current_frac[active] = (
            self.balance_count_accum[active]
            / self.balance_total_accum[active].unsqueeze(-1)
        )
        self.expert_ema_fraction[active] = torch.lerp(
            self.expert_ema_fraction[active],
            current_frac[active],
            self.bias_ema_rate,
        )

        target = 1.0 / self.num_routed
        delta = torch.zeros_like(self.router_balance_bias)
        delta[active] = current_frac[active] - target
        self.router_balance_bias.add_(delta, alpha=-self.bias_update_rate)
        self.router_balance_bias.clamp_(-self.bias_max, self.bias_max)

        self.balance_count_accum.zero_()
        self.balance_total_accum.zero_()

    def _hash_route(
        self,
        x: Tensor,
        x_flat: Tensor,
        shared_out: Tensor,
        token_ids: Tensor,
        loop_idx: int,
    ) -> tuple[Tensor, Tensor]:
        """Deterministic top-1 hash routing (ARCHITECTURE.md §7.5).

        Each token is sent to exactly one expert,
            expert_id = (token_id + loop_idx) % num_routed_experts
        with gating weight 1.0. The shared expert is unaffected; we return the
        same (shared_out, routed_out) contract as the learned path so the Block
        is oblivious to the routing mode. Routing is deterministic, so there is
        no balance/z/seq aux loss to learn — those are set to zero tensors (kept
        non-None so the model's accumulation stays well-defined), and the
        telemetry attributes are populated from the hard hash histogram (never
        left stale from a previous forward, so the collapse monitor stays sane).
        """
        B, S, D = x.shape
        N = B * S
        E = self.num_routed
        device = x.device

        # Loop-indexed hash assignment, one expert per token.
        assign = (token_ids.reshape(N) + loop_idx) % E  # (N,), long

        # Dispatch: gather every token routed to each expert, run it, scatter
        # back at gate weight 1.0. No capacity cap — top-1 hashing is balanced
        # in expectation and dropping tokens here would only add noise.
        moe_out = torch.zeros_like(x_flat)
        for ei, expert in enumerate(self.experts):
            token_indices = (assign == ei).nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue
            # Cast to moe_out.dtype: under bf16 autocast the expert may
            # emit a dtype that index_add_ rejects against the bf16 buffer.
            moe_out.index_add_(
                0, token_indices,
                expert(x_flat[token_indices]).to(moe_out.dtype),
            )
        moe_out = moe_out.view(B, S, D)

        # Aux losses: deterministic routing has nothing to balance. Keep them as
        # zero tensors (not None) so OSRTModel.forward's `is not None` checks and
        # the wrapper's normalisation see a valid contribution.
        zero = torch.zeros((), device=device)
        self.balance_loss = zero
        self.z_loss = zero
        self.seq_balance_loss = zero

        # Telemetry from the hard hash assignment. f_i = fraction of tokens on
        # expert i (top-1, so it sums to 1). Entropy of f is the only meaningful
        # signal here; per-token entropy is 0 (a one-hot assignment) and the
        # "router confidence" metrics are 1.0 by construction.
        with torch.no_grad():
            counts = torch.bincount(assign, minlength=E).float()
            f = counts / counts.sum().clamp_min(1.0)
            f_list = f.tolist()
            f_log = torch.log(f.clamp_min(1e-10))
            assign_ent = -(f * f_log).sum().item()

            self.last_per_token_entropy[loop_idx] = 0.0
            self.last_marginal_entropy[loop_idx] = assign_ent
            self.last_assignment_entropy[loop_idx] = assign_ent
            self.last_expert_fraction[loop_idx] = f_list
            self.last_drop_rate[loop_idx] = 0.0
            self.last_raw_max_prob[loop_idx] = 1.0
            self.last_top_margin[loop_idx] = 1.0
            # Mirror onto the prebias and clean diagnostic families so the
            # collapse monitor (which reads last_clean_*) sees the deterministic
            # assignment rather than stale learned-router values.
            self.last_prebias_per_token_entropy[loop_idx] = 0.0
            self.last_prebias_marginal_entropy[loop_idx] = assign_ent
            self.last_prebias_assignment_entropy[loop_idx] = assign_ent
            self.last_prebias_expert_fraction[loop_idx] = f_list
            self.last_prebias_raw_max_prob[loop_idx] = 1.0
            self.last_prebias_top_margin[loop_idx] = 1.0
            self.last_clean_per_token_entropy[loop_idx] = 0.0
            self.last_clean_marginal_entropy[loop_idx] = assign_ent
            self.last_clean_assignment_entropy[loop_idx] = assign_ent
            self.last_clean_expert_fraction[loop_idx] = f_list
            self.last_clean_raw_max_prob[loop_idx] = 1.0
            self.last_clean_top_margin[loop_idx] = 1.0

        return shared_out, moe_out

    def _dispatch_loop(
        self, x_flat: Tensor, top_idx: Tensor, top_probs: Tensor, capacity: int,
    ) -> tuple[Tensor, int]:
        """Per-expert .nonzero() gather + index_add dispatch (the default).

        For each expert, gather every token that picked it at ANY top-k rank,
        apply the per-expert capacity cap (dropping a shuffled subset on
        overflow), run the expert, and scatter-add the gated output. Returns
        (moe_out_flat (N, D), total_dropped). The data-dependent .nonzero()
        graph-breaks torch.compile — see _dispatch_grouped for the fused path.
        """
        moe_out = torch.zeros_like(x_flat)
        total_dropped = 0
        for ei, expert in enumerate(self.experts):
            is_chosen = (top_idx == ei)  # (N, K), bool
            token_indices, rank_indices = is_chosen.nonzero(as_tuple=True)
            if token_indices.numel() == 0:
                continue
            # Capacity cap. nonzero() returns token-major order, so a naive
            # [:capacity] always drops late positions — shuffle first so every
            # position has equal survival probability. In eval capacity == N*K
            # so this never triggers.
            if token_indices.numel() > capacity:
                total_dropped += (token_indices.numel() - capacity)
                perm = torch.randperm(
                    token_indices.numel(), device=token_indices.device,
                )
                keep = perm[:capacity]
                token_indices = token_indices[keep]
                rank_indices = rank_indices[keep]
            expert_input = x_flat[token_indices]  # (T, D)
            expert_output = expert(expert_input)   # (T, D)
            # Gate = renormalised softmax prob for this (token, rank) pair;
            # applied in fp32 then cast (index_add_ rejects an fp32 source
            # against a bf16 buffer under autocast).
            gates = top_probs[token_indices, rank_indices].unsqueeze(-1)
            moe_out.index_add_(
                0, token_indices,
                (expert_output * gates).to(moe_out.dtype),
            )
        return moe_out, total_dropped

    def prepack_expert_weights(self) -> None:
        """Pre-stack + pre-cast the routed experts' SwiGLU weights for decode.

        _grouped_ffn otherwise torch.stack's all E experts' weights AND casts
        them fp32->bf16 on EVERY call — at decode that is 3 stacks x 18 MoE
        invocations x every token (~2.5B weight elements, ~35 GB/token of
        nominal traffic). NOTE: the predicted throughput win did NOT materialise
        under fullgraph torch.compile (paired bench: no b1 gain — Inductor
        likely already hoists the constant stack/cast, and/or decode is
        host-bound); kept because it is cheap at load time, value-identical,
        and may pay once CUDA-graph replay removes the host overhead. Costs
        ~791 MiB of extra bf16 buffers on the 605M preset.
        The weights are frozen at inference, so build the exact same
        (E, in, out) bf16 tensors ONCE and reuse. Non-persistent buffers:
        excluded from state_dict (checkpoint layout unchanged), moved by
        .to(device) with the module. Inference-only — call via
        optimize_for_inference(); training keeps the per-call stacking (weights
        change every step). Stale after any weight update; re-call to refresh.
        """
        cdt = torch.bfloat16
        self.register_buffer(
            "_packed_w_gate",
            torch.stack([e.w_gate.weight.t() for e in self.experts]).to(cdt),
            persistent=False,
        )
        self.register_buffer(
            "_packed_w_up",
            torch.stack([e.w_up.weight.t() for e in self.experts]).to(cdt),
            persistent=False,
        )
        self.register_buffer(
            "_packed_w_down",
            torch.stack([e.w_down.weight.t() for e in self.experts]).to(cdt),
            persistent=False,
        )

    def _invalidate_packed_weights(self) -> None:
        """Drop the prepacked buffers (fast path falls back to per-call
        stacking). Registered-buffer assignment to None keeps the names valid
        for getattr while freeing the memory."""
        if getattr(self, "_packed_w_gate", None) is not None:
            self._packed_w_gate = None
            self._packed_w_up = None
            self._packed_w_down = None

    def train(self, mode: bool = True) -> "MoELayer":
        # Entering training invalidates the packs: optimizer steps mutate the
        # expert weights in place and the packed copies would go silently
        # stale. Zero cost on the decode hot path (eval stays packed).
        if mode:
            self._invalidate_packed_weights()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        # New weights (checkpoint swap into a live model) invalidate any packs
        # built from the old ones. Re-run prepack_expert_weights() (or
        # optimize_for_inference()) after loading to re-enable the fast path.
        self._invalidate_packed_weights()
        super()._load_from_state_dict(*args, **kwargs)

    def _grouped_ffn(
        self, x_sorted: Tensor, offs: Tensor, use_kernel: bool | None = None,
    ) -> Tensor:
        """SwiGLU across all experts via grouped GEMM.

        x_sorted: (T, D) tokens sorted into contiguous per-expert spans.
        offs:     (E,) int32 cumulative END offsets (one per routed expert).
        Stacks the per-expert SwiGLU weights into (E, D, H)/(E, H, D) and runs
        three grouped matmuls. use_kernel=None auto-selects torch._grouped_mm on
        CUDA (fused, compile-friendly) and the CPU-safe reference otherwise
        (the kernel's CPU backward is broken). Weights are cast to the token
        dtype so the matmul precision matches the loop path's autocast.

        When prepack_expert_weights() has run (inference), the pre-stacked bf16
        buffers are used instead — same values, zero per-call stack/cast cost.
        """
        if use_kernel is None:
            use_kernel = x_sorted.is_cuda
        packed = getattr(self, "_packed_w_gate", None)
        if packed is not None and use_kernel:
            # Inference fast path: identical (E, in, out) bf16 tensors, built
            # once. bf16(w) commutes with stack/t(), so values match the
            # per-call path bit-for-bit.
            w_gate_b, w_up_b, w_down_b = (
                self._packed_w_gate, self._packed_w_up, self._packed_w_down,
            )
            cdt = torch.bfloat16
            xs = x_sorted.to(cdt)
            gate = torch._grouped_mm(xs, w_gate_b, offs=offs)
            up = torch._grouped_mm(xs, w_up_b, offs=offs)
            e0 = self.experts[0]
            h = _glu_combine(
                gate, up, clamp=e0.clamp, situ=e0.situ,
                b_gate=e0.situ_beta_gate, b_up=e0.situ_beta_up,
            )
            return torch._grouped_mm(h.to(cdt), w_down_b, offs=offs)
        # nn.Linear weight is (out, in); grouped_mm wants b = (E, in, out).
        w_gate = torch.stack([e.w_gate.weight.t() for e in self.experts])
        w_up = torch.stack([e.w_up.weight.t() for e in self.experts])
        w_down = torch.stack([e.w_down.weight.t() for e in self.experts])
        if use_kernel:
            # torch._grouped_mm (compiled) supports only bf16/fp16. The model
            # trains under bf16 autocast, so casting tokens + weights to bf16
            # also matches the loop path's matmul precision (autocast casts
            # nn.Linear inputs to bf16). The fp32 residual stream is cast here
            # exactly as autocast would for the loop's nn.Linear calls.
            cdt = torch.bfloat16
            xs = x_sorted.to(cdt)
            gate = torch._grouped_mm(xs, w_gate.to(cdt), offs=offs)
            up = torch._grouped_mm(xs, w_up.to(cdt), offs=offs)
        else:
            dt = x_sorted.dtype
            gate = _ref_grouped_mm(x_sorted, w_gate.to(dt), offs)
            up = _ref_grouped_mm(x_sorted, w_up.to(dt), offs)
        e0 = self.experts[0]
        h = _glu_combine(
            gate, up, clamp=e0.clamp, situ=e0.situ,
            b_gate=e0.situ_beta_gate, b_up=e0.situ_beta_up,
        )
        if use_kernel:
            out = torch._grouped_mm(h.to(cdt), w_down.to(cdt), offs=offs)
        else:
            out = _ref_grouped_mm(h, w_down.to(x_sorted.dtype), offs)
        return out

    def _dispatch_grouped(
        self, x_flat: Tensor, top_idx: Tensor, top_probs: Tensor,
    ) -> tuple[Tensor, int]:
        """Grouped-GEMM dispatch (B4). Dropless by construction.

        Flatten the (token, rank) pairs, sort by chosen expert, run one grouped
        SwiGLU over the sorted tokens, gate, then scatter-add back per token.
        Equivalent to _dispatch_loop in the no-drop regime; in training it keeps
        every token (no capacity cap). Fixed-shape ops only (argsort,
        scatter_add, cumsum, index_add) so the path is torch.compile-clean AND
        CUDA-graph capturable. Returns (moe_out_flat (N, D), 0).
        """
        N, D = x_flat.shape
        K = self.top_k
        E = self.num_routed
        pair_expert = top_idx.reshape(-1)                  # (N*K,)
        pair_gate = top_probs.reshape(-1)                  # (N*K,)
        pair_token = torch.arange(
            N, device=x_flat.device,
        ).repeat_interleave(K)                             # (N*K,)
        # Sort pairs by expert (stable → deterministic) so each expert's tokens
        # form one contiguous span for the grouped GEMM.
        order = torch.argsort(pair_expert, stable=True)
        sorted_token = pair_token[order]
        sorted_gate = pair_gate[order]
        # Per-expert counts via scatter_add, NOT torch.bincount: CUDA bincount
        # synchronizes internally (device->host max-value inspection), which
        # invalidates CUDA-graph stream capture (cudaErrorStreamCaptureInvalidated
        # — located via the generated inductor partition code). scatter_add of
        # ones is pure device work with identical integer results.
        counts = torch.zeros(
            E, dtype=torch.long, device=x_flat.device,
        ).scatter_add_(0, pair_expert, torch.ones_like(pair_expert))
        offs = counts.cumsum(0).to(torch.int32)            # (E,) cumulative ends
        x_sorted = x_flat[sorted_token]                    # (N*K, D)
        out = self._grouped_ffn(x_sorted, offs)            # (N*K, D)
        # Gate in fp32 (like the loop), then scatter-add back per token.
        out = out * sorted_gate.unsqueeze(-1)
        moe_out = torch.zeros_like(x_flat)
        moe_out.index_add_(0, sorted_token, out.to(moe_out.dtype))
        return moe_out, 0

    def _dispatch_bmm(
        self, x_flat: Tensor, top_idx: Tensor, top_probs: Tensor,
    ) -> tuple[Tensor, int]:
        """Small-N decode dispatch via gather + bmm. CUDA-graph capture-safe.

        torch._grouped_mm performs CUDA operations that are not permitted
        during stream capture (cudaErrorStreamCaptureUnsupported — located via
        the inductor partition traceback), so the decode step cannot use
        _dispatch_grouped under CUDA graphs. At decode N (=B) is tiny, so
        instead: index_select each (token, rank) pair's expert weights from
        the prepacked (E, in, out) stacks and run three bmms. No sort, no
        counts, no offsets — every op is capture-safe, and for N*K this small
        it also skips _dispatch_grouped's argsort/scatter entirely.

        Same experts, same weights; bmm-vs-grouped bf16 accumulation order may
        differ (gate with logits/ppl like every other inference change).
        Requires prepack_expert_weights() (falls back to stacking on the fly
        if the packs are absent — correct, just slower).
        """
        N, D = x_flat.shape
        K = self.top_k
        pair_expert = top_idx.reshape(-1)                    # (N*K,)
        w_gate = getattr(self, "_packed_w_gate", None)
        if w_gate is None:
            cdt = x_flat.dtype
            w_gate = torch.stack(
                [e.w_gate.weight.t() for e in self.experts]).to(cdt)
            w_up = torch.stack(
                [e.w_up.weight.t() for e in self.experts]).to(cdt)
            w_down = torch.stack(
                [e.w_down.weight.t() for e in self.experts]).to(cdt)
        else:
            w_up, w_down = self._packed_w_up, self._packed_w_down
        cdt = w_gate.dtype

        # (N*K, 1, D) @ (N*K, D, H) -> (N*K, 1, H)
        x_pairs = (
            x_flat.to(cdt).unsqueeze(1).expand(N, K, D).reshape(N * K, 1, D)
        )
        gate = torch.bmm(x_pairs, w_gate.index_select(0, pair_expert))
        up = torch.bmm(x_pairs, w_up.index_select(0, pair_expert))
        e0 = self.experts[0]
        h = _glu_combine(
            gate, up, clamp=e0.clamp, situ=e0.situ,
            b_gate=e0.situ_beta_gate, b_up=e0.situ_beta_up,
        )
        out = torch.bmm(h.to(cdt), w_down.index_select(0, pair_expert))
        # Gate and combine the K experts per token.
        out = out.view(N, K, D) * top_probs.unsqueeze(-1).to(out.dtype)
        return out.sum(dim=1).to(x_flat.dtype), 0

    def forward(
        self,
        x: Tensor,
        loop_idx: int,
        token_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through MoE.

        Args:
            x: Hidden states (B, S, dim).
            loop_idx: Current recursive loop index.
            token_ids: Optional (B, S) input token ids. Only consumed when this
                block hash-routes (block_idx < hash_routing_blocks); the learned
                router never looks at them. Defaults to None so the standard
                path and signature stay backward-compatible.

        Returns:
            (shared_out, routed_out): both (B, S, dim). Caller applies
            moe_gate to routed_out only; shared_out is always full weight.
        """
        B, S, D = x.shape
        N = B * S
        x_flat = x.reshape(N, D)

        # Reset losses (prevents stale values in eval)
        self.balance_loss = None
        self.z_loss = None
        self.seq_balance_loss = None

        # Shared expert (always active, not gated by caller's moe_gate)
        shared_out = self.shared_expert(x)

        # ── Hash routing (ARCHITECTURE.md §7.5) ──
        # Deterministic top-1 dispatch: expert_id = (token_id + loop_idx) % E,
        # a loop-indexed hash. No learned router, no balance loss/z-loss/Gumbel.
        # Used to stabilise early blocks before the learned router warms up.
        if self.use_hash_routing:
            if token_ids is None:
                raise ValueError(
                    "hash routing requires token_ids at the MoE layer; "
                    "OSRTModel.forward must thread input_ids through the block."
                )
            return self._hash_route(x, x_flat, shared_out, token_ids, loop_idx)

        # Router: add loop embedding, project to expert scores
        loop_emb = self.loop_embeddings.weight[loop_idx].view(1, 1, D)
        router_input = x + loop_emb
        router_logits = self.router(router_input.reshape(N, D))  # (N, E)

        # The routing math below operates on a per-expert "probability-like"
        # view of three router states — raw (pre-bias, no Gumbel), clean (bias,
        # no Gumbel) and the noisy selection path (bias + Gumbel). Two affinity
        # transforms are supported and produce these three views differently:
        #
        #   "softmax"       — historical Mixtral/v5 path. The balance bias is
        #                     added to the LOGITS pre-softmax; each view is a
        #                     softmax over its logits. Bit-identical to before.
        #   "sqrt_softplus" — DeepSeek-V4. affinity = sqrt(softplus(logits)) is
        #                     always non-negative; the balance bias is added to
        #                     the AFFINITY (not the logits); top-k selection and
        #                     the renormalised gating weights operate on that
        #                     balanced affinity. Telemetry and the Switch balance
        #                     loss consume an affinity-normalised probability
        #                     view (affinity / affinity.sum) so every downstream
        #                     entropy/fraction/z-loss stays well-defined.
        # Inference fast path (eval + telemetry off): only the deployed
        # routing decision is needed — balanced affinity -> ONE top-k ->
        # renormalised gates -> dispatch. The clean/prebias top-k views and
        # the balance/Z/seq losses exist for training gradient + telemetry;
        # at eval probs == clean_probs exactly (Gumbel is training-only), so
        # they are duplicates. Skipping them removes 2 top-ks + 3 fp32
        # reductions per MoE call (x18/token at decode) and shrinks the
        # CUDA-graph capture surface.
        compute_aux = self.training or self.telemetry_enabled
        affinity_mode = self.router_affinity
        if affinity_mode == "sqrt_softplus":
            # Non-negative per-expert affinity. softplus keeps it smooth and
            # strictly positive; sqrt compresses the tail (DeepSeek-V4).
            affinity = torch.sqrt(F.softplus(router_logits))  # (N, E)
            if self.bias_enabled:
                loop_bias = self.router_balance_bias[loop_idx].view(1, -1)
                clean_affinity = affinity + loop_bias
            else:
                clean_affinity = affinity
            # The aux-loss-free balance bias can push an affinity negative;
            # clamp at 0 so the normalised view and the gating weights stay
            # non-negative (the bias still steers top-k selection via ordering).
            clean_affinity = clean_affinity.clamp_min(0.0)

            # Training-time noisy exploration: Gumbel is added to the balanced
            # AFFINITY (not logits) so cold experts still get explored under the
            # affinity transform. Annealed to 0 by the trainer before eval.
            selection_affinity = clean_affinity
            if self.training:
                u = torch.rand_like(clean_affinity).clamp_(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(u))
                tau = cast(Tensor, self.gumbel_tau).to(
                    dtype=clean_affinity.dtype,
                )
                selection_affinity = (clean_affinity + tau * gumbel).clamp_min(
                    0.0,
                )
            probs = selection_affinity / selection_affinity.sum(
                dim=-1, keepdim=True,
            ).clamp_min(1e-9)
            if compute_aux:
                raw_router_probs = affinity / affinity.sum(
                    dim=-1, keepdim=True,
                ).clamp_min(1e-9)
                clean_probs = clean_affinity / clean_affinity.sum(
                    dim=-1, keepdim=True,
                ).clamp_min(1e-9)
        else:
            if self.bias_enabled:
                loop_bias = self.router_balance_bias[loop_idx].view(1, -1)
                clean_logits = router_logits + loop_bias
            else:
                clean_logits = router_logits

            # Training-time noisy top-k exploration. This prevents experts that
            # lose the first few router updates from going permanently cold. The
            # trainer anneals gumbel_tau to zero before the 5k health gate, so
            # the final pass is evaluated on the clean router.
            selection_logits = clean_logits
            if self.training:
                u = torch.rand_like(clean_logits).clamp_(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(u))
                tau = cast(Tensor, self.gumbel_tau).to(dtype=clean_logits.dtype)
                selection_logits = clean_logits + tau * gumbel

            # Softmax probabilities
            probs = F.softmax(selection_logits, dim=-1)  # (N, E)
            if compute_aux:
                raw_router_probs = F.softmax(router_logits, dim=-1)
                # "Clean" means deterministic deployed routing: bias applied,
                # no Gumbel. Raw un-biased logits are diagnostic only once the
                # controller is enabled.
                clean_probs = F.softmax(clean_logits, dim=-1)  # (N, E)

        # Top-k selection (raw probs, before renormalisation)
        raw_top_probs, top_idx = probs.topk(self.top_k, dim=-1)  # (N, K)
        if compute_aux:
            clean_raw_top_probs, clean_top_idx = clean_probs.topk(
                self.top_k, dim=-1,
            )
            prebias_raw_top_probs, raw_balance_top_idx = raw_router_probs.topk(
                self.top_k, dim=-1,
            )
        if self.training and self.balance_accum_enabled:
            prebias_score = (
                affinity if affinity_mode == "sqrt_softplus" else router_logits
            )
            self._accumulate_balance_counts(
                clean_top_idx, loop_idx, score=prebias_score,
            )

        # Renormalise so the K chosen gates sum to 1. Without this, the MoE
        # output would be down-weighted when K > 1 just because softmax is
        # spread across E>K experts. Renormalisation keeps the MoE branch
        # at a consistent magnitude regardless of K.
        top_probs = raw_top_probs / raw_top_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)

        # Per-expert capacity. In training, enforce the cap to force
        # balancing pressure. In eval/inference, disable drops entirely so
        # generation is chunk-stable (prefill-then-decode must match a full
        # forward). Inference non-determinism across chunks was a documented
        # v4 failure mode — v5 makes eval drop-free by construction.
        if self.training:
            capacity = max(
                1,
                int(math.ceil(self.capacity_factor * self.top_k * N / self.num_routed)),
            )
        else:
            capacity = N * self.top_k  # effectively unlimited (one pair per slot)

        if compute_aux:
            # Switch balance loss extended to top-k. Compute it on the RAW
            # router logits, not the noisy dispatch path or the bias-corrected
            # clean path. Gumbel is exploration and bias is an external
            # controller; the aux gradient must still push the learned router
            # itself away from collapse. Dispatch below still uses bias+Gumbel
            # top_idx/probs.
            #   f_i = fraction of token-expert pairs routed to expert i.
            #         Count each top-k membership, divide by N*K so sum(f)=1.
            #   p_i = mean softmax prob for expert i (sums to 1).
            #   loss = E * sum(f_i * p_i). Minimum at uniform = 1.0.
            raw_balance_one_hot = F.one_hot(
                raw_balance_top_idx, num_classes=self.num_routed,
            )
            # Compute balance loss in fp32. Under bf16 autocast, f·p can
            # underflow late in training when both are near 1/E (= 0.125 for
            # E=8); fp32 keeps the product and sum precise so the gradient
            # signal survives into long runs.
            raw_balance_f = (
                raw_balance_one_hot.float().sum(dim=(0, 1)) / (N * self.top_k)
            )
            raw_balance_p = raw_router_probs.float().mean(dim=0)
            self.balance_loss = self.num_routed * (
                raw_balance_f * raw_balance_p
            ).sum()

            # Router Z-loss (ST-MoE §3.2): mean_token (logsumexp(logits))^2.
            # Bounds the absolute magnitude of router logits so bf16/fp8
            # softmax exponentials don't overflow, and keeps early softmax
            # distributions flatter so cold experts retain non-zero gradient
            # through LR warmup. Computed on raw router logits (pre-bias,
            # pre-Gumbel) so the penalty acts on the learned router itself.
            # fp32 for the same precision reasons as balance_loss above.
            z = torch.logsumexp(router_logits.float(), dim=-1)  # (N,)
            self.z_loss = (z ** 2).mean()

            # Sequence-wise balance loss (DeepSeek-V3 §5.2). Penalises
            # imbalance INSIDE each individual sequence, complementing the
            # global balance_loss above. Useful at long context (Phase 3
            # seq_len=8192) where one document can dominate one micro-batch
            # even when the global batch averages to balanced. Computed
            # under no_grad would defeat the purpose — we want the per-seq
            # gradient to push the router away from intra-sequence collapse.
            # Uses the same raw (un-noised) routing decisions as
            # balance_loss for a coherent gradient signal.
            seq_one_hot = raw_balance_one_hot.float().view(
                B, S, self.top_k, self.num_routed,
            )
            f_seq = seq_one_hot.sum(dim=(1, 2)) / (S * self.top_k)  # (B, E)
            p_seq = raw_router_probs.float().view(B, S, self.num_routed).mean(
                dim=1,
            )                                                       # (B, E)
            self.seq_balance_loss = self.num_routed * (
                f_seq * p_seq
            ).sum(dim=-1).mean()
        else:
            # Inference fast path: consumers null-check these (OSRTModel's
            # accumulation + the trainer), so None cleanly signals "not
            # computed this forward" — and drops any stale training tensors.
            self.balance_loss = None
            self.z_loss = None
            self.seq_balance_loss = None

        # Dispatch the gated top-k assignment to the routed experts. Two
        # numerically-equivalent (in the no-drop regime) implementations (B4):
        #   loop:    per-expert .nonzero() gather + index_add — correct, but the
        #            data-dependent .nonzero() is the only torch.compile break.
        #   grouped: sort pairs by expert + one grouped GEMM, dropless.
        if self.grouped_gemm:
            if not self.training and N * self.top_k <= 32:
                # Inference small-N (decode) path: capture-safe gather+bmm —
                # torch._grouped_mm is illegal under CUDA-graph capture, and
                # at N*K this small bmm also skips the argsort/scatter setup.
                # Threshold 32 (= b16 decode): the batch-scaling probe showed
                # bmm is throughput POISON past that (tiny GEMVs; b32 sampled
                # decode ran 618 tok/s on bmm vs 3,747 at b128 on grouped) —
                # so batched rollouts use the grouped GEMM, while b<=16
                # interactive decode stays CUDA-graph capturable.
                moe_out, total_dropped = self._dispatch_bmm(
                    x_flat, top_idx, top_probs,
                )
            else:
                moe_out, total_dropped = self._dispatch_grouped(
                    x_flat, top_idx, top_probs,
                )
        else:
            moe_out, total_dropped = self._dispatch_loop(
                x_flat, top_idx, top_probs, capacity,
            )

        moe_out = moe_out.view(B, S, D)

        # Telemetry (detached, CPU-side scalars).
        # These fixes address v4 misdiagnosis:
        #   - "router_entropy" was the entropy of the batch-marginal p vector,
        #     which stays at ln(E) for any well-balanced router even when
        #     per-token routing is razor-sharp. Rename to marginal_entropy
        #     and add per_token_entropy as the real sharpness signal.
        #   - max_prob was reported AFTER top-k renormalisation, so a
        #     uniform top-2 router showed max_prob = 0.5 not 1/E. Report raw.
        #   - Add top_margin = raw top-1 prob - raw top-2 prob, which
        #     directly measures router confidence in its primary pick.
        # Telemetry block — gated so non-logging training steps skip
        # the ~21 .item()/.tolist() CUDA syncs per MoE forward. The
        # training loop sets self.telemetry_enabled = False on
        # non-logging steps via OSRTForCausalLM.set_moe_telemetry().
        # When skipped, the self.last_* attributes retain the values
        # from the previous logging step — that's safe because
        # consumers (_collect_moe_metrics, MoE telemetry
        # block) only read them on logging steps too.
        if not self.telemetry_enabled:
            return shared_out, moe_out
        with torch.no_grad():
            # Dispatch/noisy assignment stats. These keep the existing telemetry
            # semantics: last_expert_fraction and marginal_entropy describe the
            # actual noisy dispatch path while Gumbel exploration is enabled.
            # Computed here (inside the telemetry guard) rather than above so
            # non-logging training steps skip them entirely — f/p/dispatch_one_hot
            # feed ONLY the telemetry below; the Switch balance loss uses the
            # separate raw_balance_* tensors, so this move is bit-identical.
            dispatch_one_hot = F.one_hot(top_idx, num_classes=self.num_routed)
            f = dispatch_one_hot.float().sum(dim=(0, 1)) / (N * self.top_k)
            p = probs.float().mean(dim=0)

            # Per-token entropy — the real sharpness metric. Uniform per-token
            # softmax => ln(E). Sharp routing => much lower. Average over tokens.
            log_probs = torch.log(probs.clamp_min(1e-10))
            per_token_ent = -(probs * log_probs).sum(dim=-1)  # (N,)
            per_token_ent_mean = per_token_ent.mean().item()

            # Marginal entropy (entropy of mean_token p). High = balanced,
            # low = some experts globally never picked. Keep as balance proxy.
            p_log = torch.log(p.clamp_min(1e-10))
            marginal_ent = -(p * p_log).sum().item()

            # Assignment entropy (hard f). Mirrors marginal but over hard picks.
            f_log = torch.log(f.clamp_min(1e-10))
            assign_ent = -(f * f_log).sum().item()

            # Drop rate (across all N*K dispatch opportunities). 0 at inference.
            drop_rate = total_dropped / max(N * self.top_k, 1)

            # Raw top-1 router confidence (before renormalisation).
            # Uniform router gives 1/E; preferences show as > 1/E.
            raw_max = raw_top_probs[:, 0].mean().item()

            # Top-1 vs top-2 margin (raw probs). Large = strong primary pick.
            if self.top_k >= 2:
                top_margin = (raw_top_probs[:, 0] - raw_top_probs[:, 1]).mean().item()
            else:
                top_margin = raw_top_probs[:, 0].mean().item()

            self.last_per_token_entropy[loop_idx] = per_token_ent_mean
            self.last_marginal_entropy[loop_idx] = marginal_ent
            self.last_assignment_entropy[loop_idx] = assign_ent
            self.last_expert_fraction[loop_idx] = f.tolist()
            self.last_drop_rate[loop_idx] = drop_rate
            self.last_raw_max_prob[loop_idx] = raw_max
            self.last_top_margin[loop_idx] = top_margin

            prebias_log_probs = torch.log(raw_router_probs.clamp_min(1e-10))
            prebias_per_token_ent = -(
                raw_router_probs * prebias_log_probs
            ).sum(dim=-1)
            prebias_p = raw_router_probs.float().mean(dim=0)
            prebias_p_log = torch.log(prebias_p.clamp_min(1e-10))
            prebias_marginal_ent = -(prebias_p * prebias_p_log).sum().item()
            prebias_one_hot = F.one_hot(
                raw_balance_top_idx, num_classes=self.num_routed,
            ).to(raw_router_probs.dtype)
            prebias_f = prebias_one_hot.sum(dim=(0, 1)) / (N * self.top_k)
            prebias_f_log = torch.log(prebias_f.clamp_min(1e-10))
            prebias_assign_ent = -(prebias_f * prebias_f_log).sum().item()
            prebias_raw_max = prebias_raw_top_probs[:, 0].mean().item()
            if self.top_k >= 2:
                prebias_top_margin = (
                    prebias_raw_top_probs[:, 0]
                    - prebias_raw_top_probs[:, 1]
                ).mean().item()
            else:
                prebias_top_margin = prebias_raw_top_probs[:, 0].mean().item()

            self.last_prebias_per_token_entropy[loop_idx] = (
                prebias_per_token_ent.mean().item()
            )
            self.last_prebias_marginal_entropy[loop_idx] = prebias_marginal_ent
            self.last_prebias_assignment_entropy[loop_idx] = prebias_assign_ent
            self.last_prebias_expert_fraction[loop_idx] = prebias_f.tolist()
            self.last_prebias_raw_max_prob[loop_idx] = prebias_raw_max
            self.last_prebias_top_margin[loop_idx] = prebias_top_margin

            clean_log_probs = torch.log(clean_probs.clamp_min(1e-10))
            clean_per_token_ent = -(clean_probs * clean_log_probs).sum(dim=-1)
            clean_p = clean_probs.mean(dim=0)
            clean_p_log = torch.log(clean_p.clamp_min(1e-10))
            clean_marginal_ent = -(clean_p * clean_p_log).sum().item()
            clean_one_hot = F.one_hot(
                clean_top_idx, num_classes=self.num_routed,
            ).to(clean_probs.dtype)
            clean_f = clean_one_hot.sum(dim=(0, 1)) / (N * self.top_k)
            clean_f_log = torch.log(clean_f.clamp_min(1e-10))
            clean_assign_ent = -(clean_f * clean_f_log).sum().item()
            clean_raw_max = clean_raw_top_probs[:, 0].mean().item()
            if self.top_k >= 2:
                clean_top_margin = (
                    clean_raw_top_probs[:, 0]
                    - clean_raw_top_probs[:, 1]
                ).mean().item()
            else:
                clean_top_margin = clean_raw_top_probs[:, 0].mean().item()

            self.last_clean_per_token_entropy[loop_idx] = (
                clean_per_token_ent.mean().item()
            )
            self.last_clean_marginal_entropy[loop_idx] = clean_marginal_ent
            self.last_clean_assignment_entropy[loop_idx] = clean_assign_ent
            self.last_clean_expert_fraction[loop_idx] = clean_f.tolist()
            self.last_clean_raw_max_prob[loop_idx] = clean_raw_max
            self.last_clean_top_margin[loop_idx] = clean_top_margin

        # Return (shared, routed) so the Block can apply moe_gate only to
        # the routed contribution. Shared expert stays at full weight.
        return shared_out, moe_out


# ── Recursive Block ─────────────────────────────────────────────────────


@contextmanager
def _balance_accumulation(moe: MoELayer, enabled: bool):
    previous = moe.balance_accum_enabled
    moe.balance_accum_enabled = enabled
    try:
        yield
    finally:
        moe.balance_accum_enabled = previous


@torch.compiler.disable
def _checkpoint_block(block_fn, *args, context_fn):
    # Dynamo's higher-order-op tracer raises NotImplementedError on
    # checkpoint(..., context_fn=...). Wrapping the call in
    # torch.compiler.disable forces an eager fallback for just the
    # checkpoint dispatch; the block_fn itself is still compiled when the
    # outer model is wrapped in torch.compile because the inner call
    # re-enters the compiled graph.
    return gradient_checkpoint(
        block_fn, *args, use_reentrant=False, context_fn=context_fn,
    )


class StaticKVCache:
    """Preallocated post-RoPE/post-norm K/V decode cache ("speed mode").

    The default (latent) cache stores the un-rotated compressed latent and
    grows by torch.cat, recomputing v_from_k + K-norm + RoPE over the ENTIRE
    history every decode step. This cache instead stores the finished K and V
    per effective layer in fixed (B, kv_heads, max_len, head_dim) buffers with
    a DEVICE-side cursor — no cat, no historical recompute, and (crucially)
    static tensor shapes/addresses: the prerequisite for CUDA-graph capture.

    Cost: 2x the latent cache's memory (K and V vs one latent), ~150 MB at
    B=1/ctx-4096 on the 605M preset. Historical K/V are position-frozen, so
    caching them is mathematically exact; bf16 GEMV-vs-GEMM accumulation order
    on the new token's v_from_k differs from the latent path -> gate with
    ppl/logit error, not token identity.

    Decode-only (S=1 steps). Prefill runs the latent path once, then
    RecursiveBlock.write_latent_to_static converts it into these buffers.
    The cursor advances ONCE per model forward (advance()), not per layer.
    """

    def __init__(
        self,
        num_layers: int,
        batch: int,
        kv_heads: int,
        head_dim: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.max_len = max_len
        self.k = [
            torch.zeros(batch, kv_heads, max_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v = [
            torch.zeros(batch, kv_heads, max_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        # Number of VALID positions [0, cursor). Device-side: decode indexing
        # (index_copy_/index_select/mask compare) consumes it without a host
        # sync, keeping the step CUDA-graph capturable.
        self.cursor = torch.zeros(1, dtype=torch.long, device=device)
        # Fixed position index row for the validity mask (kpos <= cursor-1).
        self.kpos = torch.arange(max_len, device=device)
        # CUDA-graph contract: these tensors keep one address for the model's
        # lifetime. mark_static_address tells compile's cudagraph wrapper NOT
        # to copy them into its private static storage — so the in-graph
        # index_copy_ mutations land in OUR buffers and persist across replays
        # (the HF StaticCache pattern). No-op without reduce-overhead.
        try:
            from torch._dynamo import mark_static_address
            for t in (*self.k, *self.v, self.cursor, self.kpos):
                mark_static_address(t)
        except ImportError:  # older torch — static cache still works uncaptured
            pass

    def advance(self) -> None:
        """Advance past the token just written. Call once per decode forward."""
        self.cursor += 1

    def layer(self, idx: int) -> "_StaticLayerView":
        return _StaticLayerView(
            k=self.k[idx], v=self.v[idx],
            cursor=self.cursor, kpos=self.kpos, max_len=self.max_len,
        )


class _StaticLayerView(NamedTuple):
    """One effective layer's slice of a StaticKVCache, as _attention sees it."""

    k: Tensor
    v: Tensor
    cursor: Tensor
    kpos: Tensor
    max_len: int


class RecursiveBlock(nn.Module):
    """Physical transformer block: attention + MoE (no dense FFN).

    FFN path is:
        shared_expert(x)
        + moe_gate * sum_{(t, k) in top_k} gate_{t,k} * expert_{top_idx[t,k]}(x_t)
    wrapped inside MoELayer. Top-k gates are renormalised so they sum to 1
    per token, and tokens exceeding per-expert capacity are dropped from
    that expert's branch (training only; disabled at inference).
    No parallel dense path — shared expert replaces it.
    """

    def __init__(self, config: OSRTConfig, block_idx: int = 0) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.kv_heads = config.num_kv_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.group_size = self.heads // self.kv_heads

        # Attention — GQA with an MLA-style compressed K/V latent.
        #   q_proj:    full query projection (heads × head_dim)
        #   kv_down:   compress the hidden state to a single latent of
        #              kv_dim (= kv_heads × head_dim). THIS is the only
        #              thing cached — un-rotated.
        #   v_from_k:  derive V from that same latent (V = W·c + b). Both
        #              K and V are linear functions of one cached latent,
        #              the same expressivity class as DeepSeek MLA's shared
        #              c_KV, at ~half the cache of storing K and V.
        self.norm_attn = nn.RMSNorm(config.dim)
        self.q_proj = nn.Linear(config.dim, self.heads * self.head_dim, bias=False)
        self.kv_down = nn.Linear(config.dim, self.kv_dim, bias=False)
        self.v_from_k = nn.Linear(self.kv_dim, self.kv_dim, bias=True)
        # QK-Norm: per-head RMSNorm on q and k before RoPE+SDPA. Bounds
        # attention logits so they don't explode in bf16/fp8 — protects
        # the downstream MoE router from inheriting pathological hidden
        # states. Per-head (head_dim) is the standard formulation; sharing
        # the norm parameter across heads keeps the addition lightweight
        # (~head_dim params per block) and matches Gemma2/Chameleon.
        self.norm_q = nn.RMSNorm(config.head_dim)
        self.norm_k = nn.RMSNorm(config.head_dim)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

        # Attention sink (ARCHITECTURE.md §6.6). Per-head learnable sink logits,
        # initialised to zeros (sink_logit=0 ⇒ the sink contributes exp(0)=1 to
        # the denominator). The sink adds an extra term to the softmax
        # denominator only — its "value" is zero, so it never enters the
        # numerator/output:
        #   s_{h,i,j} = exp(z_{h,i,j}) / (Σ_k exp(z_{h,i,k}) + exp(sink[h]))
        # This lets a query's weights sum to < 1 (a head can attend to
        # "nothing" when no key is relevant). The parameter is 1D (heads,) →
        # routed to AdamW by build_param_groups with no change there. None when
        # disabled so the standard SDPA path stays bit-identical.
        self.attention_sink = config.attention_sink
        if config.attention_sink:
            self.sink_logits = nn.Parameter(torch.zeros(self.heads))

        # MoE (shared + routed), pre-norm
        self.norm_moe = nn.RMSNorm(config.dim)
        self.moe = MoELayer(config, moe_seed=block_idx, block_idx=block_idx)

        # Gate on the MoE (routed) branch. Reparameterised through
        # softplus so the EFFECTIVE gate is always > 0:
        #   effective = softplus(moe_gate) = log(1 + exp(moe_gate))
        # The raw parameter is initialised to log(e - 1) ≈ 0.5413 so
        # softplus(raw) ≈ 1.0 at step 0 (matches the previous unbounded
        # 1.0 init). Without this constraint the scalar can drift
        # negative under task gradient and zero out the routed branch
        # entirely, recreating the v4 "dense crutch" failure mode where
        # the always-on shared expert does all the work and routed
        # experts receive no learning signal.
        # Read `effective_moe_gate()` (or compute F.softplus(moe_gate)
        # at use sites) to get the actual gate value.
        self.moe_gate = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))

        # Manifold-Constrained Hyper-Connections (one per sub-block, shared
        # across loop iterations). The block then
        # uses the proven standard single-stream residual.

    def effective_moe_gate(self) -> Tensor:
        return F.softplus(self.moe_gate)

    def _attention(
        self,
        x_in: Tensor,
        adapter_a: Tensor,
        adapter_b: Tensor,
        adapter_scale: float,
        rope_cos: Tensor,
        rope_sin: Tensor,
        past_key_value: Tensor | None,
        use_cache: bool,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Attention sub-block contribution (pre-residual): GQA + MLA latent.

        Returns (out_proj(attn) + adapter, present_latent). The caller adds it
        into the residual stream.
        """
        if isinstance(past_key_value, _StaticLayerView):
            return self._attention_static(
                x_in, adapter_a, adapter_b, adapter_scale,
                rope_cos, rope_sin, past_key_value,
            )
        B, S, D = x_in.shape
        adapter_out = adapter_scale * (x_in @ adapter_a @ adapter_b)

        h = self.norm_attn(x_in)
        q = self.q_proj(h).view(B, S, self.heads, self.head_dim)
        c_kv_new = self.kv_down(h)            # (B, S, kv_dim) — un-rotated latent

        # The cache holds ONLY the un-rotated latent. K and V are recomputed
        # from the full latent every step: RoPE is positional and V-from-K
        # must operate on un-rotated K, so neither may be cached rotated.
        if past_key_value is not None:
            c_kv = torch.cat([past_key_value, c_kv_new], dim=1)  # (B, L+S, kv_dim)
        else:
            c_kv = c_kv_new
        present_kv = c_kv if use_cache else None
        total_len = c_kv.shape[1]
        past_len = total_len - S

        # Derive K and V from the same latent (both linear in c_kv).
        k = c_kv.view(B, total_len, self.kv_heads, self.head_dim)
        v = self.v_from_k(c_kv).view(B, total_len, self.kv_heads, self.head_dim)

        # QK-Norm before RoPE.
        q = self.norm_q(q)
        k = self.norm_k(k)
        # Queries rotate at the new positions [past_len:total_len]; keys were
        # just rebuilt un-rotated, so they rotate over the whole [0:total_len].
        q = apply_rope(
            q, rope_cos[:, past_len:total_len].to(q.dtype),
            rope_sin[:, past_len:total_len].to(q.dtype),
        )
        k = apply_rope(
            k, rope_cos[:, :total_len].to(k.dtype),
            rope_sin[:, :total_len].to(k.dtype),
        )
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # GQA attention: q has `heads`, k/v have `kv_heads`; enable_gqa lets
        # SDPA broadcast KV groups without materialising repeated heads.
        gqa = self.group_size > 1
        if self.attention_sink:
            # Attention sink (ARCHITECTURE.md §6.6): the sink adds an extra term
            # to the softmax denominator only, which SDPA cannot express. Use
            # the manual path so we can apply the exact log-sum-exp rescale.
            attn_out = self._attention_with_sink(
                q, k, v, S, total_len, past_len,
            )
        elif key_padding_mask is not None:
            # Left-padded batch (eval): combine causal + key-padding into one
            # additive mask. RoPE is relative, so left-padding preserves the
            # relative positions among the real (contiguous, right-aligned)
            # tokens — no position_ids fix is needed; we only stop queries from
            # attending to pad KEYS. key_padding_mask is (B, total_len), 1=real.
            neg = torch.finfo(q.dtype).min
            qpos = torch.arange(
                past_len, total_len, device=q.device,
            ).view(S, 1)                                   # query abs positions
            kpos = torch.arange(total_len, device=q.device).view(1, total_len)
            causal = (kpos > qpos)                         # (S, total_len)
            pad = (key_padding_mask == 0).view(B, 1, total_len)  # (B,1,total_len)
            masked = causal.view(1, S, total_len) | pad          # (B, S, total_len)
            # Keep each query's own position unmasked so a pad-query row is
            # never fully masked (all -inf → NaN softmax → poisons the cached
            # latent across the 18 effective layers). Real queries are
            # unaffected: their diagonal is already a real key.
            self_pos = (kpos == qpos).view(1, S, total_len)
            masked = masked & ~self_pos
            attn_mask = torch.zeros(
                B, 1, S, total_len, device=q.device, dtype=q.dtype,
            ).masked_fill(masked.view(B, 1, S, total_len), neg)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=gqa,
            )
        elif past_len > 0 and S > 1:
            attn_mask = torch.full(
                (S, total_len), float("-inf"), device=q.device, dtype=q.dtype,
            )
            attn_mask = torch.triu(attn_mask, diagonal=1 + past_len)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=gqa,
            )
        elif S > 1:
            # Explicit branch instead of is_causal=(S > 1): under
            # torch.compile(dynamic=True) S is a SymInt, so (S > 1) is a
            # SymBool, which SDPA rejects (fullgraph break). The `elif`
            # inserts a shape guard -> two specializations (prefill S>1,
            # decode S==1), each passing a plain Python bool. Same math.
            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=gqa,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, enable_gqa=gqa,
            )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_out) + adapter_out, present_kv

    def _attention_static(
        self,
        x_in: Tensor,
        adapter_a: Tensor,
        adapter_b: Tensor,
        adapter_scale: float,
        rope_cos: Tensor,
        rope_sin: Tensor,
        view: _StaticLayerView,
    ) -> tuple[Tensor, None]:
        """Single-token decode against a StaticKVCache ("speed mode").

        Computes K/V for the NEW token only (the cached history is finished
        post-norm/post-RoPE K/V — position-frozen, never recomputed), writes
        them at the device-side cursor, and runs SDPA over the fixed-length
        buffer with a validity mask. Static shapes + no host sync: the form
        CUDA-graph capture requires. Assumes S == 1 (generate's decode steps).

        Math note: per-position values equal the latent path exactly; only the
        bf16 accumulation order of the new token's v_from_k (GEMV here vs
        history-wide GEMM there) differs -> gate with ppl/logit error.
        """
        B, S, D = x_in.shape
        adapter_out = adapter_scale * (x_in @ adapter_a @ adapter_b)

        h = self.norm_attn(x_in)
        q = self.q_proj(h).view(B, S, self.heads, self.head_dim)
        c_new = self.kv_down(h)                              # (B, 1, kv_dim)

        pos = view.cursor                                    # (1,) device-side
        cos = rope_cos.index_select(1, pos)                  # (1, 1, 1, hd)
        sin = rope_sin.index_select(1, pos)
        q = apply_rope(self.norm_q(q), cos, sin)
        k_new = apply_rope(
            self.norm_k(c_new.view(B, S, self.kv_heads, self.head_dim)),
            cos, sin,
        )
        v_new = self.v_from_k(c_new).view(B, S, self.kv_heads, self.head_dim)

        # Write the finished K/V at the cursor (in-place, fixed addresses).
        view.k.index_copy_(2, pos, k_new.transpose(1, 2).to(view.k.dtype))
        view.v.index_copy_(2, pos, v_new.transpose(1, 2).to(view.v.dtype))

        # Validity mask over the whole buffer: keep [0, cursor] (history + the
        # token just written). Device-side compare — no Python :cursor slice.
        mask = (view.kpos <= pos).view(1, 1, 1, view.max_len)
        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2).to(view.k.dtype), view.k, view.v,
            attn_mask=mask, enable_gqa=self.group_size > 1,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D).to(x_in.dtype)
        return self.out_proj(attn_out) + adapter_out, None

    def write_latent_to_static(
        self,
        c_kv: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        k_buf: Tensor,
        v_buf: Tensor,
    ) -> None:
        """One-time prefill conversion: latent cache -> finished K/V buffers.

        Applies exactly the latent path's K derivation (norm_k then RoPE over
        [0:L]) and V derivation (v_from_k) to the prefill latent, writing the
        results into a StaticKVCache's per-layer buffers at positions [0, L).
        """
        B, L, _ = c_kv.shape
        k = self.norm_k(c_kv.view(B, L, self.kv_heads, self.head_dim))
        k = apply_rope(k, rope_cos[:, :L], rope_sin[:, :L])
        v = self.v_from_k(c_kv).view(B, L, self.kv_heads, self.head_dim)
        k_buf[:, :, :L] = k.transpose(1, 2).to(k_buf.dtype)
        v_buf[:, :, :L] = v.transpose(1, 2).to(v_buf.dtype)

    def _attention_with_sink(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        S: int,
        total_len: int,
        past_len: int,
    ) -> Tensor:
        """Manual GQA attention with a per-head learnable sink (§6.6/§6.7).

        Exact log-sum-exp rescale of the standard attention output. If `out`
        and `lse = log Σ_k exp(z_{i,k})` are the usual (sink-free) attention
        output and per-query log-sum-exp of the scores, then adding exp(sink[h])
        to the denominator simply multiplies the output by
            Σexp(z) / (Σexp(z) + exp(sink)) = sigmoid(lse - sink[h]),
        because the sink's value is zero and so contributes nothing to the
        numerator. We therefore compute the masked scores, derive `out` and
        `lse` from one softmax/logsumexp, and rescale per head.

        flex_attention(return_lse=True) was investigated as the "flash + lse"
        route but on this target (torch 2.12, CPU) it (a) emits a deprecation
        warning for return_lse, (b) without torch.compile materialises the full
        score matrix anyway (no fused-kernel speedup), and (c) needs a custom
        mask_mod to express GQA broadcasting. The manual path below materialises
        the same score matrix, reuses the EXACT causal masking the SDPA path
        uses, and is fully correct; it is O(S·total_len) in memory per head —
        fine for our sequence lengths and the simplest thing that is provably
        right. Inputs q:(B,heads,S,hd), k/v:(B,kv_heads,total_len,hd).
        """
        B, H, _, hd = q.shape
        # Expand GQA groups so every query head sees its KV head. SDPA does this
        # internally via enable_gqa; here we do it explicitly with
        # repeat_interleave on the kv-head dim (groups of `group_size`).
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        scale = 1.0 / math.sqrt(hd)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,S,total_len)

        # Causal masking — identical semantics to the SDPA path. For the
        # cached-decode case the S query positions occupy [past_len:total_len],
        # so a key j is visible to query i iff j <= past_len + i. With S == 1
        # (single-token decode) all `total_len` keys are visible and no mask is
        # needed (matches SDPA's is_causal=False branch).
        if S > 1:
            row = torch.arange(S, device=scores.device).view(S, 1)
            col = torch.arange(total_len, device=scores.device).view(1, total_len)
            causal = col <= (past_len + row)  # (S, total_len) bool, True = keep
            scores = scores.masked_fill(~causal, float("-inf"))

        # Per-query log-sum-exp of the (masked) scores, then standard softmax.
        # Compute in fp32 for a stable exp/log; the sink rescale is sensitive to
        # the lse magnitude. lse: (B,H,S).
        scores_f = scores.float()
        lse = torch.logsumexp(scores_f, dim=-1)
        attn_weights = torch.softmax(scores_f, dim=-1).to(v.dtype)
        out = torch.matmul(attn_weights, v)  # (B,H,S,hd)

        # Sink rescale: multiply each head's output by sigmoid(lse - sink[h]).
        # sink_logits: (H,) → broadcast over (B,H,S). Done in fp32 then cast.
        sink = self.sink_logits.float().view(1, H, 1)
        rescale = torch.sigmoid(lse - sink).unsqueeze(-1).to(out.dtype)
        return out * rescale

    def _moe(
        self, x_in: Tensor, loop_idx: int, token_ids: Tensor | None = None,
    ) -> Tensor:
        """MoE sub-block contribution (pre-residual): shared + gated routed.

        token_ids (B, S) is forwarded to the MoE layer; it is only consumed when
        this block hash-routes, otherwise ignored."""
        h_shared, h_routed = self.moe(
            self.norm_moe(x_in), loop_idx, token_ids=token_ids,
        )
        return h_shared + self.effective_moe_gate() * h_routed

    def forward(
        self,
        x: Tensor,
        adapter_a: Tensor,
        adapter_b: Tensor,
        adapter_scale: float,
        rope_cos: Tensor,
        rope_sin: Tensor,
        loop_idx: int,
        past_key_value: Tensor | None = None,
        use_cache: bool = False,
        token_ids: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run attention then MoE. `x` is (B, S, D).

        token_ids (B, S) is the optional input-token tensor used only by
        hash-routing blocks; it defaults to None and is otherwise ignored."""
        f_attn, present_kv = self._attention(
            x, adapter_a, adapter_b, adapter_scale,
            rope_cos, rope_sin, past_key_value, use_cache,
            key_padding_mask=key_padding_mask,
        )
        x = x + f_attn
        x = x + self._moe(x, loop_idx, token_ids=token_ids)
        return x, present_kv


# ── MTP (Multi-Token Prediction) head ───────────────────────────────────


class MTPHead(nn.Module):
    """A single Multi-Token Prediction head (ARCHITECTURE.md §9.3, §11.4).

    Small projection applied to the FINAL post-norm_out hidden state before
    the WEIGHT-TIED LM head (the embedding) turns it into vocab logits. Head
    k predicts the token at offset +(1+k) — i.e. +2, +3, ... — during
    TRAINING only. These params are an auxiliary objective: they densify the
    training signal (DeepSeek-V3/V4) and are DROPPABLE at deployment, since
    inference/generation only ever uses the main +1 LM head.

    Structure per §9.3: RMSNorm(dim) + Linear(dim, dim, bias=False). The
    caller applies the tied embedding (via F.linear) to project to vocab, so
    no separate vocab matrix lives here.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x_final: Tensor) -> Tensor:
        return self.proj(self.norm(x_final))


# ── Main Model ──────────────────────────────────────────────────────────


class OSRTPreTrainedModel(PreTrainedModel):
    """Base class for OSRT models."""

    config_class = OSRTConfig
    base_model_prefix = "model"
    # Gradient checkpointing is managed internally via the private
    # OSRTModel._osrt_grad_ckpt gate (set by the trainer), NOT HF's mechanism —
    # HF's gradient_checkpointing_enable() is not wired to it. Advertise False
    # so HF doesn't attempt its own (which trips post_init and isn't what runs).
    supports_gradient_checkpointing = False
    # Keep each recursive block intact under device_map="auto" sharding.
    _no_split_modules = ["RecursiveBlock"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            custom_std = getattr(module, "_osrt_init_std", None)
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=custom_std if custom_std is not None else std,
            )


class OSRTModel(OSRTPreTrainedModel):
    """Core OSRT model (without LM head)."""

    def __init__(self, config: OSRTConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)

        cos, sin = compute_rope_freqs(
            config.max_position_embeddings,
            config.head_dim,
            config.rope_theta,
            scaling=config.rope_scaling,
        )
        # persistent=True so the rope tables are saved in the checkpoint and
        # restored by from_pretrained. They're derived from config, but HF's
        # from_pretrained builds the skeleton on the meta device and would
        # otherwise materialise these non-loaded buffers as uninitialised
        # garbage (corrupting RoPE on every reloaded model). ~2MB at the real
        # config — negligible. The on-the-fly recompute path still handles
        # seq_len beyond the cached range.
        self.register_buffer("rope_cos", cos, persistent=True)
        self.register_buffer("rope_sin", sin, persistent=True)

        # Physical blocks with distinct block-idx seeds so experts differ
        # across blocks too (not just within a block).
        self.blocks = nn.ModuleList(
            [RecursiveBlock(config, block_idx=bi) for bi in range(config.num_blocks)]
        )

        # Per-pass low-rank adapters
        total_pairs = config.num_blocks * config.recursive_loops
        self.adapters_a = nn.ParameterList(
            [nn.Parameter(torch.randn(config.dim, config.adapter_rank) * 0.01)
             for _ in range(total_pairs)]
        )
        self.adapters_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(config.adapter_rank, config.dim))
             for _ in range(total_pairs)]
        )
        self.adapter_scale = config.adapter_alpha / config.adapter_rank

        # Recursive-loop collapse telemetry. Per effective layer (loop × block)
        # we record the relative residual update ||Δx|| / ||x|| and the hidden
        # norm ||x||. A deep loop whose update → 0 has collapsed to a no-op.
        # Gated on telemetry_enabled (set per-step by the trainer, like the MoE
        # telemetry) so the hook never runs on normal compiled steps — keeps the
        # B4 fullgraph clean. Populated by forward; read by _collect_moe_metrics.
        self.telemetry_enabled: bool = True
        n_eff = config.num_blocks * config.recursive_loops
        self.last_loop_update_norm: list[float] = [0.0] * n_eff
        self.last_loop_hidden_norm: list[float] = [0.0] * n_eff

        self.norm_loop = nn.RMSNorm(config.dim)
        self.norm_out = nn.RMSNorm(config.dim)

        # Activation checkpointing for the recursive blocks (training-only;
        # the forward guards it with `self.training and not use_cache`).
        #
        # IMPORTANT: gate on our OWN private attribute, NOT the HF-managed
        # `gradient_checkpointing` name. OSRTModel subclasses PreTrainedModel,
        # which manages `gradient_checkpointing` via gradient_checkpointing_
        # enable()/post_init — setting config.gradient_checkpointing=True makes
        # post_init try to enable it and raise (our model doesn't implement
        # HF's hook) or silently no-op (version-dependent). So checkpointing is
        # OFF at construction and the trainer flips this private gate at
        # runtime (run_training). HF never touches `_osrt_grad_ckpt`.
        self._osrt_grad_ckpt = False

        # Side-effect storage for per-loop auxiliary LM-head losses.
        # Populated by forward() when aux_loop_loss_weight > 0 and the
        # model is in training mode. Consumed by OSRTForCausalLM
        # to compute the aux loss term, and by the train loop for
        # per-loop logging.
        self.last_intermediate_hiddens: list[Tensor] | None = None

    def _resolve_num_loops(self, num_loops: int | None) -> int:
        """Validate and resolve the variable loop count (ARCHITECTURE.md §12.2).

        The inference-compute knob: run only the first K of the trained
        recursive_loops. None → recursive_loops (the trained/default count,
        bit-identical to the historical path). Otherwise K must satisfy
        1 <= K <= recursive_loops; the aux-per-loop-LM-head training (§9.2)
        makes those reduced-loop outputs usable. Raises on out-of-range K so a
        caller can't silently run a malformed loop count.
        """
        if num_loops is None:
            return self.config.recursive_loops
        if not (1 <= num_loops <= self.config.recursive_loops):
            raise ValueError(
                f"num_loops must be in [1, recursive_loops="
                f"{self.config.recursive_loops}], got {num_loops}"
            )
        return num_loops

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: list[Tensor] | None = None,
        use_cache: bool = False,
        num_loops: int | None = None,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> tuple[
        Tensor, list[Tensor], Tensor, Tensor, Tensor,
        list[Tensor] | None,
    ]:
        """Forward pass.

        Args:
            input_ids: (B, S) token ids.
            past_key_values: per-effective-layer cached latents, or None.
            use_cache: return the per-layer present latents for decode.
            num_loops: optional variable loop count (ARCHITECTURE.md §12.2).
                None → config.recursive_loops (bit-identical to before). When
                set to K (1 <= K <= recursive_loops) only the first K recursive
                loops run before collapse/norm_out + the LM head — the
                controllable inference-compute knob (fewer loops = faster,
                slightly lower quality). Validated by _resolve_num_loops.
                NOTE: when use_cache is on, the cache is built per effective
                layer = num_blocks * loop; a reduced K writes/reads only the
                first K*num_blocks layers, so a multi-call decode (prefill +
                steps) MUST use the same K throughout or the layer indices stop
                lining up. generate() enforces that by threading one num_loops
                through the whole call.

        Returns:
            (hidden, loop_rms, balance_loss, z_loss, seq_balance_loss, presents)
        Each loss is the SUM across all (num_blocks * loops_run) MoE
        applications. The wrapper normalises by that count.
        """
        # Resolve the variable loop count (§12.2). When None this is exactly
        # recursive_loops, so every downstream count/index is unchanged.
        n_loops_to_run = self._resolve_num_loops(num_loops)

        x = self.embedding(input_ids)
        S = input_ids.shape[1]
        # The KV cache holds one latent per effective layer that actually ran.
        # With a reduced loop count K only the first K*num_blocks layers exist,
        # so the cache contract is keyed off the loops we are about to run (==
        # recursive_loops in the default path, so this is unchanged there).
        expected_past_layers = self.config.num_blocks * n_loops_to_run

        # Static ("speed mode") cache: fixed post-RoPE K/V buffers + device
        # cursor. Blocks receive per-layer views; the RoPE span is the whole
        # fixed buffer (static shape). Latent validation does not apply.
        static_cache = (
            past_key_values if isinstance(past_key_values, StaticKVCache)
            else None
        )
        past_length = 0
        if static_cache is not None:
            if len(static_cache.k) != expected_past_layers:
                raise ValueError(
                    f"StaticKVCache has {len(static_cache.k)} layers, "
                    f"expected {expected_past_layers}."
                )
        elif past_key_values is not None:
            if len(past_key_values) != expected_past_layers:
                raise ValueError(
                    f"Invalid past_key_values: expected "
                    f"{expected_past_layers} entries, "
                    f"got {len(past_key_values)}."
                )
            for idx, layer_past in enumerate(past_key_values):
                if layer_past is None:
                    continue
                if not isinstance(layer_past, torch.Tensor):
                    raise ValueError(
                        f"past_key_values[{idx}] must be a latent Tensor "
                        f"(B, seq, kv_dim)."
                    )
                layer_len = layer_past.shape[1]
                if past_length == 0:
                    past_length = layer_len
                elif layer_len != past_length:
                    raise ValueError(
                        f"KV cache length mismatch at layer {idx}."
                    )

        required_seq_len = (
            static_cache.max_len if static_cache is not None else past_length + S
        )
        # Pass the FULL-range [0:required_seq_len] cos/sin to each block: K is
        # rebuilt un-rotated from the cached latent and must be rotated over
        # the whole span, while Q is rotated only at its new positions. The
        # block slices each internally from past_len.
        if required_seq_len <= self.rope_cos.shape[1]:
            cos = self.rope_cos[:, :required_seq_len, :, :]
            sin = self.rope_sin[:, :required_seq_len, :, :]
        else:
            rope_cos, rope_sin = compute_rope_freqs(
                required_seq_len,
                self.rope_cos.shape[-1],
                theta=getattr(self.config, "rope_theta", 10000.0),
                device=x.device,
                scaling=getattr(self.config, "rope_scaling", None),
            )
            cos = rope_cos[:, :required_seq_len, :, :].to(
                device=self.rope_cos.device, dtype=self.rope_cos.dtype
            )
            sin = rope_sin[:, :required_seq_len, :, :].to(
                device=self.rope_sin.device, dtype=self.rope_sin.dtype
            )

        loop_rms: list[Tensor] = []
        total_balance_loss = torch.tensor(0.0, device=x.device)
        total_z_loss = torch.tensor(0.0, device=x.device)
        total_seq_balance_loss = torch.tensor(0.0, device=x.device)

        # Per-loop aux-loss capture (architecture fix for loop collapse).
        # Enabled when config.aux_loop_loss_weight > 0 and training. We
        # capture the hidden state at the END of each non-final loop
        # (after the 3 blocks, BEFORE norm_loop) so the aux LM head can
        # apply norm_out to it and predict the next token from the
        # intermediate representation. Forces gradient signal into
        # loops 0..N-2 instead of letting loop N-1 absorb everything.
        intermediate_hiddens: list[Tensor] = []
        capture_aux = (
            getattr(self.config, "aux_loop_loss_weight", 0.0) > 0.0
            and self.training
        )

        # Loop dropout (stochastic depth). With probability
        # loop_dropout_prob in training, truncate the loop chain to a
        # random length in [min_loops, n_loops_to_run]. The main task
        # loss then flows from a shorter chain's final output, forcing
        # the truncation point loop to be standalone-useful. Complements
        # the aux loss — aux pushes intermediate loops to predict in
        # parallel, dropout makes their predictions become the actual
        # model output some fraction of the time.
        #
        # NOTE: the ceiling here is the ALREADY-resolved n_loops_to_run
        # (== recursive_loops in the default num_loops=None path, so this
        # is bit-identical to before). The §12.2 inference knob and loop
        # dropout compose cleanly: an explicit num_loops=K caps the chain
        # at K, and dropout may only shorten it further. Dropout is
        # training-only, so an inference forward (model.eval()) always runs
        # exactly the resolved n_loops_to_run.
        if (
            self.training
            and getattr(self.config, "loop_dropout_prob", 0.0) > 0.0
            and random.random() < self.config.loop_dropout_prob
        ):
            min_loops = max(2, getattr(self.config, "loop_dropout_min_loops", 3))
            max_loops = n_loops_to_run
            if max_loops > min_loops:
                n_loops_to_run = random.randint(min_loops, max_loops)

        use_ckpt = self._osrt_grad_ckpt and self.training
        if use_ckpt and (use_cache or past_key_values is not None):
            raise ValueError(
                "KV caching is incompatible with gradient checkpointing."
            )
        presents: list[Tensor] | None = [] if use_cache else None

        for loop in range(n_loops_to_run):
            for block_idx, block in enumerate(self.blocks):
                idx = loop * self.config.num_blocks + block_idx
                adapter_a = self.adapters_a[idx]
                adapter_b = self.adapters_b[idx]
                if static_cache is not None:
                    layer_past = static_cache.layer(idx)
                elif past_key_values is not None:
                    layer_past = past_key_values[idx]
                else:
                    layer_past = None

                # Loop-collapse telemetry: snapshot the residual before this
                # block so we can record how much it changes it. Gated on
                # telemetry_enabled — on normal (compiled) steps this is skipped
                # entirely, so the fullgraph stays clean; on log steps it adds
                # the same .item() breaks the MoE telemetry already does.
                collect_loop = self.telemetry_enabled
                if collect_loop:
                    x_prev = x.detach()

                if use_ckpt:
                    def _block_fn(
                        _x, _a, _b, _cos, _sin,
                        _block=block, _scale=self.adapter_scale, _loop=loop,
                        _tok=input_ids,
                    ):
                        # token_ids is captured (closure default), not a
                        # checkpoint input — it carries no gradient and only
                        # hash-routing blocks read it.
                        return _block(
                            _x, _a, _b, _scale, _cos, _sin, _loop,
                            token_ids=_tok,
                        )[0]

                    def _context_fn(_block=block):
                        return (
                            _balance_accumulation(_block.moe, True),
                            _balance_accumulation(_block.moe, False),
                        )

                    x = _checkpoint_block(
                        _block_fn, x, adapter_a, adapter_b, cos, sin,
                        context_fn=_context_fn,
                    )
                else:
                    x, present_kv = block(
                        x, adapter_a, adapter_b,
                        self.adapter_scale, cos, sin,
                        loop_idx=loop,
                        past_key_value=layer_past,
                        use_cache=use_cache,
                        token_ids=input_ids,
                        key_padding_mask=attention_mask,
                    )
                    if presents is not None:
                        presents.append(present_kv)

                if collect_loop:
                    base = x_prev.norm().clamp_min(1e-6)
                    upd = (x.detach() - x_prev).norm()
                    self.last_loop_update_norm[idx] = (upd / base).item()
                    self.last_loop_hidden_norm[idx] = base.item()

                # Accumulate router auxiliary losses (Switch balance,
                # Z-loss, sequence-wise balance). Each is set as a
                # side-effect on the MoE module during forward; the
                # wrapper normalises by the number of MoE applications.
                if block.moe.balance_loss is not None:
                    total_balance_loss = total_balance_loss + block.moe.balance_loss
                if block.moe.z_loss is not None:
                    total_z_loss = total_z_loss + block.moe.z_loss
                if block.moe.seq_balance_loss is not None:
                    total_seq_balance_loss = (
                        total_seq_balance_loss + block.moe.seq_balance_loss
                    )

            # Capture pre-norm_loop hidden for aux-loss computation.
            # Only capture non-final loops of THIS forward pass — under
            # loop dropout, the "final" loop varies per batch. The
            # hidden state at position n_loops_to_run - 1 will feed
            # the main LM head, so no aux for it.
            if capture_aux and loop < n_loops_to_run - 1:
                intermediate_hiddens.append(x)

            loop_rms.append(x.float().pow(2).mean().sqrt())
            if loop < n_loops_to_run - 1:
                x = self.norm_loop(x)

        x = self.norm_out(x)
        # Expose intermediate hiddens to the CausalLM wrapper. Set to
        # None when not capturing so downstream code can do a cheap
        # truthy check.
        self.last_intermediate_hiddens = (
            intermediate_hiddens if capture_aux else None
        )
        return (
            x, loop_rms,
            total_balance_loss, total_z_loss, total_seq_balance_loss,
            presents,
        )

    def set_moe_telemetry(self, enabled: bool) -> None:
        """Toggle per-MoE-layer telemetry calculation in forward.

        When disabled, each MoELayer skips its ~21 .item()/.tolist()
        calls — one CUDA sync each on GPU. The training loops set this
        to False on non-logging steps so MoE diagnostics are only paid
        for when actually consumed.

        Consumers (`_collect_moe_metrics`, the MoE telemetry block,
        `monitoring.moe_health`) read the `block.moe.last_*` lists; on
        disabled steps these retain the previous-step values, but no
        consumer reads them on those steps (all reads are inside
        `if should_log:` guards).
        """
        self.telemetry_enabled = enabled  # gates the loop-collapse hook
        for blk in self.blocks:
            blk.moe.telemetry_enabled = enabled


class OSRTForCausalLM(OSRTPreTrainedModel):
    """OSRT with causal LM head. HF-compatible.

    LM head is weight-tied to embeddings (via F.linear with embedding.weight).
    Saves ~50M params vs untied for 32K×1536 embedding. Matches v4.
    """

    def __init__(self, config: OSRTConfig) -> None:
        super().__init__(config)
        self.model = OSRTModel(config)

        # Multi-Token Prediction heads (ARCHITECTURE.md §9.3, §11.4). Created
        # ONLY when mtp_heads > 0 so the default (0) path is bit-identical:
        # the attribute is an empty ModuleList, adds no params, and the loss
        # block below short-circuits. Head k (0-indexed) predicts the token at
        # offset +(2+k). These are training-time-only params — never used at
        # inference (generate() ignores them) and droppable at deployment.
        self.mtp_heads = nn.ModuleList(
            [MTPHead(config.dim) for _ in range(config.mtp_heads)]
        )

        # HF's post_init walks all nn.Linear and calls _init_weights on them,
        # which would overwrite any orthogonal init done in MoELayer.__init__.
        self.post_init()
        # Apply orthogonal per-expert init AFTER post_init so it survives.
        for block in self.model.blocks:
            block.moe.apply_orthogonal_init()

        # Last-forward loss components (for training-loop logging).
        # These are plain tensors, set during forward. The training loop
        # reads them directly instead of us extending the HF ModelOutput.
        self.last_task_loss: Tensor | None = None
        self.last_balance_loss: Tensor | None = None
        self.last_balance_loss_normalised: Tensor | None = None
        self.last_z_loss: Tensor | None = None
        self.last_z_loss_normalised: Tensor | None = None
        self.last_seq_balance_loss: Tensor | None = None
        self.last_seq_balance_loss_normalised: Tensor | None = None
        # Per-loop aux losses (when aux_loop_loss_weight > 0 + training).
        # Indexed loop 0..N-2 (final loop has no aux). Each entry is the
        # raw CE loss for predicting next-token from that loop's hidden.
        self.last_per_loop_aux_losses: list[Tensor] = []
        self.last_aux_loop_total: Tensor | None = None

        # Optional compiled forward for inference (set by
        # optimize_for_inference). generate()/speculative call self._fwd, which
        # dispatches here when present so compilation is actually EXERCISED on
        # the decode path — self.forward() alone bypasses model.compile()'s
        # __call__ wrapping (dynamo compiles __call__, not a direct .forward()).
        self._compiled_forward = None
        # Separate CUDA-graph decode callable (reduce-overhead, static shapes),
        # used by _fwd ONLY for StaticKVCache steps so prefill's grouped GEMM
        # (capture-unsupported) never enters stream capture.
        self._compiled_decode = None
        # MTP telemetry (when mtp_heads > 0 + training + labels). last_mtp_loss
        # is the detached weighted sum added to the training loss; None when MTP
        # is off or the head contributed nothing. last_mtp_losses holds the
        # detached per-head raw CE values (length == config.mtp_heads).
        self.last_mtp_loss: Tensor | None = None
        self.last_mtp_losses: list[Tensor] = []

    def set_moe_telemetry(self, enabled: bool) -> None:
        """Delegate to OSRTModel.set_moe_telemetry — see docstring there."""
        self.model.set_moe_telemetry(enabled)

    def optimize_for_inference(
        self,
        compile_model: bool = True,
        reduce_overhead: bool = False,
    ) -> "OSRTForCausalLM":
        """Prepare this model for fast generation/eval. Returns self.

        Three steps (measured on midtrain3_final, A100, real generate()):
          1. set_moe_telemetry(False) — drops the ~21 .item()/.tolist() CUDA
             syncs per MoE forward (x18 effective layers). These are training
             collapse-guards; at inference they are pure overhead AND their
             .item() calls graph-break torch.compile.
          2. prepack_expert_weights() per block — value-identical; throughput-
             neutral under fullgraph compile so far (see its docstring), kept
             for the post-CUDA-graph regime. ~791 MiB extra bf16 buffers.
          3. torch.compile(self.forward, fullgraph=True, dynamic=True) into
             self._compiled_forward. MEASURED end-to-end generate(): decode
             b1 5.0 -> 24.2 tok/s, b32 153 -> 705.6, prefill 196 -> 46 ms.
             QUALITY: held-out math ppl unchanged (2.8827 == 2.8827); greedy
             free-generation is NOT bit-reproducible (bf16 fused-reduction
             reorder flips ~4% of argmax near-ties; max|Δlogit|~0.9). Accepted
             tradeoff — gate on ppl/logit error, not token identity.

        NOT applied automatically — training must keep telemetry ON and stay
        uncompiled. Call this only on a model dedicated to inference. Compiles
        lazily on the first forward of each new (batch, seq) shape; a couple of
        warmup forwards amortise it (a cold fullgraph+dynamic trace of this
        model is tens of minutes; persist the inductor cache across runs).
        `mode="reduce-overhead"` (CUDA graphs) is deliberately NOT used —
        capture fails on the MoE data-dependent ops (needs the static-KV-cache
        refactor first).

        IMPORTANT: compiles self.forward into self._compiled_forward, which
        generate()/speculative dispatch to via self._fwd. A bare self.compile()
        would NOT take effect on the decode path — dynamo wraps __call__, but
        generate() historically called self.forward() directly (bypassing it).
        A CompileCounter test guards that generate() actually compiles.

        Sets the same two dynamo flags the training loop sets (train.py, B4
        grouped-GEMM block): torch._grouped_mm's data-dependent per-expert
        offsets otherwise graph-break on .item()/.tolist() — measured at
        inference as ~432 host syncs/token (18 MoE layers x 3 grouped_mm x 8
        offsets) and ~40 graph segments. With the flags, training verified
        fullgraph compiles with 0 breaks; fullgraph=True here asserts the same
        so any future capture-blocker fails loudly instead of silently
        fragmenting the decode step.
        """
        self.eval()
        self.set_moe_telemetry(False)
        # Prepack routed-expert weights (value-identical; throughput-neutral
        # under fullgraph compile so far — see prepack_expert_weights). Call
        # AFTER load_state_dict; buffers are non-persistent, move with .to().
        for blk in self.model.blocks:
            blk.moe.prepack_expert_weights()
        if compile_model:
            import torch._dynamo as _dynamo
            _dynamo.config.capture_scalar_outputs = True
            _dynamo.config.capture_dynamic_output_shape_ops = True
            # dynamic=True: the latent cache length grows every token; without
            # it dynamo re-specializes (an expensive unbacked-symint compile)
            # per cache length — measured as a >30 min compile stall on A100.
            # One symbolic graph instead. Handles prefill + latent decode.
            self._compiled_forward = torch.compile(
                self.forward, fullgraph=True, dynamic=True,
            )
            if reduce_overhead:
                # SEPARATE CUDA-graph decode callable, used by _fwd only for
                # StaticKVCache steps (generate(cache_impl="static")). Why
                # separate: prefill contains torch._grouped_mm, whose internal
                # CUDA calls are illegal during stream capture
                # (cudaErrorStreamCaptureUnsupported) — the decode step avoids
                # it via _dispatch_bmm, but a shared reduce-overhead callable
                # would try to capture the prefill graph too and die there.
                # Requirements already in place: static shapes/addresses
                # (StaticKVCache + mark_static_address), device-side cursor,
                # and a decode path with no .item()/sync ops (telemetry off,
                # bmm dispatch) — capture_scalar_outputs=True is harmless here
                # because the flag only matters where scalar reads exist.
                self._compiled_decode = torch.compile(
                    self.forward, fullgraph=True, mode="reduce-overhead",
                )
        return self

    def _fwd(self, *args, **kwargs) -> CausalLMOutputWithPast:
        """Dispatch to the compiled forward when inference-optimized, else the
        eager forward. Decode paths (generate/speculative) call this so
        torch.compile is actually exercised — see optimize_for_inference.

        StaticKVCache steps route to the separate CUDA-graph decode callable
        when present (reduce_overhead=True): prefill's grouped GEMM is illegal
        under stream capture, so the two must not share a compiled callable."""
        if (
            self._compiled_decode is not None
            and isinstance(kwargs.get("past_key_values"), StaticKVCache)
        ):
            return self._compiled_decode(*args, **kwargs)
        if self._compiled_forward is not None:
            return self._compiled_forward(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embedding = value

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        past_key_values: list[Tensor | None] | None = None,
        use_cache: bool = False,
        num_loops: int | None = None,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Causal-LM forward.

        num_loops (ARCHITECTURE.md §12.2) is the optional variable-loop
        inference knob: None → config.recursive_loops (bit-identical to the
        historical path); K in [1, recursive_loops] runs only the first K
        recursive loops before norm_out + the (tied) LM head. The aux per-loop
        LM-head training (§9.2) is what makes a reduced K still predictive.
        It is threaded straight into OSRTModel.forward, which validates it and
        keys the KV-cache layer count (num_blocks * K) off it — see the note
        there about keeping K consistent across a cached decode.
        """
        (
            hidden, loop_rms,
            balance_loss, z_loss, seq_balance_loss,
            presents,
        ) = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            num_loops=num_loops,
            attention_mask=attention_mask,
        )

        # Weight-tied LM head
        logits = F.linear(hidden, self.model.embedding.weight)

        # Reset loss attributes (prevent stale values in eval-without-labels)
        self.last_task_loss = None
        self.last_balance_loss = None
        self.last_balance_loss_normalised = None
        self.last_z_loss = None
        self.last_z_loss_normalised = None
        self.last_seq_balance_loss = None
        self.last_seq_balance_loss_normalised = None
        self.last_per_loop_aux_losses = []
        self.last_aux_loop_total = None
        self.last_mtp_loss = None
        self.last_mtp_losses = []

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :self.config.real_vocab_size]
            shift_logits = shift_logits.contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.real_vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Normalise each aux loss by num MoE applications so the
            # coefficient matches per-layer weight (not per-whole-model sum).
            #
            # Use ACTUAL loops run (len(loop_rms)) rather than the configured
            # depth (self.config.recursive_loops). Under loop_dropout_prob > 0
            # the model truncates the loop chain to a random length and only
            # accumulates per-block losses from the loops that executed —
            # dividing by the configured full depth halves the regularizer
            # exactly on the stochastic-depth batches that need it most. The
            # OSRTModel returns loop_rms whose length always equals the number
            # of loops actually run.
            n_moe_layers = self.config.num_blocks * max(1, len(loop_rms))
            balance_norm = balance_loss / n_moe_layers
            z_norm = z_loss / n_moe_layers
            seq_balance_norm = seq_balance_loss / n_moe_layers

            # Per-loop aux LM-head losses (architecture fix). Captures the
            # hidden state at the END of each non-final loop, applies
            # norm_out + the (weight-tied) LM head, and computes CE
            # against the same shifted labels. Adds gradient signal to
            # intermediate loops 0..N-2.
            #
            # Per-loop weighting: if config.per_loop_aux_weights is set
            # (list of length n_intermediate_loops), each loop's CE is
            # multiplied by its weight before summing. Otherwise all
            # intermediate losses use the uniform aux_loop_loss_weight.
            aux_loop_total = torch.tensor(0.0, device=task_loss.device)
            per_loop_aux: list[Tensor] = []
            intermediate_hiddens = self.model.last_intermediate_hiddens
            aux_weight = getattr(self.config, "aux_loop_loss_weight", 0.0)
            per_loop_weights = getattr(
                self.config, "per_loop_aux_weights", None,
            )
            # Fused chunked linear-CE (memory). 0 = off (bit-identical to the
            # F.linear+CE path below); > 0 routes the aux/MTP head losses
            # through osrt.fused_ce so only ~1/chunks of the (N, vocab) logits
            # live at once. Same loss + gradients — tests/test_fused_ce.py.
            fused_ce_chunks = getattr(
                self.config, "fused_cross_entropy_chunks", 0,
            )
            if (
                self.training
                and aux_weight > 0.0
                and intermediate_hiddens
            ):
                for i, h_loop in enumerate(intermediate_hiddens):
                    h_norm = self.model.norm_out(h_loop)
                    if fused_ce_chunks > 0:
                        aux_l = fused_linear_cross_entropy(
                            h_norm[:, :-1, :].reshape(-1, h_norm.shape[-1]),
                            self.model.embedding.weight,
                            shift_labels.reshape(-1),
                            real_vocab_size=self.config.real_vocab_size,
                            ignore_index=-100,
                            n_chunks=fused_ce_chunks,
                        )
                    else:
                        h_logits = F.linear(
                            h_norm, self.model.embedding.weight,
                        )
                        h_shift = h_logits[
                            ..., :-1, :self.config.real_vocab_size
                        ].contiguous().float()
                        aux_l = F.cross_entropy(
                            h_shift.view(-1, self.config.real_vocab_size),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        )
                    per_loop_aux.append(aux_l)
                    # Per-loop scaling. When per_loop_weights is set,
                    # use it (must be at least as long as the captured
                    # intermediates); else uniform weight 1.0 (then
                    # the overall aux_weight multiplies the sum).
                    if (
                        per_loop_weights is not None
                        and i < len(per_loop_weights)
                    ):
                        w = per_loop_weights[i]
                    else:
                        w = 1.0
                    aux_loop_total = aux_loop_total + w * aux_l

            # Multi-Token Prediction loss (ARCHITECTURE.md §9.3, §11.4). For
            # head k = 1..mtp_heads, predict the token at offset +(1+k) from the
            # FINAL hidden state `hidden` (post-norm_out, the same state feeding
            # the main +1 LM head). Each head applies its small RMSNorm+Linear
            # projection, then the WEIGHT-TIED embedding (via F.linear) turns it
            # into vocab logits. The targets are `labels` shifted by (1+k); the
            # tail positions that run off the sequence end are dropped (we slice
            # logits to [:, :T-(1+k), :] and labels to [:, (1+k):]). CE is
            # computed in fp32 with ignore_index=-100 to match the main loss.
            # These MTP-head params are TRAINING-TIME ONLY (droppable at
            # deployment); the loss is added to the total only when training.
            mtp_total = torch.tensor(0.0, device=task_loss.device)
            per_mtp: list[Tensor] = []
            if self.training and len(self.mtp_heads) > 0:
                seq_len = labels.shape[-1]
                for k, head in enumerate(self.mtp_heads):
                    offset = k + 2  # head k (0-indexed) → future offset +(2+k)
                    # Skip heads whose target offset runs entirely off the
                    # end of this (short) sequence — no positions to predict.
                    if offset >= seq_len:
                        per_mtp.append(
                            torch.tensor(0.0, device=task_loss.device)
                        )
                        continue
                    head_hidden = head(hidden)
                    if fused_ce_chunks > 0:
                        # Fused chunked linear-CE (memory): same loss/grads.
                        # Logits at position i predict token at i+offset, so the
                        # last `offset` positions have no in-range target → drop.
                        mtp_l = fused_linear_cross_entropy(
                            head_hidden[:, :-offset, :].reshape(
                                -1, head_hidden.shape[-1],
                            ),
                            self.model.embedding.weight,
                            labels[..., offset:].reshape(-1),
                            real_vocab_size=self.config.real_vocab_size,
                            ignore_index=-100,
                            n_chunks=fused_ce_chunks,
                        )
                    else:
                        head_logits = F.linear(
                            head_hidden, self.model.embedding.weight,
                        )
                        # Logits at position i predict token at i+offset, so the
                        # last `offset` positions have no in-range target → drop.
                        m_shift_logits = head_logits[
                            ..., :-offset, :self.config.real_vocab_size
                        ].contiguous().float()
                        m_shift_labels = labels[..., offset:].contiguous()
                        mtp_l = F.cross_entropy(
                            m_shift_logits.view(-1, self.config.real_vocab_size),
                            m_shift_labels.view(-1),
                            ignore_index=-100,
                        )
                    per_mtp.append(mtp_l)
                    mtp_total = mtp_total + mtp_l

            # Total loss: add aux losses ONLY during training. Eval loss must
            # be pure task CE so eval perplexity and held-out comparisons
            # aren't polluted by hyperparameter choices. Training loops that
            # want the aux signals at eval time should read the
            # last_*_normalised attributes instead.
            if self.training:
                loss = (
                    task_loss
                    + self.config.router_aux_loss_coeff * balance_norm
                    + self.config.router_z_loss_coeff * z_norm
                    + self.config.router_seq_balance_loss_coeff
                    * seq_balance_norm
                    + aux_weight * aux_loop_total
                    + self.config.mtp_loss_weight * mtp_total
                )
            else:
                loss = task_loss

            # Stash per-loop aux losses (detached) for telemetry.
            self.last_per_loop_aux_losses = [t.detach() for t in per_loop_aux]
            self.last_aux_loop_total = (
                aux_loop_total.detach() if per_loop_aux else None
            )

            # Stash MTP losses (detached) for telemetry. last_mtp_loss is the
            # weighted sum actually added to the training loss; None when MTP is
            # off or contributed nothing. last_mtp_losses holds the per-head raw
            # (unweighted) CE values, one per head (length == config.mtp_heads).
            self.last_mtp_losses = [t.detach() for t in per_mtp]
            self.last_mtp_loss = (
                (self.config.mtp_loss_weight * mtp_total).detach()
                if per_mtp else None
            )

            # Expose components for logging — always set, regardless of mode.
            self.last_task_loss = task_loss.detach()
            self.last_balance_loss = balance_loss.detach()
            self.last_balance_loss_normalised = balance_norm.detach()
            self.last_z_loss = z_loss.detach()
            self.last_z_loss_normalised = z_norm.detach()
            self.last_seq_balance_loss = seq_balance_loss.detach()
            self.last_seq_balance_loss_normalised = seq_balance_norm.detach()

        # Cast to FloatTensor to satisfy HF's type stubs. The runtime types
        # are correct — logits comes from F.linear on a float hidden state,
        # and loss is a scalar float tensor from cross_entropy.
        return CausalLMOutputWithPast(
            loss=cast("torch.FloatTensor | None", loss),
            logits=cast("torch.FloatTensor", logits),
            past_key_values=presents,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 512,
        attention_mask: Tensor | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        eos_token_id: int | None = None,
        stop_token_ids: list[int] | None = None,
        repetition_penalty: float = 1.0,
        num_loops: int | None = None,
        speculative: bool = False,
        spec_draft_tokens: int = 4,
        cache_impl: str = "latent",
        **kwargs,
    ) -> Tensor:
        """Autoregressive generation with KV cache.

        cache_impl="static" switches decode to the StaticKVCache "speed mode":
        prefill runs the normal latent path once, is converted into fixed
        post-RoPE K/V buffers, and every decode step then touches only the new
        token (no cat, no history recompute, static shapes — CUDA-graph-ready).
        ~2x cache memory; per-position math identical to "latent" (bf16
        accumulation order differs — gate with ppl, not token identity).
        Unsupported with attention_mask (left-padded batches) or speculative.

        The model's forward already supports past_key_values + use_cache
        (per-effective-layer KV cache, 18 layers for default v5). This
        method does the prefill + decode loop: one full forward over the
        prompt to seed the cache, then one single-token forward per step
        consuming the cache. That turns O(N) per-step attention cost into
        O(1) and is ~3x faster than the non-cached path for a 256-token
        generation on this architecture.

        Defaults are IFEval-safe (greedy, no repetition penalty). Pass
        temperature>0 to sample; top_p < 1 and top_k > 0 gate the
        sample population. Sampling reuses the standard top-k then
        top-p nucleus filtering pattern.

        Caller is expected to set the model to eval mode if they want
        KV drops disabled at the MoE layer — the training vs inference
        switch happens in MoELayer.forward via self.training, which
        .train(False) toggles.

        num_loops (ARCHITECTURE.md §12.2) selects the variable inference
        loop count: None → config.recursive_loops (full quality, default,
        bit-identical to before); K in [1, recursive_loops] runs only the
        first K loops at every step for a speed/quality trade-off. The SAME
        K is threaded through both the prefill and every decode forward, so
        the MLA cache (num_blocks * K latents per token) stays index-consistent
        across the whole call — mixing loop counts mid-generation would
        misalign the per-effective-layer cache and is therefore not allowed.

        speculative (ARCHITECTURE.md §12.3) gates a greedy speculative-decode
        fast path: draft spec_draft_tokens tokens cheaply at
        config.spec_draft_loops loops, then verify them in ONE forward at the
        full loop count, committing the longest matching greedy prefix plus the
        verifier's correction. Default False keeps the standard decode loop
        below bit-identical. The verifier loop count is the full
        config.recursive_loops (or `num_loops` when set, which also caps the
        draft loops). See _generate_speculative for the (documented,
        NOT distribution-preserving) acceptance rule.
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        # Speculative decoding (§12.3) is a separate, self-contained decode
        # loop. It is a GREEDY throughput optimization, so it only makes sense
        # at temperature 0 (or very low temp); we route to it only when asked.
        if speculative:
            # Speculative decoding is a GREEDY throughput path; it does not
            # implement sampling or padded-batch masking. Reject those args
            # rather than silently ignoring them (finding #6).
            if attention_mask is not None:
                raise ValueError(
                    "speculative=True does not support attention_mask "
                    "(left-padded batches)."
                )
            if temperature > 0 or top_p < 1.0 or top_k > 0:
                raise ValueError(
                    "speculative=True is greedy-only; temperature/top_p/top_k "
                    "are not supported."
                )
            return self._generate_speculative(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                stop_token_ids=stop_token_ids,
                repetition_penalty=repetition_penalty,
                num_loops=num_loops,
                spec_draft_tokens=spec_draft_tokens,
            )

        # Prefill over the prompt, keeping KV cache for decode.
        # HF's CausalLMOutputWithPast types past_key_values as
        # `Cache | None`, but our forward returns a plain list of
        # per-layer latent tensors. Cast locally so ty/mypy line up.
        PastKV = list[Tensor | None]
        context = input_ids[:, -self.config.max_position_embeddings:]
        # Running key-padding mask over the full cached span. Truncated to match
        # the (possibly truncated) prompt context, then extended by 1 (real)
        # per decode step. None → no padding → existing fast paths, bit-identical.
        attn = None
        if attention_mask is not None:
            attn = attention_mask[:, -self.config.max_position_embeddings:]
        out = self._fwd(
            context, use_cache=True, num_loops=num_loops, attention_mask=attn,
        )
        past_key_values = cast("PastKV | None", out.past_key_values)

        # cache_impl="static": convert the prefill latents into fixed K/V
        # buffers once; decode steps then run _attention_static against them.
        static_cache: StaticKVCache | None = None
        if cache_impl == "static":
            if attn is not None:
                raise ValueError(
                    "cache_impl='static' does not support attention_mask "
                    "(left-padded batches)."
                )
            latents = cast("list[Tensor]", past_key_values)
            B, prompt_len = context.shape
            max_len = min(
                self.config.max_position_embeddings,
                prompt_len + max_new_tokens,
            )
            mdl = self.model
            static_cache = StaticKVCache(
                num_layers=len(latents), batch=B,
                kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim,
                max_len=max_len, device=context.device,
                dtype=torch.bfloat16 if context.is_cuda else torch.float32,
            )
            cos = mdl.rope_cos[:, :max_len].to(latents[0].dtype)
            sin = mdl.rope_sin[:, :max_len].to(latents[0].dtype)
            for idx, c_kv in enumerate(latents):
                blk = mdl.blocks[idx % self.config.num_blocks]
                blk.write_latent_to_static(
                    c_kv, cos, sin, static_cache.k[idx], static_cache.v[idx],
                )
            static_cache.cursor.fill_(prompt_len)
            past_key_values = None  # latents no longer needed

        # Precompute stop tensor if any
        stop_tensor = None
        if stop_token_ids:
            stop_tensor = torch.tensor(list(stop_token_ids), device=input_ids.device)

        # Per-row finished mask. A row is "finished" once it has ever
        # emitted eos_token_id on any decode step. Once finished, we
        # overwrite its next-token with EOS so downstream callers can
        # cleanly truncate, and we stop updating logits for it.
        batch_size = input_ids.shape[0]
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=input_ids.device,
        )
        logits_tensor = cast(Tensor, out.logits)
        logits_last = (
            logits_tensor[:, -1, :self.config.real_vocab_size].float()
        )
        # Preallocate the output buffer instead of repeatedly torch.cat-ing
        # a 1-token column onto a growing tensor. The old pattern paid
        # O(prompt + step) memory bandwidth on EVERY decode step. With a
        # 400-token rollout that's ~80,000 copied positions vs the new
        # cost of one preallocation + in-place writes. Cursor tracks the
        # next-write position; we slice generated[:, :cursor] for
        # repetition-penalty / return so the unwritten tail (zero-filled)
        # never leaks into observable output.
        total_len = input_ids.shape[1] + max_new_tokens
        generated = torch.zeros(
            batch_size, total_len,
            dtype=input_ids.dtype, device=input_ids.device,
        )
        generated[:, :input_ids.shape[1]] = input_ids
        cursor = input_ids.shape[1]

        for step_idx in range(max_new_tokens):
            if step_idx > 0:
                # Decode: pass only the newest token + existing cache.
                new_tok = generated[:, cursor - 1:cursor]
                # Don't trim past_key_values when the cache exceeds
                # max_position_embeddings — left-truncating the cache
                # shifts the absolute RoPE indices that the forward
                # derives from past_key_values[idx].shape[1] (the
                # past_length read in OSRTModel.forward), so cached K
                # (rotated at original absolute positions) and the new K
                # (rotated at the post-trim shifted index) end up in
                # different positional bases and attention breaks.
                # The forward already handles required_seq_len > the
                # precomputed RoPE range by recomputing cos/sin on demand
                # (the else-branch of the rope_cos slice), so letting the
                # cache grow naturally is safe. Memory cost grows with generation
                # length; if that becomes a constraint, the right fix
                # is sliding-window with re-rotation, not a naive trim.
                if attn is not None:
                    # New token is always real → append a 1 to the mask so the
                    # cached pad columns stay masked as the span grows.
                    attn = torch.cat(
                        [attn, torch.ones(
                            batch_size, 1, dtype=attn.dtype, device=attn.device,
                        )],
                        dim=1,
                    )
                if static_cache is not None:
                    out = self._fwd(
                        new_tok,
                        past_key_values=static_cache,
                        use_cache=False,
                        num_loops=num_loops,
                    )
                    static_cache.advance()
                else:
                    out = self._fwd(
                        new_tok,
                        past_key_values=past_key_values,
                        use_cache=True,
                        num_loops=num_loops,
                        attention_mask=attn,
                    )
                    past_key_values = cast("PastKV | None", out.past_key_values)
                logits_tensor = cast(Tensor, out.logits)
                logits_last = (
                    logits_tensor[:, -1, :self.config.real_vocab_size].float()
                )

            # Repetition penalty (disabled by default). Vectorised so the
            # cost stays O(B*T) on-device instead of O(B*|set|) Python-loop
            # overhead per decode step. gather pulls each previously-seen
            # token's current logit, scales it, and scatters back. When a
            # token id appears multiple times in `generated[b]` the same
            # scaled value is written for every occurrence, so last-write-
            # wins on scatter is safe — semantically identical to the
            # original "apply once per unique id" loop.
            if repetition_penalty != 1.0:
                vocab = logits_last.shape[-1]
                # Slice to the actually-written portion — the preallocated
                # tail (zeros) would otherwise penalise token 0 every step.
                already = generated[:, :cursor]
                gen_clamped = already.clamp(max=vocab - 1)
                in_vocab = (already < vocab)
                score = torch.gather(logits_last, 1, gen_clamped)
                penalised = torch.where(
                    score > 0,
                    score / repetition_penalty,
                    score * repetition_penalty,
                )
                # Where the gathered position was out-of-vocab, write the
                # original score back (no-op) so scatter doesn't corrupt
                # in-vocab logits with garbage from clamped indices.
                penalised = torch.where(in_vocab, penalised, score)
                logits_last = logits_last.clone()
                logits_last.scatter_(1, gen_clamped, penalised)

            if temperature > 0:
                next_logits = logits_last / temperature
                if top_k > 0:
                    topk_vals, _ = torch.topk(
                        next_logits, min(top_k, next_logits.size(-1)),
                    )
                    next_logits[
                        next_logits < topk_vals[:, -1:]
                    ] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True,
                    )
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_mask = cumprobs - sorted_probs >= top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    next_logits.scatter_(1, sorted_indices, sorted_logits)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits_last.argmax(dim=-1, keepdim=True)

            # Force already-finished rows to keep emitting EOS so the
            # tensor stays rectangular, their completion is stable, and
            # we stop polluting them with extra tokens.
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )

            # In-place write into the preallocated buffer (no growing
            # tensor copy). next_token is (B, 1); generated[:, c:c+1] is
            # the matching slice — copy_ keeps the dtype/layout sane.
            generated[:, cursor:cursor + 1].copy_(next_token)
            cursor += 1

            # Per-row termination. A row is finished once it has EVER
            # emitted EOS or any stop_token_id — we track that in
            # `finished` and break only when every row has finished at
            # some point, not only when all rows happen to emit a stop
            # token on the same step.
            #
            # stop_token_ids/stop_tensor lets callers stop on chat-template markers
            # like <|/answer|> (token 10) or <|user|> (token 11). Useful
            # because MOPD-distilled models often generate additional
            # answer blocks after the first one or try to start a new
            # user turn — stopping on those keeps inference output clean.
            nt = next_token.squeeze(-1)
            if eos_token_id is not None:
                finished = finished | (nt == eos_token_id)
            if stop_tensor is not None:
                finished = finished | torch.isin(nt, stop_tensor)
            if ((eos_token_id is not None or stop_tensor is not None)
                    and bool(finished.all())):
                break

        # Slice to actually-written length so the preallocated tail
        # (zeros) never leaks into callers.
        return generated[:, :cursor]

    @torch.no_grad()
    def _generate_speculative(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: int | None,
        stop_token_ids: list[int] | None,
        repetition_penalty: float,
        num_loops: int | None,
        spec_draft_tokens: int,
    ) -> Tensor:
        """Greedy speculative decoding via a low-loop draft (§12.3).

        ┌──────────────────────────────────────────────────────────────────┐
        │ GREEDY-ONLY / NOT DISTRIBUTION-PRESERVING.                         │
        │                                                                    │
        │ Real speculative *sampling* (Leviathan et al. 2023, Chen et al.    │
        │ 2023) accepts/rejects each drafted token with a probabilistic      │
        │ correction so the emitted sequence is distributed EXACTLY as       │
        │ sampling from the verifier alone. This routine instead accepts the │
        │ longest GREEDY-matching prefix and emits the verifier's greedy     │
        │ argmax on the first mismatch. That is provably identical to plain  │
        │ greedy (temperature 0) decoding from the verifier, but it is NOT a │
        │ valid sampler for temperature > 0 — it is a throughput trick for   │
        │ greedy / very-low-temperature generation only. generate() routes   │
        │ here only when the caller passes speculative=True.                 │
        └──────────────────────────────────────────────────────────────────┘

        Mechanism. The verifier runs the full loop count (num_loops or
        config.recursive_loops); the drafter runs config.spec_draft_loops
        loops, which the aux per-loop LM-head training (§9.2) makes predictive.
        Each round:
          1. DRAFT — autoregressively greedy-decode `D = spec_draft_tokens`
             tokens one at a time at the cheap draft loop count, advancing a
             draft-side MLA cache.
          2. VERIFY — run ONE full-loop forward over the D drafted tokens to
             get the verifier's greedy prediction at each drafted position in
             parallel.
          3. COMMIT — accept a draft while it equals the verifier's greedy
             token. On the first mismatch emit the verifier's token and stop;
             if all D match, additionally emit the verifier's bonus token for
             the position after the block (a free extra token — the hallmark
             speculative speed-up).

        Cache handling. Two independent MLA caches are kept because the cache
        is loop-count-specific (num_blocks * loops latents per token): a draft
        cache at the draft loop count and a verify cache at the full loop
        count. Each is a list of (B, seq, kv_dim) latent tensors, so it can be
        TRUNCATED along the sequence axis. The verifier forward over the draft
        block advances the verify cache by D tokens; the accepted-prefix
        latents are exactly correct (an accepted draft equals the verifier's
        prediction, i.e. the actually-committed token), so we keep them and
        slice off the stale tail past the acceptance point. MoE capacity drops
        are disabled in eval (MoELayer.forward), so the one-shot parallel
        verify reproduces what a token-by-token decode would have produced and
        the acceptance check is exact.
        """
        PastKV = list[Tensor | None]

        def _trunc(past: "PastKV | None", length: int) -> "PastKV | None":
            """Slice every per-layer latent to the first `length` positions
            along the sequence axis (dim=1). The MLA cache is plain tensors,
            so a stale speculative tail is just a view away from being dropped."""
            if past is None or length <= 0:
                return None
            return [
                None if p is None else p[:, :length, :] for p in past
            ]

        # The drafter may not exceed the verifier's loop count. When the caller
        # caps the verifier via num_loops, the draft loop count is min(draft,
        # cap) so the draft never runs MORE compute than the verifier.
        full_loops = self.model._resolve_num_loops(num_loops)
        draft_loops = min(self.config.spec_draft_loops, full_loops)

        context = input_ids[:, -self.config.max_position_embeddings:]
        batch_size = context.shape[0]
        device = context.device
        D = spec_draft_tokens

        # Precompute stop tensor if any
        stop_tensor = None
        if stop_token_ids:
            stop_tensor = torch.tensor(list(stop_token_ids), device=device)

        def _greedy(logits_row: Tensor, gen: Tensor) -> Tensor:
            """Greedy next-token from a (B, vocab) logit row, with the same
            optional repetition penalty math as generate() so the two paths
            agree token-for-token at temperature 0."""
            logits_last = logits_row[:, :self.config.real_vocab_size].float()
            if repetition_penalty != 1.0:
                vocab = logits_last.shape[-1]
                gen_clamped = gen.clamp(max=vocab - 1)
                in_vocab = (gen < vocab)
                score = torch.gather(logits_last, 1, gen_clamped)
                penalised = torch.where(
                    score > 0,
                    score / repetition_penalty,
                    score * repetition_penalty,
                )
                penalised = torch.where(in_vocab, penalised, score)
                logits_last = logits_last.clone()
                logits_last.scatter_(1, gen_clamped, penalised)
            return logits_last.argmax(dim=-1, keepdim=True)  # (B, 1)

        # CACHE INVARIANT (matches the standard decode loop above):
        #   at the TOP of every round, verify_past / draft_past each cover the
        #   first `cache_len` positions of `generated`, where
        #       cache_len == generated.shape[1] - 1
        #   i.e. the cache holds every committed token EXCEPT the most recent
        #   one, which is the pending input for the round. We prefill over
        #   context[:, :-1] and keep the final prompt token uncached to
        #   establish that invariant.
        prompt_len = context.shape[1]
        generated = context.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if prompt_len > 1:
            seed = context[:, :-1]
            v_out = self._fwd(seed, use_cache=True, num_loops=full_loops)
            verify_past = cast("PastKV | None", v_out.past_key_values)
            d_out = self._fwd(seed, use_cache=True, num_loops=draft_loops)
            draft_past = cast("PastKV | None", d_out.past_key_values)
            cache_len = prompt_len - 1
        else:
            # A length-1 prompt has nothing to pre-cache; both caches start
            # empty and the single token is the first pending input.
            verify_past = None
            draft_past = None
            cache_len = 0

        produced = 0
        while produced < max_new_tokens:
            pending = generated[:, -1:]  # the one uncached committed token

            # ── 1. DRAFT D tokens at the cheap loop count ──
            # Each draft forward feeds one token + the draft cache. After the
            # block, draft_past covers cache_len + D positions: the pending
            # token plus drafts 0..D-2 (drafts[D-1] is the last OUTPUT, not yet
            # an input).
            draft_input = pending
            running = generated
            draft_tokens: list[Tensor] = []
            for _ in range(D):
                dd = self._fwd(
                    draft_input, past_key_values=draft_past,
                    use_cache=True, num_loops=draft_loops,
                )
                draft_past = cast("PastKV | None", dd.past_key_values)
                dtok = _greedy(cast(Tensor, dd.logits)[:, -1, :], running)
                draft_tokens.append(dtok)
                running = torch.cat([running, dtok], dim=1)
                draft_input = dtok
            drafts = torch.cat(draft_tokens, dim=1)  # (B, D)

            # ── 2. VERIFY all D drafts in ONE full-loop forward ──
            # Input is [pending, draft_0, ..., draft_{D-1}] (D + 1 tokens). The
            # verifier's logits at input position i predict the token that
            # follows the i-th input, so:
            #   v_logits[:, i]   for i in 0..D-1  → the verifier's greedy
            #                    proposal for the SAME slot drafts[:, i] fills.
            #   v_logits[:, D]   (fed draft_{D-1}) → the BONUS token that
            #                    follows the whole accepted block.
            # The forward grows verify_past by D + 1 positions; we keep only the
            # accepted prefix's latents (drop the speculative tail).
            verify_input = torch.cat([pending, drafts], dim=1)  # (B, D+1)
            vv = self._fwd(
                verify_input, past_key_values=verify_past,
                use_cache=True, num_loops=full_loops,
            )
            verify_past_full = cast("PastKV | None", vv.past_key_values)
            v_logits = cast(Tensor, vv.logits)  # (B, D+1, vocab)

            # verify_preds[:, i] verifies drafts[:, i]; bonus is the D-th.
            if repetition_penalty == 1.0:
                verify_preds = v_logits[
                    :, :D + 1, :self.config.real_vocab_size].float().argmax(dim=-1)
            else:
                verify_preds_list = []
                running = generated
                for i in range(D + 1):
                    vt = _greedy(v_logits[:, i, :], running)
                    verify_preds_list.append(vt)
                    running = torch.cat([running, vt], dim=1)
                verify_preds = torch.cat(verify_preds_list, dim=1)  # (B, D+1)

            # ── 3. COMMIT longest greedy-matching prefix + 1 correction ──
            # accept = number of leading positions where the draft matches the
            # verifier across ALL rows (conservative but correct for B > 1).
            all_match = (drafts == verify_preds[:, :D]).all(dim=0)  # (D,)
            mismatches = (~all_match).nonzero(as_tuple=True)[0]
            accept = int(mismatches[0].item()) if mismatches.numel() > 0 else D

            new_cols: list[Tensor] = [drafts[:, i:i + 1] for i in range(accept)]
            # On a mismatch emit the verifier's correction at slot `accept`; on
            # a full accept emit the verifier's bonus token at slot D. Both come
            # straight from verify_preds (which has D + 1 columns).
            new_cols.append(verify_preds[:, accept:accept + 1])

            # Slice to budget limits
            limit = max_new_tokens - produced
            if len(new_cols) > limit:
                new_cols = new_cols[:limit]

            # Mask finished rows so they keep emitting EOS (rectangular tensor).
            masked_cols = []
            for col in new_cols:
                if eos_token_id is not None:
                    col = torch.where(finished.unsqueeze(-1),
                                      torch.full_like(col, eos_token_id), col)
                masked_cols.append(col)

                t = col.squeeze(-1)
                if eos_token_id is not None:
                    finished = finished | (t == eos_token_id)
                if stop_tensor is not None:
                    finished = finished | torch.isin(t, stop_tensor)

            if masked_cols:
                generated = torch.cat([generated, *masked_cols], dim=1)
                produced += len(masked_cols)

            # Re-establish the cache invariant by truncation alone (no extra
            # forward needed). Both forwards fed inputs through sequence
            # position cache_len + D (verify) / cache_len + D - 1 (draft); an
            # accepted draft equals the verifier's prediction, so the latents
            # the forwards produced for the pending token and drafts 0..accept-1
            # are exactly the committed-token latents. We keep cache positions
            # 0 .. cache_len + accept (length cache_len + accept + 1) and drop
            # the stale speculative tail. The final emitted token (correction or
            # bonus) is deliberately left OUT of the cache — it becomes the next
            # round's pending input. The verify path fed D + 1 inputs (it
            # includes draft_{D-1}), so even a full-accept round has the last
            # accepted draft's latent already cached; the draft path fed only D
            # inputs, so on a full accept its last accepted latent is missing
            # and a single extend re-adds it (rare path).
            keep = cache_len + accept + 1
            verify_past = _trunc(verify_past_full, keep)
            draft_past = _trunc(draft_past, min(keep, cache_len + D))

            # The verify cache is now correct up to `keep`. Bring the draft
            # cache to the same length when a full accept left it one short
            # (draft_{D-1} was never fed as a draft input). cache_len becomes
            # the number of cached positions = keep.
            if draft_past is not None:
                d_have = draft_past[0].shape[1] if draft_past[0] is not None else 0
            else:
                d_have = 0
            if d_have < keep:
                ext_toks = generated[:, d_have:keep]
                de = self._fwd(
                    ext_toks, past_key_values=draft_past,
                    use_cache=True, num_loops=draft_loops,
                )
                draft_past = cast("PastKV | None", de.past_key_values)
            cache_len = keep

            if produced >= max_new_tokens:
                break
            if (
                (eos_token_id is not None or stop_tensor is not None)
                and bool(finished.all())
            ):
                break

        # An all-accept round can overshoot the budget by one (the bonus
        # token); trim to exactly context + max_new_tokens.
        max_len = context.shape[1] + max_new_tokens
        return generated[:, :max_len]


# ── HF auto-class registration ───────────────────────────────────────────
# Lets AutoConfig / AutoModelForCausalLM recognise model_type "osrt" whenever
# the osrt package is imported, so `AutoModelForCausalLM.from_pretrained(dir)`
# and `.from_config(cfg)` work without naming the class. register_for_auto_class
# additionally makes save_pretrained write the auto_map into config.json (the
# hook trust_remote_code loading reads). Full trust_remote_code loading WITHOUT
# the package installed also needs self-contained modeling/config files in the
# repo (osrt's cross-module imports — fused_ce, hra, muon — aren't auto-copied);
# that consolidation is a separate follow-up. With the package installed, the
# Auto* path here is fully functional.
try:
    AutoConfig.register("osrt", OSRTConfig)
    AutoModelForCausalLM.register(OSRTConfig, OSRTForCausalLM)
except ValueError:
    pass  # already registered (module re-imported)

OSRTConfig.register_for_auto_class()
OSRTForCausalLM.register_for_auto_class("AutoModelForCausalLM")

