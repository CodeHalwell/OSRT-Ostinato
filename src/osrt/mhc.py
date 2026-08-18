"""Manifold-Constrained Hyper-Connections (mHC).

Replaces the standard residual with a learned, per-token mix over an
`n_hc`-channel residual stream (ARCHITECTURE.md §8). For each sub-block:

    X_{l+1} = B_l @ X_l  +  C_l ⊗ F_l(A_l · X_l)

where A_l (channel→layer-input mix, σ-bounded), C_l (layer-output→channels,
2σ-bounded) and B_l (channel→channel residual mix) are generated dynamically
from the residual stream. B_l is projected onto the Birkhoff polytope (doubly
stochastic) by Sinkhorn-Knopp, which guarantees ‖B_l‖₂ ≤ 1 — the residual
transform is non-expansive, so the 18 effective layers stay numerically stable.

The generator matrices are owned per sub-block and SHARED across loop
iterations (one mHC instance per attention sub-block, one per MoE sub-block,
per physical block).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def sinkhorn_doubly_stochastic(logits: Tensor, iters: int) -> Tensor:
    """Project per-token n×n logit matrices onto the Birkhoff polytope.

    logits: (..., n, n). Returns a non-negative matrix whose rows and columns
    each sum to ~1 (doubly stochastic), hence spectral norm ≤ 1.

    Runs in the LOG domain (alternating logsumexp normalization) — the naive
    exp-then-divide form produces exploding gradients through the 20 iterations
    and drives training to NaN. Log-domain Sinkhorn is the stable standard.
    """
    log_m = logits
    for _ in range(iters):
        log_m = log_m - torch.logsumexp(log_m, dim=-1, keepdim=True)  # rows
        log_m = log_m - torch.logsumexp(log_m, dim=-2, keepdim=True)  # cols
    return log_m.exp()


class ManifoldHyperConnection(nn.Module):
    """Dynamic hyper-connection for one sub-block (shared across loops)."""

    def __init__(
        self,
        dim: int,
        n_hc: int = 4,
        sinkhorn_iters: int = 20,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_hc = n_hc
        self.sinkhorn_iters = sinkhorn_iters
        flat = n_hc * dim

        # Dynamic generators (2D → Muon). Operate on the flattened, normalized
        # residual stream.
        self.norm = nn.RMSNorm(flat)
        self.w_pre = nn.Linear(flat, n_hc, bias=False)
        self.w_res = nn.Linear(flat, n_hc * n_hc, bias=False)
        self.w_post = nn.Linear(flat, n_hc, bias=False)

        # Static biases. S_res initialized to identity so that at step 0 (with
        # small alphas and zero-ish dynamic term) B ≈ identity — the stream
        # starts as a near-standard residual, then learns to mix.
        self.s_pre = nn.Parameter(torch.zeros(n_hc))
        self.s_res = nn.Parameter(torch.eye(n_hc).reshape(-1) * 4.0)
        self.s_post = nn.Parameter(torch.zeros(n_hc))

        # Learnable gates on the dynamic component, initialized small.
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))

    def generate(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """X: (B, S, n_hc, dim) → (A, B, C).

        A: (B, S, n_hc) ∈ [0,1]      channel → layer-input mix
        B: (B, S, n_hc, n_hc)        doubly-stochastic residual mix
        C: (B, S, n_hc) ∈ [0,2]      layer-output → channel mix
        """
        b, s, c, d = X.shape
        flat = self.norm(X.reshape(b, s, c * d))
        a = torch.sigmoid(self.alpha_pre * self.w_pre(flat) + self.s_pre)
        c_out = 2.0 * torch.sigmoid(self.alpha_post * self.w_post(flat) + self.s_post)
        b_raw = (self.alpha_res * self.w_res(flat) + self.s_res).reshape(b, s, c, c)
        b_mat = sinkhorn_doubly_stochastic(b_raw, self.sinkhorn_iters)
        return a, b_mat, c_out

    @staticmethod
    def input_view(X: Tensor, a: Tensor) -> Tensor:
        """Collapse the channel stream to one layer input: Σ_c a_c · X_c."""
        return torch.einsum("bsc,bscd->bsd", a, X)

    @staticmethod
    def update(X: Tensor, b_mat: Tensor, c_out: Tensor, f_out: Tensor) -> Tensor:
        """Residual update: B @ X + C ⊗ f_out."""
        mixed = torch.einsum("bsij,bsjd->bsid", b_mat, X)
        contrib = c_out.unsqueeze(-1) * f_out.unsqueeze(-2)  # (B,S,n_hc,dim)
        return mixed + contrib
