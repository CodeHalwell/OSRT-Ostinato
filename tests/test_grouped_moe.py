"""B4 — grouped-GEMM MoE dispatch.

The learned top-2 dispatch can run two ways:
  • loop path (default): per-expert `.nonzero()` gather + index_add. Correct,
    but the data-dependent `.nonzero()` graph-breaks torch.compile (measured:
    it is the ONLY break in the model — removing it → fullgraph).
  • grouped path (moe_grouped_gemm=True): sort token-expert pairs by expert,
    one grouped GEMM across all experts (torch._grouped_mm on CUDA, a loop-of-
    matmuls reference on CPU), scatter back. Dropless by construction.

Parity is provable only in the NO-DROP regime (eval mode → deterministic
routing + capacity = N*top_k). In training the loop path drops tokens
(randperm) and the grouped path is dropless, so they legitimately diverge —
the training-path acceptance gate is the Modal loss trajectory tracking the
loop baseline, not these tests.

torch._grouped_mm's BACKWARD is CUDA-only (CPU kernel is broken in 2.10), so
local gradient tests use the reference primitive; the kernel's own backward is
covered by a CUDA-gated test + the Modal sanity run.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_model import tiny_config  # noqa: E402

from osrt.model import MoELayer, _ref_grouped_mm  # noqa: E402


# Aligned dims so torch._grouped_mm's 16-byte stride constraint holds in fp32
# (dim, hidden both multiples of 4). tiny_config: dim=128, expert_hidden→64.
def _cfg(**over):
    base = dict(num_routed_experts=8, top_k_experts=2, expert_hidden=64)
    base.update(over)
    return tiny_config(**base)


def _paired_layers(seed=0, **cfg_over):
    """Two identically-weighted MoELayers: one loop, one grouped."""
    torch.manual_seed(seed)
    loop = MoELayer(_cfg(**cfg_over)).eval()
    grouped = MoELayer(_cfg(moe_grouped_gemm=True, **cfg_over)).eval()
    grouped.load_state_dict(loop.state_dict())
    return loop, grouped


# ── reference grouped-mm primitive ──────────────────────────────────────────
def test_ref_grouped_mm_matches_manual_loop():
    torch.manual_seed(0)
    a = torch.randn(10, 8)
    b = torch.randn(3, 8, 8)
    offs = torch.tensor([3, 3 + 5, 10], dtype=torch.int32)  # sizes 3,5,2
    out = _ref_grouped_mm(a, b, offs)
    ref = torch.cat([a[0:3] @ b[0], a[3:8] @ b[1], a[8:10] @ b[2]], dim=0)
    assert torch.allclose(out, ref, atol=1e-5)


def test_ref_grouped_mm_handles_empty_group():
    torch.manual_seed(0)
    a = torch.randn(8, 8)
    b = torch.randn(2, 8, 8)
    offs = torch.tensor([0, 8], dtype=torch.int32)  # group 0 empty
    out = _ref_grouped_mm(a, b, offs)
    assert torch.allclose(out, a @ b[1], atol=1e-5)


def test_ref_grouped_mm_matches_kernel_forward():
    """The vendor kernel forward must match the reference (kernel fwd works on
    CPU; only its backward is broken). Guards against the kernel and reference
    drifting in the convention they expect for offs/layout."""
    torch.manual_seed(0)
    a = torch.randn(12, 8)
    b = torch.randn(4, 8, 8)
    offs = torch.tensor([3, 6, 9, 12], dtype=torch.int32)
    ref = _ref_grouped_mm(a, b, offs)
    kern = torch._grouped_mm(a, b, offs=offs)
    assert torch.allclose(ref, kern, atol=1e-4)


# ── flag plumbing ───────────────────────────────────────────────────────────
def test_moe_grouped_gemm_defaults_off():
    assert tiny_config().moe_grouped_gemm is False


def test_grouped_flag_wired_to_layer():
    layer = MoELayer(_cfg(moe_grouped_gemm=True))
    assert layer.grouped_gemm is True
    assert MoELayer(_cfg()).grouped_gemm is False


# ── forward parity (no-drop regime) ─────────────────────────────────────────
def test_grouped_dispatch_matches_loop_eval_nodrop():
    loop, grouped = _paired_layers()
    assert grouped.grouped_gemm is True  # ensures RED until the path exists
    x = torch.randn(2, 8, loop.dim)
    tok = torch.randint(0, 512, (2, 8))

    _, out_loop = loop(x, loop_idx=0, token_ids=tok)
    assert loop.last_drop_rate[0] == 0.0, "eval must be drop-free for valid parity"

    _, out_grp = grouped(x, loop_idx=0, token_ids=tok)
    assert grouped.last_drop_rate[0] == 0.0
    maxdiff = (out_loop - out_grp).abs().max().item()
    assert torch.allclose(out_loop, out_grp, atol=1e-4, rtol=1e-4), f"maxdiff={maxdiff}"


def test_grouped_dispatch_parity_multiple_loops():
    """Parity must hold at every loop index (loop embedding changes routing)."""
    loop, grouped = _paired_layers(seed=3)
    x = torch.randn(2, 8, loop.dim)
    tok = torch.randint(0, 512, (2, 8))
    for li in range(2):
        _, ol = loop(x, loop_idx=li, token_ids=tok)
        _, og = grouped(x, loop_idx=li, token_ids=tok)
        assert torch.allclose(ol, og, atol=1e-4, rtol=1e-4), f"loop {li}"


# ── gradient parity (reference primitive — CPU backward works) ──────────────
def test_grouped_dispatch_grad_matches_loop():
    """My gather/sort/gate/scatter gradient (the bug-prone part) must match the
    loop path. Uses the reference grouped-mm so CPU backward is valid; the
    vendor kernel's own backward is covered on GPU."""
    loop, grouped = _paired_layers(seed=1)
    loop.train(False)
    grouped.train(False)
    x = torch.randn(2, 8, loop.dim, requires_grad=False)
    tok = torch.randint(0, 512, (2, 8))

    xl = x.clone().requires_grad_(True)
    _, ol = loop(xl, loop_idx=0, token_ids=tok)
    ol.sum().backward()

    xg = x.clone().requires_grad_(True)
    _, og = grouped(xg, loop_idx=0, token_ids=tok)
    og.sum().backward()

    assert torch.allclose(xl.grad, xg.grad, atol=1e-4, rtol=1e-4)
    # Expert weight grads must match. An expert that received no tokens this
    # batch gets grad=None on the loop path (expert never called → optimizer
    # skips it) but grad=zeros on the grouped path (it's in the stacked weight
    # tensor; empty group → zero grad). Both mean "no update" — normalise
    # None→zeros so the comparison reflects that equivalence.
    for (n, pl), (_, pg) in zip(
        loop.experts.named_parameters(), grouped.experts.named_parameters()
    ):
        gl = pl.grad if pl.grad is not None else torch.zeros_like(pl)
        gg = pg.grad if pg.grad is not None else torch.zeros_like(pg)
        assert torch.allclose(gl, gg, atol=1e-4, rtol=1e-4), n


# ── vendor kernel backward (GPU only) ───────────────────────────────────────
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="kernel backward is CUDA-only")
def test_grouped_kernel_backward_gpu():
    loop, grouped = _paired_layers(seed=2)
    loop, grouped = loop.cuda(), grouped.cuda()
    x = torch.randn(2, 8, loop.dim, device="cuda")
    tok = torch.randint(0, 512, (2, 8), device="cuda")
    xg = x.clone().requires_grad_(True)
    _, og = grouped(xg, loop_idx=0, token_ids=tok)
    og.sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
