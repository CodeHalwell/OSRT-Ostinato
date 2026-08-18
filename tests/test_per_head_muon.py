"""Per-head Muon (Kimi K3 §2.5) — opt-in via config.per_head_muon / the
`head_dim` group key. Orthogonalises each attention head's block of q_proj /
kv_down / v_from_k separately instead of the full matrix.

Guards the properties that make it a safe A/B toggle:
  1. newton_schulz5_perhead is a correct per-block orthogonalisation.
  2. A group with `head_dim` takes the per-head path; without it, the full path
     (unchanged) — and per-head keeps the same update magnitude (no LR shift).
  3. build_param_groups only splits out the attention group when opted in;
     default returns the flat list unchanged.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from osrt.muon import (  # noqa: E402
    Muon,
    build_param_groups,
    newton_schulz5,
    newton_schulz5_perhead,
)


# ── 1. per-head NS numerics ─────────────────────────────────────────────────
def test_perhead_reshape_matches_blockwise_ns():
    head_dim, n_heads, cols = 8, 4, 32
    g = torch.randn(n_heads * head_dim, cols)
    out = newton_schulz5_perhead(g, head_dim)
    # Must equal stacking NS on each contiguous head-block of rows.
    ref = torch.cat([
        newton_schulz5(g[h * head_dim:(h + 1) * head_dim]) for h in range(n_heads)
    ])
    assert out.shape == g.shape
    assert torch.allclose(out, ref, atol=1e-5)


def test_perhead_blocks_are_normalized():
    # NS5 (bf16, 5 steps) pulls each block's singular values toward 1, so an
    # orthonormal-rows block has Frobenius norm ~ sqrt(head_dim). Checks each
    # head-block was independently normalised (not left at its raw random scale).
    head_dim, n_heads, cols = 8, 4, 32   # head_dim < cols → orthonormal rows
    g = torch.randn(n_heads * head_dim, cols) * 7.0  # arbitrary raw scale
    out = newton_schulz5_perhead(g, head_dim).float()
    target = head_dim ** 0.5
    for h in range(n_heads):
        blk = out[h * head_dim:(h + 1) * head_dim]
        assert 0.75 * target <= blk.norm().item() <= 1.25 * target, f"head {h}"


def test_perhead_requires_divisible_out_dim():
    g = torch.randn(30, 16)  # 30 not divisible by head_dim=8
    try:
        newton_schulz5_perhead(g, 8)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ── 2. optimizer step: per-head vs full ─────────────────────────────────────
def _one_step(param, grad, groups_extra=None):
    p = torch.nn.Parameter(param.clone())
    p.grad = grad.clone()
    grp = {"params": [p]}
    if groups_extra:
        grp.update(groups_extra)
    Muon([grp], lr=0.1, momentum=0.0, nesterov=False).step()
    return (p.data - param).clone()


def test_per_head_step_differs_from_full_same_magnitude():
    torch.manual_seed(0)
    head_dim, n_heads, cols = 8, 4, 32
    w0 = torch.randn(n_heads * head_dim, cols)
    grad = torch.randn_like(w0)

    d_full = _one_step(w0, grad)                                  # no head_dim
    d_head = _one_step(w0, grad, {"head_dim": head_dim})          # per-head

    # The per-head branch was taken → a genuinely different update direction …
    assert not torch.allclose(d_full, d_head, atol=1e-4)
    # … but the magnitude stays comparable (per-block shape scale keeps the
    # effective LR the same up to NS5's approximation noise), so no gross LR
    # confound between the two arms of the A/B.
    rel = abs(d_full.norm().item() - d_head.norm().item()) / d_full.norm().item()
    assert rel < 0.15, f"magnitude drifted {rel:.3f}"


def test_head_dim_ignored_when_not_divisible():
    # A non-attention matrix accidentally in a head_dim group must fall back to
    # the full path rather than crash (rows % head_dim != 0).
    w0 = torch.randn(30, 16)
    grad = torch.randn_like(w0)
    d_head = _one_step(w0, grad, {"head_dim": 8})
    d_full = _one_step(w0, grad)
    assert torch.allclose(d_head, d_full, atol=1e-5)


def test_default_muon_unchanged():
    # A plain param list (no head_dim anywhere) must be exactly the full path.
    w0 = torch.randn(32, 32)
    grad = torch.randn_like(w0)
    p = torch.nn.Parameter(w0.clone())
    p.grad = grad.clone()
    Muon([p], lr=0.1, momentum=0.0, nesterov=False).step()
    ref = -0.1 * newton_schulz5(grad).to(p.dtype)  # rows==cols → shape_scale 1
    assert torch.allclose(p.data - w0, ref, atol=1e-5)


# ── 3. build_param_groups split ─────────────────────────────────────────────
def _named():
    mk = lambda *s: torch.nn.Parameter(torch.randn(*s))  # noqa: E731
    return [
        ("model.blocks.0.attn.q_proj.weight", mk(64, 32)),
        ("model.blocks.0.attn.kv_down.weight", mk(16, 32)),
        ("model.blocks.0.attn.v_from_k.weight", mk(16, 16)),
        ("model.blocks.0.attn.out_proj.weight", mk(32, 32)),
        ("model.blocks.0.moe.experts.0.w_gate.weight", mk(64, 32)),
        ("model.embedding.weight", mk(100, 32)),
        ("model.blocks.0.norm.weight", mk(32)),
    ]


def test_default_keeps_flat_muon_list():
    muon, _ = build_param_groups(_named(), weight_decay=0.1)
    # Flat list of Parameters (not group dicts); attention projections included.
    assert all(isinstance(p, torch.nn.Parameter) for p in muon)
    assert len(muon) == 5  # q,kv_down,v_from_k,out_proj,w_gate (embed/norm → AdamW)


def test_per_head_splits_attention_into_its_own_group():
    muon, _ = build_param_groups(
        _named(), weight_decay=0.1, per_head_attn=True, head_dim=8,
    )
    assert isinstance(muon, list) and isinstance(muon[0], dict)
    attn_grp, other_grp = muon[0], muon[1]
    assert attn_grp["head_dim"] == 8
    assert len(attn_grp["params"]) == 3      # q_proj, kv_down, v_from_k
    assert len(other_grp["params"]) == 2     # out_proj, w_gate (full-matrix)
    # Optimizer accepts the grouped output and steps without error.
    for p in attn_grp["params"] + other_grp["params"]:
        p.grad = torch.randn_like(p)
    Muon(muon, lr=0.01).step()
