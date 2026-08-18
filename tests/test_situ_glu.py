"""SiTU-GLU (Kimi K3 §2.3.2) — a param-free smooth-cap alternative to the hard
SwiGLU clamp, wired behind config.situ_glu (default False).

Guards the three properties that make it a safe A/B toggle:
  1. Default (situ=False) is bit-identical to the original SwiGLU path.
  2. situ=True bounds the expert output to |out| <= beta_gate * beta_up and
     matches SwiGLU near the origin.
  3. It adds NO parameters, so eager/grouped paths stay in parity and a
     situ=False checkpoint loads into a situ=True model unchanged.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from test_model import tiny_config  # noqa: E402

from osrt.model import ExpertFFN, MoELayer, _glu_combine  # noqa: E402


def _cfg(**over):
    base = dict(num_routed_experts=8, top_k_experts=2, expert_hidden=64)
    base.update(over)
    return tiny_config(**base)


# ── 1. default path unchanged ───────────────────────────────────────────────
def test_glu_combine_default_is_swiglu():
    g, u = torch.randn(4, 16), torch.randn(4, 16)
    out = _glu_combine(g, u, clamp=None, situ=False, b_gate=4.0, b_up=25.0)
    assert torch.equal(out, F.silu(g) * u)


def test_glu_combine_default_respects_hard_clamp():
    g, u = torch.randn(4, 16) * 10, torch.randn(4, 16) * 10
    out = _glu_combine(g, u, clamp=3.0, situ=False, b_gate=4.0, b_up=25.0)
    ref = F.silu(g.clamp(max=3.0)) * u.clamp(min=-3.0, max=3.0)
    assert torch.equal(out, ref)


def test_expertffn_default_matches_manual_swiglu():
    torch.manual_seed(0)
    ffn = ExpertFFN(32, 64).eval()
    x = torch.randn(3, 32)
    manual = ffn.w_down(F.silu(ffn.w_gate(x)) * ffn.w_up(x))
    assert torch.allclose(ffn(x), manual, atol=1e-6)


# ── 2. SiTU-GLU numerics ────────────────────────────────────────────────────
def test_situ_output_is_bounded():
    b_gate, b_up = 4.0, 25.0
    # Extreme inputs that would blow up unbounded SwiGLU.
    g, u = torch.randn(64, 128) * 50, torch.randn(64, 128) * 50
    out = _glu_combine(g, u, clamp=None, situ=True, b_gate=b_gate, b_up=b_up)
    assert out.abs().max().item() <= b_gate * b_up + 1e-4
    assert torch.isfinite(out).all()


def test_situ_matches_swiglu_near_origin():
    # SiTU-GLU matches SwiGLU to first order around 0 (tanh(z/b)~z/b there).
    g, u = torch.randn(8, 16) * 0.05, torch.randn(8, 16) * 0.05
    situ = _glu_combine(g, u, clamp=None, situ=True, b_gate=4.0, b_up=25.0)
    swiglu = F.silu(g) * u
    assert torch.allclose(situ, swiglu, atol=1e-3)


def test_situ_gradients_flow_past_the_cap():
    # The point of the smooth cap: in the transition band past where a hard
    # clamp would kick in, SiTU keeps gradients nonzero, whereas the hard clamp
    # zeros them. (Deep saturation z >> b still vanishes — tanh saturates — but
    # the band is wide, not a cliff.)
    bg, bu = 4.0, 25.0
    g = torch.full((4,), 6.0, requires_grad=True)   # past a clamp of 3
    u = torch.full((4,), 6.0, requires_grad=True)
    _glu_combine(g, u, clamp=None, situ=True, b_gate=bg, b_up=bu).sum().backward()
    assert g.grad.abs().sum() > 0 and u.grad.abs().sum() > 0

    # Contrast: a hard clamp at 3 zeros both gradients at input 6.
    gc = torch.full((4,), 6.0, requires_grad=True)
    uc = torch.full((4,), 6.0, requires_grad=True)
    _glu_combine(gc, uc, clamp=3.0, situ=False, b_gate=bg, b_up=bu).sum().backward()
    assert gc.grad.abs().sum() == 0 and uc.grad.abs().sum() == 0


# ── 3. param-free ⇒ parity + checkpoint compatibility ───────────────────────
def test_situ_adds_no_parameters():
    plain = ExpertFFN(32, 64)
    situ = ExpertFFN(32, 64, situ=True)
    assert set(plain.state_dict()) == set(situ.state_dict())


def test_situ_checkpoint_loads_into_either_flag():
    torch.manual_seed(0)
    off = MoELayer(_cfg())                       # trained with SwiGLU
    on = MoELayer(_cfg(situ_glu=True))           # A/B variant
    on.load_state_dict(off.state_dict(), strict=True)  # raises on any mismatch
    assert on.experts[0].situ is True and off.experts[0].situ is False


def test_grouped_and_loop_paths_agree_under_situ():
    """The SiTU activation must be applied identically in the eager loop path
    and the grouped-GEMM path (both route through _glu_combine)."""
    torch.manual_seed(0)
    loop = MoELayer(_cfg(situ_glu=True)).eval()
    grouped = MoELayer(_cfg(situ_glu=True, moe_grouped_gemm=True)).eval()
    grouped.load_state_dict(loop.state_dict())
    x = torch.randn(2, 8, loop.dim)
    tok = torch.randint(0, 512, (2, 8))
    _, ol = loop(x, loop_idx=0, token_ids=tok)
    _, og = grouped(x, loop_idx=0, token_ids=tok)
    assert loop.last_drop_rate[0] == 0.0 and grouped.last_drop_rate[0] == 0.0
    assert torch.allclose(ol, og, atol=1e-4, rtol=1e-4)
