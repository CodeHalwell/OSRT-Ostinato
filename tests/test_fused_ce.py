"""Parity tests for the chunked fused linear-cross-entropy (review item B2).

The fused path must produce the SAME loss and the SAME gradients (w.r.t. both the
hidden state and the tied weight) as the reference
``F.cross_entropy(F.linear(hidden, weight)[:real_vocab].float(), labels)`` — within
floating-point tolerance — while only ever materialising one chunk of the
(N, vocab) logits at a time. It is gated behind a config flag and defaults OFF, so
the standard training path stays bit-identical; these tests prove that turning it
ON does not change the math.
"""

import torch
import torch.nn.functional as F

from osrt.fused_ce import fused_linear_cross_entropy


def _reference(hidden, weight, labels, real_vocab, ignore_index=-100):
    logits = F.linear(hidden, weight[:real_vocab]).float()
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


def test_fused_ce_matches_reference_loss_and_grads():
    torch.manual_seed(0)
    N, D, V = 40, 16, 50
    hidden = torch.randn(N, D, dtype=torch.float32)
    weight = torch.randn(V, D, dtype=torch.float32)
    labels = torch.randint(0, V, (N,))
    labels[3] = -100  # exercise ignore_index
    labels[10] = -100

    h1 = hidden.clone().requires_grad_(True)
    w1 = weight.clone().requires_grad_(True)
    ref = _reference(h1, w1, labels, V)
    ref.backward()

    h2 = hidden.clone().requires_grad_(True)
    w2 = weight.clone().requires_grad_(True)
    fused = fused_linear_cross_entropy(
        h2, w2, labels, real_vocab_size=V, ignore_index=-100, n_chunks=4,
    )
    fused.backward()

    assert torch.allclose(fused, ref, atol=1e-5, rtol=1e-4), (fused.item(), ref.item())
    assert torch.allclose(h2.grad, h1.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(w2.grad, w1.grad, atol=1e-5, rtol=1e-4)


def test_fused_ce_real_vocab_slice_matches_reference():
    # weight has more rows (padded vocab) than real_vocab_size; only the first
    # real_vocab_size rows participate, exactly like the model's logit slice.
    torch.manual_seed(1)
    N, D, V_real, V_pad = 24, 12, 30, 48
    hidden = torch.randn(N, D)
    weight = torch.randn(V_pad, D)
    labels = torch.randint(0, V_real, (N,))

    h1 = hidden.clone().requires_grad_(True)
    w1 = weight.clone().requires_grad_(True)
    ref = _reference(h1, w1, labels, V_real)
    ref.backward()

    h2 = hidden.clone().requires_grad_(True)
    w2 = weight.clone().requires_grad_(True)
    fused = fused_linear_cross_entropy(
        h2, w2, labels, real_vocab_size=V_real, n_chunks=3,
    )
    fused.backward()

    assert torch.allclose(fused, ref, atol=1e-5, rtol=1e-4)
    assert torch.allclose(h2.grad, h1.grad, atol=1e-5, rtol=1e-4)
    # Padded rows (>= V_real) get no gradient signal in either path.
    assert torch.allclose(w2.grad, w1.grad, atol=1e-5, rtol=1e-4)


def test_fused_ce_single_chunk_equals_reference():
    # n_chunks=1 is the degenerate "no chunking" case; must still match.
    torch.manual_seed(2)
    N, D, V = 16, 8, 20
    hidden = torch.randn(N, D, requires_grad=True)
    weight = torch.randn(V, D, requires_grad=True)
    labels = torch.randint(0, V, (N,))
    fused = fused_linear_cross_entropy(
        hidden, weight, labels, real_vocab_size=V, n_chunks=1,
    )
    ref = _reference(
        hidden.detach().requires_grad_(True),
        weight.detach().requires_grad_(True),
        labels, V,
    )
    assert torch.allclose(fused, ref, atol=1e-6, rtol=1e-5)


def test_fused_ce_all_ignored_returns_zero_not_nan():
    # F.cross_entropy returns NaN when every target is ignored (0/0). The fused
    # path deliberately returns a finite 0.0 instead — safer for the aux/MTP
    # heads on degenerate (fully-masked) micro-batches. Documented divergence.
    torch.manual_seed(3)
    N, D, V = 8, 8, 20
    hidden = torch.randn(N, D, requires_grad=True)
    weight = torch.randn(V, D, requires_grad=True)
    labels = torch.full((N,), -100)
    fused = fused_linear_cross_entropy(hidden, weight, labels, real_vocab_size=V)
    assert torch.isfinite(fused)
    assert fused.item() == 0.0


# ── Model integration (config flag wiring) ──────────────────────────────

from osrt.config import OSRTConfig  # noqa: E402
from osrt.model import OSRTForCausalLM  # noqa: E402


def _tiny_cfg(**overrides) -> OSRTConfig:
    defaults = dict(
        dim=64, heads=4, head_dim=16,
        vocab_size=128, real_vocab_size=128,
        num_blocks=2, recursive_loops=3,
        num_routed_experts=4, top_k_experts=2,
        expert_hidden=32, shared_expert_hidden=64,
        max_position_embeddings=32,
        aux_loop_loss_weight=0.1, mtp_heads=2,
    )
    defaults.update(overrides)
    return OSRTConfig(**defaults)


def test_config_rejects_negative_fused_ce_chunks():
    import pytest
    with pytest.raises(ValueError):
        _tiny_cfg(fused_cross_entropy_chunks=-1)


def test_fused_ce_invoked_only_when_enabled(monkeypatch):
    """The fused path must actually be called for the aux/MTP heads when the
    flag is > 0, and never when it is 0. Drives the model wiring."""
    import osrt.fused_ce as fce
    real = fce.fused_linear_cross_entropy
    calls = {"n": 0}

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr("osrt.model.fused_linear_cross_entropy", spy, raising=False)

    B, S = 2, 16
    ids = torch.randint(0, 128, (B, S))

    cfg_off = _tiny_cfg(fused_cross_entropy_chunks=0)
    torch.manual_seed(0)
    model_off = OSRTForCausalLM(cfg_off).train()
    calls["n"] = 0
    model_off(ids, labels=ids.clone()).loss.backward()
    assert calls["n"] == 0, "fused CE must not run when flag is 0"

    cfg_on = _tiny_cfg(fused_cross_entropy_chunks=4)
    torch.manual_seed(0)
    model_on = OSRTForCausalLM(cfg_on).train()
    calls["n"] = 0
    model_on(ids, labels=ids.clone()).loss.backward()
    # 2 aux-loop intermediate heads + 2 MTP heads = 4 fused calls.
    assert calls["n"] >= 4, f"fused CE must run for aux/MTP heads, got {calls['n']}"


def test_model_loss_and_grad_unchanged_when_fused_ce_enabled():
    """Turning the fused flag on must not change the training loss or the tied
    embedding gradient beyond fp tolerance (the safety guarantee)."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = OSRTForCausalLM(cfg).train()
    B, S = 2, 16
    ids = torch.randint(0, cfg.real_vocab_size, (B, S))
    labels = ids.clone()

    def run(chunks):
        cfg.fused_cross_entropy_chunks = chunks
        torch.manual_seed(123)  # pin any stochastic routing identically
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=labels)
        out.loss.backward()
        return out.loss.detach().clone(), model.model.embedding.weight.grad.clone()

    loss_off, g_off = run(0)
    loss_on, g_on = run(4)

    assert torch.allclose(loss_off, loss_on, atol=1e-4, rtol=1e-4), (
        loss_off.item(), loss_on.item(),
    )
    assert torch.allclose(g_off, g_on, atol=1e-4, rtol=1e-4)
