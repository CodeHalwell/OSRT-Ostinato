"""Tests for Manifold-Constrained Hyper-Connections (mHC)."""

import torch

from osrt.config import OSRTConfig
from osrt.mhc import ManifoldHyperConnection, sinkhorn_doubly_stochastic
from osrt.model import OSRTForCausalLM


def _mhc_config(**over):
    base = dict(
        dim=128, heads=4, head_dim=32, num_kv_heads=2,
        vocab_size=256, real_vocab_size=256, num_blocks=2, recursive_loops=3,
        num_routed_experts=8, top_k_experts=2, expert_hidden=64,
        shared_expert_hidden=64, max_position_embeddings=64,
        aux_loop_loss_weight=0.05, use_mhc=True, n_hc=4,
    )
    base.update(over)
    return OSRTConfig(**base)


def test_sinkhorn_is_doubly_stochastic_and_nonexpansive():
    torch.manual_seed(0)
    logits = torch.randn(8, 4, 4) * 3.0  # wide spread stresses convergence
    b = sinkhorn_doubly_stochastic(logits, 20)
    assert (b >= 0).all()
    # 20 iterations give APPROXIMATE double stochasticity; near-identity
    # init (the model's actual regime) converges to <1e-3, this stress
    # case to ~1.5e-2. Both keep the spectral norm ≈ 1 (non-expansive).
    assert (b.sum(-1) - 1).abs().max() < 2e-2, "rows must sum to ~1"
    assert (b.sum(-2) - 1).abs().max() < 2e-2, "cols must sum to ~1"
    spec = torch.linalg.matrix_norm(b, ord=2)
    assert spec.max().item() <= 1.01

    # The model's realistic near-identity regime converges tightly.
    near_id = torch.eye(4).reshape(-1).repeat(8, 1).reshape(8, 4, 4) * 4.0
    b2 = sinkhorn_doubly_stochastic(near_id + torch.randn(8, 4, 4) * 0.05, 20)
    assert (b2.sum(-1) - 1).abs().max() < 1e-2
    assert torch.linalg.matrix_norm(b2, ord=2).max().item() <= 1.0 + 1e-4


def test_mhc_shapes_input_view_and_update():
    mhc = ManifoldHyperConnection(dim=16, n_hc=4)
    X = torch.randn(2, 5, 4, 16)
    a, b, c = mhc.generate(X)
    assert a.shape == (2, 5, 4) and c.shape == (2, 5, 4)
    assert b.shape == (2, 5, 4, 4)
    assert (a >= 0).all() and (a <= 1).all()          # sigmoid-bounded
    assert (c >= 0).all() and (c <= 2).all()          # 2*sigmoid-bounded
    x_in = mhc.input_view(X, a)
    assert x_in.shape == (2, 5, 16)
    X2 = mhc.update(X, b, c, x_in)
    assert X2.shape == X.shape


def test_mhc_forward_finite_and_logits_shape():
    model = OSRTForCausalLM(_mhc_config())
    ids = torch.randint(0, 256, (2, 16))
    out = model(ids, labels=ids.clone())
    assert out.logits.shape == (2, 16, 256)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.loss).item()


def test_mhc_cached_decode_matches_full_forward():
    model = OSRTForCausalLM(_mhc_config())
    model.eval()
    full = torch.randint(0, 256, (1, 6))
    pre = model(full[:, :5], use_cache=True)
    step = model(
        full[:, 5:6], past_key_values=pre.past_key_values, use_cache=True,
    )
    ref = model(full, use_cache=False)
    assert torch.allclose(step.logits[:, -1], ref.logits[:, -1], atol=1e-4)


def test_mhc_trains_without_nan():
    """The whole point of log-domain Sinkhorn: training stays finite."""
    from osrt.muon import HybridMuonAdamW, Muon, build_param_groups

    torch.manual_seed(0)
    model = OSRTForCausalLM(_mhc_config(dim=64, head_dim=32, heads=2, num_kv_heads=1))
    model.train()
    ids = torch.randint(0, 256, (4, 16))
    labels = ids.clone()
    mp, ag = build_param_groups(model.named_parameters(), 0.01)
    opt = HybridMuonAdamW(Muon(mp, lr=0.02), torch.optim.AdamW(ag, lr=3e-3))
    first = last = None
    for step in range(30):
        opt.zero_grad()
        out = model(ids, labels=labels)
        out.loss.backward()
        opt.step()
        if step == 0:
            first = out.loss.item()
        last = out.loss.item()
    assert torch.isfinite(torch.tensor(last)), "mHC training produced NaN"
    assert last < first, "mHC model failed to reduce loss"


def test_collapse_head_exists_and_initialized_uniform():
    model = OSRTForCausalLM(_mhc_config())
    w = model.model.mhc_collapse
    assert w.shape == (4,)
    assert torch.allclose(w, torch.full((4,), 0.25))
