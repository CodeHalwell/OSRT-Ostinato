"""Tests for Phase 5b router features: sqrt(softplus) affinity + hash routing.

Both features are config-gated; the defaults (router_affinity="softmax",
hash_routing_blocks=0) preserve the historical behaviour bit-for-bit, which the
rest of the suite already covers. These tests exercise the opt-in paths.
"""

import torch

from osrt.config import OSRTConfig
from osrt.model import MoELayer, OSRTForCausalLM
from osrt.monitoring import moe_health
from osrt.muon import HybridMuonAdamW, Muon, build_param_groups


def _tiny_config(**over):
    base = dict(
        dim=128, heads=4, head_dim=32, num_kv_heads=2,
        vocab_size=256, real_vocab_size=256, num_blocks=3, recursive_loops=3,
        num_routed_experts=8, top_k_experts=2, expert_hidden=64,
        shared_expert_hidden=64, max_position_embeddings=64,
        aux_loop_loss_weight=0.05,
    )
    base.update(over)
    return OSRTConfig(**base)


def _train_steps(model, steps=20):
    torch.manual_seed(0)
    model.train()
    ids = torch.randint(0, 256, (4, 16))
    labels = ids.clone()
    mp, ag = build_param_groups(model.named_parameters(), 0.01)
    opt = HybridMuonAdamW(Muon(mp, lr=0.02), torch.optim.AdamW(ag, lr=3e-3))
    first = last = None
    for step in range(steps):
        opt.zero_grad()
        out = model(ids, labels=labels)
        out.loss.backward()
        opt.step()
        if step == 0:
            first = out.loss.item()
        last = out.loss.item()
    return first, last


# ── (1) sqrt(softplus) affinity ─────────────────────────────────────────


def test_sqrt_softplus_affinity_is_non_negative():
    """sqrt(softplus(logits)) is always >= 0, including for negative logits."""
    cfg = _tiny_config(router_affinity="sqrt_softplus")
    moe = MoELayer(cfg, moe_seed=0)
    # Drive a wide spread of router logits, including strongly negative ones.
    with torch.no_grad():
        moe.router.weight.normal_(0.0, 5.0)
    x = torch.randn(2, 16, cfg.dim) * 3.0
    affinity = torch.sqrt(
        torch.nn.functional.softplus(moe.router(x.reshape(-1, cfg.dim)))
    )
    assert (affinity >= 0).all()
    # And the layer forwards finitely on this pathological input.
    moe.eval()
    shared, routed = moe(x, loop_idx=0)
    assert torch.isfinite(shared).all()
    assert torch.isfinite(routed).all()


def test_sqrt_softplus_forward_finite_and_shape():
    model = OSRTForCausalLM(_tiny_config(router_affinity="sqrt_softplus"))
    ids = torch.randint(0, 256, (2, 16))
    out = model(ids, labels=ids.clone())
    assert out.logits.shape == (2, 16, 256)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.loss).item()


def test_sqrt_softplus_trains_without_nan():
    model = OSRTForCausalLM(
        _tiny_config(router_affinity="sqrt_softplus", dim=64, heads=2,
                     head_dim=32, num_kv_heads=1)
    )
    first, last = _train_steps(model)
    assert torch.isfinite(torch.tensor(last)), "sqrt_softplus produced NaN"
    assert last < first, "sqrt_softplus model failed to reduce loss"


def test_sqrt_softplus_monitoring_metrics_well_defined():
    """Telemetry consumed by the collapse monitor must stay populated and
    finite under the affinity-normalised probability view."""
    model = OSRTForCausalLM(_tiny_config(router_affinity="sqrt_softplus"))
    ids = torch.randint(0, 256, (4, 16))
    model(ids)
    h = moe_health(model)
    assert h.num_experts == 8
    for blk in h.load_entropy:
        for e in blk:
            assert 0.0 <= e <= 1.0001
    # The raw attributes the monitor reads are populated (not stale zeros).
    moe = model.model.blocks[0].moe
    assert all(
        f is not None for loop_f in moe.last_clean_expert_fraction
        for f in loop_f
    )
    assert all(torch.isfinite(torch.tensor(m)) for m in moe.last_marginal_entropy)
    # Balance/z losses are finite scalars.
    assert torch.isfinite(moe.balance_loss).item()
    assert torch.isfinite(moe.z_loss).item()


# ── (2) hash routing ─────────────────────────────────────────────────────


def test_hash_routing_dispatches_deterministically():
    """Blocks < hash_routing_blocks select expert (token_id+loop_idx)%E top-1;
    later blocks use the learned router."""
    cfg = _tiny_config(hash_routing_blocks=2)
    model = OSRTForCausalLM(cfg)
    model.eval()
    E = cfg.num_routed_experts

    ids = torch.randint(0, 256, (2, 16))
    model(ids)

    # Hash-routed blocks (0, 1): each loop's expert-fraction must be the
    # histogram of (token_id + loop_idx) % E over the batch (top-1, so the
    # hard-assignment fractions sum to 1 and match the deterministic hash).
    for block_idx in (0, 1):
        moe = model.model.blocks[block_idx].moe
        for loop in range(cfg.recursive_loops):
            assigned = (ids.reshape(-1) + loop) % E
            expected = torch.bincount(assigned, minlength=E).float()
            expected = expected / expected.sum()
            got = torch.tensor(moe.last_clean_expert_fraction[loop])
            assert torch.allclose(got, expected, atol=1e-5), (
                f"block {block_idx} loop {loop} hash fractions mismatch"
            )

    # Learned-routing block (2): fractions are NOT a pure hash histogram in
    # general; just assert they are a valid distribution (sum ~ 1).
    moe2 = model.model.blocks[2].moe
    got2 = torch.tensor(moe2.last_clean_expert_fraction[0])
    assert abs(got2.sum().item() - 1.0) < 1e-4


def test_hash_routing_forward_finite_and_trains():
    model = OSRTForCausalLM(
        _tiny_config(hash_routing_blocks=2, dim=64, heads=2, head_dim=32,
                     num_kv_heads=1)
    )
    ids = torch.randint(0, 256, (2, 16))
    out = model(ids, labels=ids.clone())
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.loss).item()
    first, last = _train_steps(model)
    assert torch.isfinite(torch.tensor(last))
    assert last < first, "hash-routed model failed to reduce loss"


def test_hash_routing_default_off_matches_learned():
    """hash_routing_blocks=0 (default) must use the learned router everywhere:
    expert fractions should not equal the deterministic hash histogram."""
    cfg = _tiny_config(hash_routing_blocks=0)
    model = OSRTForCausalLM(cfg)
    model.eval()
    E = cfg.num_routed_experts
    ids = torch.randint(0, 256, (2, 16))
    model(ids)
    moe = model.model.blocks[0].moe
    assigned = (ids.reshape(-1) + 0) % E
    hash_frac = torch.bincount(assigned, minlength=E).float()
    hash_frac = hash_frac / hash_frac.sum()
    got = torch.tensor(moe.last_clean_expert_fraction[0])
    # Learned top-2 routing fractions differ from the top-1 hash histogram.
    assert not torch.allclose(got, hash_frac, atol=1e-5)


def test_hash_routing_combines_with_sqrt_softplus():
    """Both features on at once must forward finitely (preset-like config)."""
    model = OSRTForCausalLM(
        _tiny_config(router_affinity="sqrt_softplus", hash_routing_blocks=2)
    )
    ids = torch.randint(0, 256, (2, 16))
    out = model(ids, labels=ids.clone())
    assert torch.isfinite(out.loss).item()
