"""Tests for Multi-Token Prediction (MTP) heads (docs/ARCHITECTURE.md §9.3, §11.4).

MTP is a TRAINING-TIME auxiliary objective: extra heads on the FINAL hidden
state predict tokens at offsets +2, +3, ... in addition to the main +1 head.
It must not change inference/generation, and with the default mtp_heads=0 it is
bit-identical to the no-MTP path.
"""

import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM


def _mtp_config(**over):
    base = dict(
        dim=128, heads=4, head_dim=32, num_kv_heads=2,
        vocab_size=256, real_vocab_size=256, num_blocks=2, recursive_loops=3,
        num_routed_experts=8, top_k_experts=2, expert_hidden=64,
        shared_expert_hidden=64, max_position_embeddings=64,
    )
    base.update(over)
    return OSRTConfig(**base)


def test_mtp_off_by_default_is_bit_identical():
    """mtp_heads=0 (default): no heads, no telemetry, loss unchanged."""
    torch.manual_seed(0)
    model = OSRTForCausalLM(_mtp_config())  # mtp_heads defaults to 0
    assert model.config.mtp_heads == 0
    assert len(model.mtp_heads) == 0, "no MTP modules when disabled"

    model.train()
    ids = torch.randint(0, 256, (2, 16))
    labels = ids.clone()
    out = model(ids, labels=labels)
    assert torch.isfinite(out.loss).item()
    # No MTP contribution / telemetry when disabled.
    assert model.last_mtp_loss is None
    assert model.last_mtp_losses == []


def test_mtp_modules_exist_when_enabled():
    model = OSRTForCausalLM(_mtp_config(mtp_heads=2))
    assert len(model.mtp_heads) == 2
    for head in model.mtp_heads:
        assert isinstance(head.norm, torch.nn.RMSNorm)
        assert isinstance(head.proj, torch.nn.Linear)
        assert head.proj.bias is None
        assert head.proj.weight.shape == (128, 128)


def test_mtp_train_loss_includes_mtp_term():
    """Training forward with MTP produces finite loss INCLUDING the mtp term."""
    torch.manual_seed(0)
    model = OSRTForCausalLM(_mtp_config(mtp_heads=2, mtp_loss_weight=0.3))
    model.train()
    ids = torch.randint(0, 256, (2, 16))
    labels = ids.clone()
    out = model(ids, labels=labels)

    assert torch.isfinite(out.loss).item()
    # MTP telemetry populated and finite.
    assert model.last_mtp_loss is not None
    assert torch.isfinite(model.last_mtp_loss).item()
    assert len(model.last_mtp_losses) == 2
    for t in model.last_mtp_losses:
        assert torch.isfinite(t).item()

    # The training loss must EXCEED pure task CE by (at least) the mtp term
    # (other aux coeffs default to non-negative too, but the mtp piece alone
    # makes the gap strictly positive here).
    assert model.last_task_loss is not None
    assert out.loss.item() > model.last_task_loss.item()
    # weighted mtp sum should equal weight * sum(per-head raw CE).
    expected = 0.3 * sum(t.item() for t in model.last_mtp_losses)
    assert abs(model.last_mtp_loss.item() - expected) < 1e-4


def test_mtp_eval_loss_is_pure_task_ce():
    """In eval mode the loss excludes the mtp term — it is pure task CE."""
    torch.manual_seed(0)
    model = OSRTForCausalLM(_mtp_config(mtp_heads=2, mtp_loss_weight=0.3))
    model.eval()
    ids = torch.randint(0, 256, (2, 16))
    labels = ids.clone()
    out = model(ids, labels=labels)

    assert model.last_task_loss is not None
    # Eval loss == task CE exactly (no aux, no mtp).
    assert abs(out.loss.item() - model.last_task_loss.item()) < 1e-6
    # MTP heads are not run in eval — no telemetry.
    assert model.last_mtp_loss is None
    assert model.last_mtp_losses == []


def test_mtp_shifted_label_alignment():
    """Head k targets offset (2+k): verify against a hand-computed CE.

    Monkeypatch each head to the identity so its logits equal the tied LM
    head's logits; then the per-head MTP CE must equal cross_entropy of those
    logits against labels shifted by the head's offset, with the tail dropped.
    """
    torch.manual_seed(0)
    import torch.nn.functional as F

    model = OSRTForCausalLM(_mtp_config(mtp_heads=2))
    model.train()
    # Make each MTP projection a pass-through (identity norm + identity proj)
    # so head(hidden) == hidden and its logits match the main LM head.
    for head in model.mtp_heads:
        head.norm = torch.nn.Identity()
        head.proj = torch.nn.Identity()

    ids = torch.randint(0, 256, (2, 16))
    labels = ids.clone()
    model(ids, labels=labels)

    # Recompute the main LM-head logits from the final hidden state.
    (hidden, *_rest) = model.model(ids)
    logits = F.linear(hidden, model.model.embedding.weight)

    for k in range(2):
        offset = k + 2
        ref_logits = logits[..., :-offset, :model.config.real_vocab_size]
        ref_logits = ref_logits.contiguous().float()
        ref_labels = labels[..., offset:].contiguous()
        ref_ce = F.cross_entropy(
            ref_logits.view(-1, model.config.real_vocab_size),
            ref_labels.view(-1),
            ignore_index=-100,
        )
        assert torch.allclose(
            model.last_mtp_losses[k], ref_ce, atol=1e-4
        ), f"head {k} CE must align with offset +{offset} targets"


def test_mtp_trains_without_nan():
    """A few optimizer steps with MTP on stay finite and reduce loss."""
    from osrt.muon import HybridMuonAdamW, Muon, build_param_groups

    torch.manual_seed(0)
    model = OSRTForCausalLM(
        _mtp_config(
            dim=64, head_dim=32, heads=2, num_kv_heads=1,
            mtp_heads=2, mtp_loss_weight=0.3,
        )
    )
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
    assert torch.isfinite(torch.tensor(last)), "MTP training produced NaN"
    assert last < first, "MTP model failed to reduce loss"
