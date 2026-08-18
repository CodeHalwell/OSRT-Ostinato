"""Tests for the per-head learnable attention sink (ARCHITECTURE.md §6.6/§6.7).

The sink adds an extra term to the softmax DENOMINATOR only (its "value" is
zero), so a query's attention weights may sum to < 1. Gated by
config.attention_sink (default False keeps the exact SDPA path). Enabling it
is implemented via the exact log-sum-exp rescale
    out_sink = out * sigmoid(lse - sink_logits[h]).
"""

import math

import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM


def _config(**over):
    base = dict(
        dim=128, heads=4, head_dim=32, num_kv_heads=2,
        vocab_size=256, real_vocab_size=256, num_blocks=2, recursive_loops=3,
        num_routed_experts=8, top_k_experts=2, expert_hidden=64,
        shared_expert_hidden=64, max_position_embeddings=64,
        aux_loop_loss_weight=0.05,
    )
    base.update(over)
    return OSRTConfig(**base)


def test_sink_param_exists_shape_and_zero_init():
    """Every block carries a (heads,) sink param, zero-initialised."""
    model = OSRTForCausalLM(_config(attention_sink=True))
    blocks = model.model.blocks
    assert len(blocks) == 2
    for block in blocks:
        assert hasattr(block, "sink_logits"), "block missing sink_logits"
        assert block.sink_logits.shape == (4,)
        assert torch.allclose(block.sink_logits, torch.zeros(4))
        assert block.sink_logits.requires_grad
    # Total sink params = heads * num_blocks.
    n_sink = sum(b.sink_logits.numel() for b in blocks)
    assert n_sink == 4 * 2


def test_sink_absent_when_disabled():
    """Default (disabled) blocks carry no sink parameter at all."""
    model = OSRTForCausalLM(_config(attention_sink=False))
    for block in model.model.blocks:
        assert not hasattr(block, "sink_logits")


def test_sink_forward_finite_and_changes_logits():
    """Enabling the sink keeps logits finite and changes the computation.

    At sink_logits=0 the sink multiplies each head's output by
    sigmoid(lse) < 1 (a mild down-weighting), so the logits must differ from
    the sink-free run — same weights otherwise (identical seed)."""
    torch.manual_seed(0)
    cfg_off = _config(attention_sink=False)
    model_off = OSRTForCausalLM(cfg_off)
    model_off.eval()

    torch.manual_seed(0)
    cfg_on = _config(attention_sink=True)
    model_on = OSRTForCausalLM(cfg_on)
    model_on.eval()
    # Same seed ⇒ all shared params identical; the only difference is the
    # (zero-init) sink that the disabled model lacks. Copy shared weights
    # across to be certain the ONLY difference is the sink term.
    on_sd = model_on.state_dict()
    for k, v in model_off.state_dict().items():
        on_sd[k].copy_(v)

    ids = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        out_off = model_off(ids).logits
        out_on = model_on(ids).logits
    assert torch.isfinite(out_on).all()
    assert not torch.allclose(out_on, out_off, atol=1e-5), (
        "attention_sink=True must change the output vs the SDPA path"
    )


def test_sink_cached_decode_matches_full_forward():
    """Cached decode == full forward with the sink enabled (atol ~1e-4)."""
    model = OSRTForCausalLM(_config(attention_sink=True))
    model.eval()
    full = torch.randint(0, 256, (1, 6))
    pre = model(full[:, :5], use_cache=True)
    step = model(
        full[:, 5:6], past_key_values=pre.past_key_values, use_cache=True,
    )
    ref = model(full, use_cache=False)
    assert torch.allclose(step.logits[:, -1], ref.logits[:, -1], atol=1e-4)


def test_sink_cached_decode_multi_token_chunk():
    """Prefill + multi-token decode chunk (S>1, past_len>0) matches full."""
    model = OSRTForCausalLM(_config(attention_sink=True))
    model.eval()
    full = torch.randint(0, 256, (1, 8))
    pre = model(full[:, :4], use_cache=True)
    chunk = model(
        full[:, 4:8], past_key_values=pre.past_key_values, use_cache=True,
    )
    ref = model(full, use_cache=False)
    assert torch.allclose(chunk.logits[:, -1], ref.logits[:, -1], atol=1e-4)


def test_sink_trains_without_nan():
    """Sink params receive gradient and training stays finite / loss drops."""
    from osrt.muon import HybridMuonAdamW, Muon, build_param_groups

    torch.manual_seed(0)
    model = OSRTForCausalLM(
        _config(attention_sink=True, dim=64, head_dim=32, heads=2, num_kv_heads=1)
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
        # The 1D sink param must get a gradient (routed to AdamW).
        if step == 0:
            for block in model.model.blocks:
                assert block.sink_logits.grad is not None
                assert torch.isfinite(block.sink_logits.grad).all()
        opt.step()
        if step == 0:
            first = out.loss.item()
        last = out.loss.item()
    assert torch.isfinite(torch.tensor(last)), "sink training produced NaN"
    assert last < first, "sink model failed to reduce loss"


def test_sink_rescale_matches_handcomputed_softmax():
    """Direct numerical check of the sink against a hand-rolled softmax that
    carries an extra exp(sink) term in its denominator, on a tiny example.

    Builds one block, drives its _attention_with_sink with a known sink, and
    compares to the explicit formula
        s_{h,i,j} = exp(z_{h,i,j}) / (Σ_k exp(z_{h,i,k}) + exp(sink[h])),
        out_{h,i} = Σ_j s_{h,i,j} v_{h,j}.
    """
    torch.manual_seed(0)
    cfg = _config(attention_sink=True, dim=16, heads=2, num_kv_heads=2, head_dim=8)
    block = OSRTForCausalLM(cfg).model.blocks[0]
    # Set non-trivial sink logits so the extra denominator term matters.
    with torch.no_grad():
        block.sink_logits.copy_(torch.tensor([0.5, -1.0]))

    B, H, S, hd = 1, 2, 5, 8
    # kv_heads == heads here ⇒ group_size 1, no GQA expansion to reason about.
    q = torch.randn(B, H, S, hd)
    k = torch.randn(B, H, S, hd)
    v = torch.randn(B, H, S, hd)

    out = block._attention_with_sink(q, k, v, S=S, total_len=S, past_len=0)

    # Reference: explicit masked softmax-with-extra-denominator-term.
    scale = 1.0 / math.sqrt(hd)
    scores = (q @ k.transpose(-2, -1)) * scale          # (B,H,S,S)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    scores = scores.double()
    sink = block.sink_logits.double().view(1, H, 1)     # (1,H,1)
    # denom = Σ_k exp(z) + exp(sink), computed stably relative to row max.
    row_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - row_max)
    denom = (exp_scores.sum(dim=-1, keepdim=True)
             + torch.exp(sink.unsqueeze(-1) - row_max))
    weights = exp_scores / denom                        # (B,H,S,S), rows sum < 1
    ref = weights @ v.double()

    assert torch.allclose(out.double(), ref, atol=1e-6), (
        "sink rescale does not match hand-computed extra-denominator softmax"
    )
    # Sanity: rows sum to < 1 (the whole point of the sink).
    assert (weights.sum(dim=-1) < 1.0).all()
