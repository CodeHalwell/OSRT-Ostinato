"""MTP-head self-speculative decoding: generate(speculative=True, spec_drafter=mtp)."""
import pytest
import torch

from osrt.model import OSRTForCausalLM

from .test_model import tiny_config


def _model(**over):
    torch.manual_seed(0)
    kw = dict(recursive_loops=2, mtp_heads=2, router_capacity_factor=10.0)
    kw.update(over)
    cfg = tiny_config(**kw)
    m = OSRTForCausalLM(cfg)
    m.train(False)
    return m, cfg


def test_mtp_speculative_equals_greedy_token_for_token():
    """Random heads propose junk, so nearly every draft is rejected and each
    round commits the verifier's own argmax: the output must equal plain
    greedy exactly. This is the cache-bookkeeping check."""
    m, cfg = _model()
    ctx = torch.randint(0, cfg.vocab_size, (1, 7))
    base = m.generate(ctx, max_new_tokens=16, temperature=0.0)
    spec = m.generate(ctx, max_new_tokens=16, temperature=0.0,
                      speculative=True, spec_drafter="mtp")
    assert torch.equal(base, spec)
    st = m.last_spec_stats
    assert st["tokens"] == 16 and 0.0 <= st["acceptance_rate"] <= 1.0
    assert st["forwards"] == st["rounds"] + 1          # prefill + one per round
    assert 1.0 <= st["tokens_per_forward"] <= 1 + cfg.mtp_heads


def test_mtp_speculative_with_oracle_drafts_accepts_everything():
    """Replace the heads with an oracle that proposes the true greedy
    continuation: every draft is accepted, output still equals greedy, and
    the round count collapses to ~N / (1 + K). Exercises the accept path and
    the accepted-prefix cache truncation."""
    m, cfg = _model()
    ctx = torch.randint(0, cfg.vocab_size, (1, 9))
    N = 18
    base = m.generate(ctx, max_new_tokens=N, temperature=0.0)
    K = cfg.mtp_heads

    real = m._mtp_drafts
    # Oracle drafter: the exact next K greedy tokens from a running cursor
    # (every round accepts all K and commits the bonus, so advance K + 1).
    cursor = {"pos": ctx.shape[1] + 1}   # first token committed by prefill

    def cursor_oracle(hidden_pos):
        d = base[:, cursor["pos"]:cursor["pos"] + K]
        cursor["pos"] += K + 1            # all K accepted + bonus each round
        if d.shape[1] < K:                # pad past the end (never checked)
            d = torch.cat([d, d.new_zeros(1, K - d.shape[1])], dim=1)
        return d
    m._mtp_drafts = cursor_oracle
    spec = m.generate(ctx, max_new_tokens=N, temperature=0.0,
                      speculative=True, spec_drafter="mtp")
    m._mtp_drafts = real
    assert torch.equal(base, spec)
    st = m.last_spec_stats
    assert st["acceptance_rate"] == 1.0
    assert st["rounds"] == -(-(N - 1) // (K + 1))    # ceil((N-1)/(K+1))
    assert st["tokens_per_forward"] > 2.0


def test_mtp_speculative_requires_heads_and_rejects_sampling():
    m0, cfg0 = _model(mtp_heads=0)
    ctx = torch.randint(0, cfg0.vocab_size, (1, 5))
    with pytest.raises(ValueError, match="mtp_heads"):
        m0.generate(ctx, max_new_tokens=4, speculative=True, spec_drafter="mtp")
    m, cfg = _model()
    with pytest.raises(ValueError, match="greedy-only"):
        m.generate(ctx, max_new_tokens=4, temperature=0.7, speculative=True,
                   spec_drafter="mtp")
    with pytest.raises(ValueError, match="spec_drafter"):
        m.generate(ctx, max_new_tokens=4, speculative=True, spec_drafter="eagle")


def test_mtp_speculative_batched_and_eos():
    m, cfg = _model()
    ctx = torch.randint(0, cfg.vocab_size, (3, 6))
    base = m.generate(ctx, max_new_tokens=12, temperature=0.0, eos_token_id=1)
    spec = m.generate(ctx, max_new_tokens=12, temperature=0.0, eos_token_id=1,
                      speculative=True, spec_drafter="mtp")
    assert base.shape == spec.shape
    assert torch.equal(base, spec)
