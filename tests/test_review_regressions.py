"""Model-side regression tests for the review fixes of 2026-09-30.

Each test pins one behaviour the review found wrong or unguarded in
`src/osrt/model.py` / `config.py` / `presets.py`. CPU only, tiny shapes.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

import osrt.model as model_mod
from osrt.config import OSRTConfig
from osrt.hra import inject_hra
from osrt.model import OSRTForCausalLM
from osrt.presets import OSRT_V7_TOKEN_IDS, build_v7_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny(**overrides) -> OSRTConfig:
    base = dict(
        dim=64, heads=2, head_dim=32, num_kv_heads=1,
        vocab_size=128, real_vocab_size=100,
        num_blocks=1, recursive_loops=2,
        num_routed_experts=4, top_k_experts=2,
        expert_hidden=64, shared_expert_hidden=64,
        use_hra=False, max_position_embeddings=64,
        situ_glu=False, swiglu_clamp=None,
    )
    base.update(overrides)
    return OSRTConfig(**base)


def _model(seed: int = 0, **overrides) -> OSRTForCausalLM:
    torch.manual_seed(seed)
    return OSRTForCausalLM(tiny(**overrides)).eval()


# ── Router: gates from the pre-bias affinity, computed in fp32 ────────────

@pytest.mark.parametrize("affinity", ["sqrt_softplus", "softmax"])
def test_balance_bias_steers_selection_only_by_default(affinity):
    """A bias too small to change the top-k must leave the routed output
    bit-identical (DeepSeek-V3 semantics). With router_bias_in_gates=True
    (the v6 behaviour) the gates move with the bias."""
    x = torch.randn(2, 6, 64)
    for in_gates, expect_same in ((False, True), (True, False)):
        m = _model(router_affinity=affinity, router_balance_mode="quantile",
                   router_bias_in_gates=in_gates)
        moe = m.model.blocks[0].moe
        _, out0 = moe(x, loop_idx=0)
        moe.router_balance_bias[0] += torch.tensor([1e-4, -1e-4, 2e-4, -2e-4])
        _, out1 = moe(x, loop_idx=0)
        assert torch.equal(out0, out1) is expect_same, (affinity, in_gates)


def test_router_logits_are_computed_in_fp32_under_autocast(monkeypatch):
    m = _model()
    moe = m.model.blocks[0].moe
    seen: list[torch.dtype] = []
    real_linear = F.linear

    def spy(inp, weight, bias=None):
        if tuple(weight.shape) == tuple(moe.router.weight.shape):
            seen.append(inp.dtype)
        return real_linear(inp, weight, bias)

    monkeypatch.setattr(model_mod.F, "linear", spy)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        m(torch.randint(0, 100, (1, 8)))
    assert seen and all(d == torch.float32 for d in seen)


# ── Preset and config plumbing ────────────────────────────────────────────

def test_v7_preset_token_ids_match_the_shipped_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(REPO, "tokenizer"))
    cfg = build_v7_config()
    assert cfg.bos_token_id == tok.bos_token_id == 49152
    assert cfg.eos_token_id == tok.eos_token_id == 49153
    assert cfg.pad_token_id == tok.pad_token_id == 49154
    for attr, token in (
        ("unk_token_id", "<|unknown|>"), ("fim_prefix_id", "<|fim_prefix|>"),
        ("fim_middle_id", "<|fim_middle|>"), ("fim_suffix_id", "<|fim_suffix|>"),
        ("think_open_id", "<|think|>"), ("think_close_id", "<|/think|>"),
        ("answer_open_id", "<|answer|>"), ("answer_close_id", "<|/answer|>"),
        ("user_token_id", "<|user|>"), ("assistant_token_id", "<|assistant|>"),
        ("system_token_id", "<|system|>"),
    ):
        assert getattr(cfg, attr) == tok.convert_tokens_to_ids(token), attr
        assert OSRT_V7_TOKEN_IDS[attr] < cfg.real_vocab_size


def test_config_rejects_token_ids_outside_the_real_vocab():
    with pytest.raises(ValueError, match="outside the real vocabulary"):
        tiny(eos_token_id=120)          # padded row, not a real token
    with pytest.raises(ValueError, match="real_vocab_size"):
        tiny(vocab_size=64, real_vocab_size=100)


def test_config_validates_aux_weights_and_loop_dropout():
    with pytest.raises(ValueError, match="per_loop_aux_weights"):
        tiny(recursive_loops=3, per_loop_aux_weights=[0.5])
    with pytest.raises(ValueError, match="loop_dropout_min_loops"):
        tiny(recursive_loops=2, loop_dropout_prob=0.5, loop_dropout_min_loops=5)
    tiny(recursive_loops=3, per_loop_aux_weights=[0.2, 0.1])   # fine


def test_heuristic_controller_above_eight_experts_warns():
    with pytest.warns(UserWarning, match="quantile"):
        tiny(num_routed_experts=28, top_k_experts=4, router_balance_mode="heuristic")


# ── Attention mask under gradient checkpointing ───────────────────────────

def test_attention_mask_is_honoured_under_gradient_checkpointing():
    m = _model(router_capacity_factor=100.0)
    m.train()
    ids = torch.randint(20, 100, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[1, :3] = 0                      # row 1 left-padded by 3

    def logits(ckpt: bool, use_mask: bool):
        m.model._osrt_grad_ckpt = ckpt
        with torch.no_grad():
            out = m(ids, attention_mask=mask if use_mask else None)
        return out.logits[1, 3:]

    ref = logits(False, True)
    assert torch.allclose(logits(True, True), ref, atol=1e-6)
    assert not torch.allclose(logits(True, False), ref, atol=1e-3)


# ── generate(): padding, static cache, sink ───────────────────────────────

def test_generate_rejects_right_padding():
    m = _model()
    ids = torch.randint(20, 100, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.long)
    mask[0, -2:] = 0                     # right padding
    with pytest.raises(ValueError, match="LEFT padding"):
        m.generate(ids, max_new_tokens=2, attention_mask=mask)


def test_static_cache_generates_past_max_position_embeddings():
    m = _model(max_position_embeddings=16)
    prompt = torch.randint(20, 100, (1, 8))
    latent = m.generate(prompt, max_new_tokens=12, eos_token_id=None)
    static = m.generate(prompt, max_new_tokens=12, eos_token_id=None,
                        cache_impl="static")
    assert latent.shape[1] == static.shape[1] == 20
    assert torch.equal(latent, static)


def test_static_cache_refuses_attention_sink():
    m = _model(attention_sink=True)
    with pytest.raises(ValueError, match="attention sink"):
        m.generate(torch.randint(20, 100, (1, 6)), max_new_tokens=2,
                   cache_impl="static")


def test_sink_attention_honours_left_padding():
    m = _model(attention_sink=True)
    with torch.no_grad():
        for blk in m.model.blocks:
            blk.sink_logits.normal_()
    prompts = [torch.randint(20, 100, (1, n)) for n in (4, 6, 7)]
    singles = [m.generate(p, max_new_tokens=4, eos_token_id=None) for p in prompts]
    width = 7
    batch = torch.zeros(3, width, dtype=torch.long)
    mask = torch.zeros(3, width, dtype=torch.long)
    for i, p in enumerate(prompts):
        batch[i, width - p.shape[1]:] = p
        mask[i, width - p.shape[1]:] = 1
    out = m.generate(batch, max_new_tokens=4, eos_token_id=None, attention_mask=mask)
    for i, s in enumerate(singles):
        assert torch.equal(out[i, width:], s[0, prompts[i].shape[1]:]), i


# ── Speculative decoding stops where greedy stops ────────────────────────

@pytest.mark.parametrize("drafter", ["mtp", "loops"])
def test_speculative_output_equals_greedy_including_eos_tail(drafter):
    m = _model(mtp_heads=2, aux_loop_loss_weight=0.05)
    prompt = torch.randint(20, 100, (2, 5))
    free = m.generate(prompt, max_new_tokens=8, eos_token_id=None)
    # Force an EOS three tokens in for row 0: use that token id as EOS.
    eos = int(free[0, prompt.shape[1] + 2])
    greedy = m.generate(prompt, max_new_tokens=8, eos_token_id=eos)
    spec = m.generate(prompt, max_new_tokens=8, eos_token_id=eos,
                      speculative=True, spec_drafter=drafter, spec_draft_tokens=3)
    assert greedy.shape == spec.shape, (greedy.shape, spec.shape)
    assert torch.equal(greedy, spec)


# ── HRA adapters must be live under the grouped-GEMM expert path ─────────

def test_injected_hra_adapters_are_live_under_grouped_gemm():
    m = _model(moe_grouped_gemm=True)
    ids = torch.randint(20, 100, (2, 6))
    with torch.no_grad():
        before = m(ids).logits.clone()
    params = inject_hra(m, rank=4)
    assert params
    with torch.no_grad():
        for p in params:
            p.normal_()
        after = m(ids).logits
    assert not torch.allclose(before, after, atol=1e-4)
    m.train()
    m(ids, labels=ids, return_logits=True).loss.backward()
    routed = [p for n, p in m.named_parameters()
              if ".experts." in n and "adapter" in n]
    assert routed and all(p.grad is not None and p.grad.abs().sum() > 0
                          for p in routed)


# ── Fused main head ──────────────────────────────────────────────────────

def test_fused_main_head_matches_unfused_loss_and_gradients():
    ids = torch.randint(20, 100, (2, 10))
    labels = ids.clone()
    labels[0, :3] = -100
    ref = _model(fused_cross_entropy_chunks=0).train()
    fused = _model(fused_cross_entropy_chunks=4).train()
    fused.load_state_dict(ref.state_dict())
    out_ref = ref(ids, labels=labels)
    out_fused = fused(ids, labels=labels)
    assert out_fused.logits is None and out_ref.logits is not None
    assert torch.allclose(out_ref.loss, out_fused.loss, atol=1e-5)
    out_ref.loss.backward()
    out_fused.loss.backward()
    g_ref = ref.model.embedding.weight.grad
    g_fused = fused.model.embedding.weight.grad
    assert torch.allclose(g_ref, g_fused, atol=1e-5)
    # opt back in to logits, and eval is never fused
    assert fused(ids, labels=labels, return_logits=True).logits is not None
    fused.eval()
    assert fused(ids, labels=labels).logits is not None


# ── Capacity cap is explicitly a loop-path feature ───────────────────────

def test_grouped_path_ignores_capacity_factor_and_reports_no_drops():
    m = _model(moe_grouped_gemm=True, router_capacity_factor=1.01).train()
    m(torch.randint(20, 100, (2, 16)))
    moe = m.model.blocks[0].moe
    assert all(r == 0.0 for r in moe.last_drop_rate)


def test_bmm_decode_path_matches_eager_on_cpu_after_prepack():
    m = _model(moe_grouped_gemm=True, mtp_heads=0)
    ids = torch.randint(20, 100, (1, 4))          # N*K = 8 <= 32 -> bmm path
    with torch.no_grad():
        eager = m(ids).logits.clone()
        m.optimize_for_inference(compile_model=False)
        packed = m(ids).logits
    assert torch.equal(eager, packed)
