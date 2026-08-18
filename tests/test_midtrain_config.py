"""Unit tests for the v6 midtrain stage: config + native-HRA load gate."""
from pathlib import Path

import pytest

from osrt.config import OSRTConfig
from osrt.hra import inject_hra
from osrt.model import OSRTForCausalLM
from osrt.train import load_model_state_or_raise

# Repo-root relative, never CWD relative: a bare relative path makes
# AutoTokenizer.from_pretrained fall back to a Hugging Face repo-id lookup
# and hit the network. v7's tokenizer is open (roadmap gate G2); skip
# cleanly when the artefact is absent.
TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "tokenizer"

pytestmark = pytest.mark.skipif(
    not (TOKENIZER_DIR / "tokenizer.json").is_file(),
    reason="tokenizer/ artefact absent — pending roadmap gate G2",
)

def tiny_config(**overrides) -> OSRTConfig:
    """Small config for fast CPU tests (mirrors tests/test_model.py)."""
    defaults = dict(
        dim=128, heads=4, head_dim=32,
        vocab_size=512, real_vocab_size=512,
        num_blocks=2, recursive_loops=2,
        num_routed_experts=8, top_k_experts=2,
        expert_hidden=64, shared_expert_hidden=128,
        adapter_rank=16, adapter_alpha=16.0,
        max_position_embeddings=64,
    )
    defaults.update(overrides)
    return OSRTConfig(**defaults)


def test_native_hra_checkpoint_loads_without_injection():
    """A model built straight from config (native HRA) round-trips its
    own state_dict with no inject_hra — the hra_native=True path."""
    cfg = tiny_config()
    src = OSRTForCausalLM(cfg)
    state = src.state_dict()

    dst = OSRTForCausalLM(cfg)  # native HRA present, NO inject_hra
    # Must not raise: keys match exactly.
    load_model_state_or_raise(dst, state, context="native-hra test")


def test_inject_hra_on_native_model_breaks_load():
    """Proves WHY the gate is needed: injecting HRALinear onto a model
    that already has native HRA changes the key namespace, so loading a
    native checkpoint then raises."""
    cfg = tiny_config()
    native_state = OSRTForCausalLM(cfg).state_dict()

    injected = OSRTForCausalLM(cfg)
    inject_hra(injected, rank=cfg.adapter_rank, scale=1.0,
               freeze_pretrained=False)

    with pytest.raises(RuntimeError, match="key mismatch"):
        load_model_state_or_raise(
            injected, native_state, context="inject-breaks-load test"
        )


def test_midtrain_config_values():
    """MidtrainConfig encodes the locked decisions (spec §2/§4.4)."""
    from osrt.train_config import MidtrainConfig

    cfg = MidtrainConfig()
    assert cfg.total_steps == 5_500
    assert cfg.peak_lr == 2e-4
    assert cfg.min_lr == 2e-5
    assert cfg.warmup_steps == 150
    assert cfg.eval_interval == 500          # eval every 500 (16 evals)
    # cosine anneals over the full total_steps (lr_anchor_step=0)
    assert cfg.phases["extend"]["end"] == 5_500
    assert cfg.lr_anchor_step == 0
    # native + trainable HRA
    assert cfg.hra_native is True
    assert cfg.hra_frozen is False
    assert cfg.hra_enabled is True
    # router exploration off
    assert cfg.router_gumbel_tau_init == 0.0
    # computed Muon LR: (peak_lr / foundation_peak 6e-4) * foundation muon 0.02
    assert cfg.muon_lr == 6.6e-3
    assert cfg.muon_min_lr == 6.6e-4
    # gate disabled — fully disabled (not just "high")
    assert cfg.early_stop_check_step > 9_000
    assert cfg.early_stop_check_step == 9_999_999
    # resume + prefix
    assert cfg.pretrained_checkpoint.endswith("osrt_v5_final.pt")
    assert cfg.stage_prefix == "midtrain"
    # Checkpointing ON — REQUIRED at seq 4096. The OFF throughput bet was
    # tested and OOM'd (78.4/79.2GB) on the sanity probe, so it's back on.
    assert cfg.gradient_checkpointing is True


def test_midtrain_phase_is_seq4096_math_mix():
    """The single 'extend' phase is seq 4096 with the knowledge mix."""
    from osrt.train_config import MidtrainConfig

    phase = MidtrainConfig().phases["extend"]
    assert phase["seq_len"] == 4096
    names = {d["name"] for d in phase["datasets"]}
    assert "nemotron-cc-math-4plus" in names
    assert "fineweb-edu" in names           # general anchor retained
    assert "cosmopedia-openstax" in names
    # weights: math/STEM/reasoning should dominate (~0.65)
    math_sci = sum(
        d["weight"] for d in phase["datasets"]
        if d["name"] in {
            "nemotron-cc-math-4plus", "nemotron-stem",
            "nemotron-math-textbooks", "nemotron-reasoning",
        }
    )
    assert 0.60 <= math_sci <= 0.70
    # per-phase sizing (the loop reads these, not the inherited top-level batch)
    assert phase["batch_size"] == 6
    assert phase["grad_accum_steps"] == 11
    # all dataset weights sum to ~1.0 — guards against a typo'd weight
    total_weight = sum(d["weight"] for d in phase["datasets"])
    assert abs(total_weight - 1.0) < 1e-9
    assert len(phase["datasets"]) == 7


def test_midtrain_sanity_writes_no_final():
    """Sanity config is a short probe that won't clobber a real final."""
    from osrt.train_config import MidtrainSanityConfig

    cfg = MidtrainSanityConfig()
    assert cfg.total_steps == 30
    assert cfg.save_final_checkpoint is False
    assert cfg.stage_prefix == "midtrain-sanity"
    assert cfg.compile_enabled is False
    # inherits the real seq/mix so VRAM is measured at production size
    assert cfg.phases["extend"]["seq_len"] == 4096


# ── SFT v1 tests (system-prompt instruction tuning) ──────────────────────

def test_reasoning_pools_split():
    from osrt.system_prompts import (
        REASONING_OFF,
        REASONING_ON,
        SYSTEM_PROMPTS,
        sample_system_prompt,
    )
    # A bare count is the wrong invariant — it fires on any addition while
    # missing the thing that actually breaks: `Random(0).choice(pool)` is the
    # historical eval default, and its result depends on POOL LENGTH. Growing a
    # pool can silently rebase every recorded number. The count is kept as a
    # tripwire, but the load-bearing assertion is that the pinned eval personas
    # do not move.
    assert len(REASONING_ON) == 14   # +word_problem_verify_0shot/1shot (2026-08-10)
    assert len(REASONING_OFF) >= 6
    assert SYSTEM_PROMPTS is REASONING_ON  # back-compat
    import random
    r = random.Random(0)
    assert sample_system_prompt(r, "on") in REASONING_ON
    assert sample_system_prompt(r, "off") in REASONING_OFF
    # default mode is "on" (preserves old single-pool callers)
    assert sample_system_prompt(r) in REASONING_ON
    import pytest
    with pytest.raises(ValueError):
        sample_system_prompt(r, "bogus")


def test_pinned_eval_personas_do_not_drift():
    """Every recorded acc_on/acc_off number was measured under these two.

    `Random(0).choice(pool)` resolving to them is an accident of pool length,
    not a guarantee. This asserts the accident still holds AND that the pinned
    constants name the same personas, so a future pool addition that would
    rebase the historical panel fails here rather than silently.
    """
    import random

    from osrt.system_prompts import (
        DEFAULT_EVAL_OFF,
        DEFAULT_EVAL_ON,
        REASONING_OFF,
        REASONING_ON,
        sample_system_prompt,
    )
    assert DEFAULT_EVAL_ON == "instruction_strict"
    assert DEFAULT_EVAL_OFF == "instruction_direct"
    names_on = {n for n, _ in REASONING_ON}
    names_off = {n for n, _ in REASONING_OFF}
    assert DEFAULT_EVAL_ON in names_on
    assert DEFAULT_EVAL_OFF in names_off
    # The eval must NOT depend on Random(seed).choice — that resolution moved
    # from instruction_strict to general_default the moment the pool went from
    # 13 to 14 entries, which would have rebased every recorded number. This
    # asserts the sampled value is now IRRELEVANT to the eval by showing it has
    # in fact diverged from the pinned default, so anyone reintroducing
    # sampling here fails loudly.
    sampled = sample_system_prompt(random.Random(0), "on")[0]
    assert sampled != DEFAULT_EVAL_ON, (
        "sampling coincidentally matches the pin again — re-check that "
        "run_reasoning_eval still resolves personas BY NAME"
    )


def test_few_shot_exemplars_cover_only_shot_personas():
    """The echo penalty must not fire on personas with no demonstration."""
    from osrt.system_prompts import FEW_SHOT_EXEMPLARS, REASONING_ON

    assert "minimal_format" not in FEW_SHOT_EXEMPLARS
    assert "instruction_strict" not in FEW_SHOT_EXEMPLARS
    assert "word_problem_verify_1shot" in FEW_SHOT_EXEMPLARS
    # the 0-shot twin must NEVER be registered — it has no demonstration, so an
    # echo penalty against it is meaningless
    assert "word_problem_verify_0shot" not in FEW_SHOT_EXEMPLARS
    for name, ex in FEW_SHOT_EXEMPLARS.items():
        assert ex.strip(), f"{name} registered an empty exemplar"
        assert "shot" in name, f"{name} has an exemplar but no 'shot' in its name"
        # the exemplar must be the DEMONSTRATION only, never the instructions
        assert "Do not repeat" not in ex, (
            f"{name}'s exemplar swallowed an instruction; a model quoting its "
            f"own instructions would be penalised"
        )
    names = {n for n, _ in REASONING_ON}
    assert set(FEW_SHOT_EXEMPLARS) <= names


def test_verify_personas_are_matched_pairs():
    """0-shot and 1-shot must differ ONLY by the demonstration."""
    from osrt.system_prompts import (
        FEW_SHOT_EXEMPLARS,
        VERIFY_EXEMPLAR_ANCHORS,
        get_by_name,
    )
    zero = get_by_name("word_problem_verify_0shot")
    one = get_by_name("word_problem_verify_1shot")
    assert one.startswith(zero), (
        "instructions differ between the pair, so a 0-shot vs 1-shot "
        "comparison would confound 'has an exemplar' with 'has different "
        "instructions'"
    )
    assert FEW_SHOT_EXEMPLARS["word_problem_verify_1shot"] in one
    assert FEW_SHOT_EXEMPLARS["word_problem_verify_1shot"] not in zero
    # anchors exist only in the 1-shot prompt; that asymmetry is what makes a
    # numeric-anchoring comparison possible
    for a in VERIFY_EXEMPLAR_ANCHORS:
        assert a in one
    assert not any(a in zero for a in ("500", "625", "1.25"))


def test_few_shot_echo_penalty_catches_copying_not_idiom():
    from osrt.rewards import compute_reward, few_shot_echo_score
    from osrt.system_prompts import FEW_SHOT_EXEMPLARS

    ex = FEW_SHOT_EXEMPLARS["word_problem_verify_1shot"]
    kw = dict(think_open="<|think|>", think_close="<|/think|>",
              answer_open="<|answer|>", answer_close="<|/answer|>",
              max_tokens=768, completion_tokens=120)

    copied = ("<|think|>Let C be the cost price. A 25% profit means the selling "
              "price is 1.25C, so 1.25C = 625. Then C = 625 / 1.25 = 500. "
              "Check: 500 x 1.25 = 625, which matches the selling price "
              "given.<|/think|><|answer|>500<|/answer|>")
    own = ("<|think|>Let P be the price. A 40% discount means 0.6P = 300, so "
           "P = 300 / 0.6 = 500. Check: 500 x 0.6 = 300, matches.<|/think|>"
           "<|answer|>500<|/answer|>")

    assert few_shot_echo_score(copied, ex)[0] > 0
    assert few_shot_echo_score(own, ex)[0] == 0, (
        "the SAME reasoning discipline with different numbers must not be "
        "penalised — that pattern is what the exemplar exists to teach"
    )
    # inactive without an exemplar, so unexemplified personas are untouched
    assert few_shot_echo_score(copied, "")[0] == 0

    r_copy, b_copy = compute_reward(copied, "500", few_shot_exemplar=ex, **kw)
    r_own, b_own = compute_reward(own, "500", few_shot_exemplar=ex, **kw)
    assert b_copy["few_shot_echo_penalty"] < 0
    assert b_own["few_shot_echo_penalty"] == 0.0
    assert r_own > r_copy, "copying must never outrank solving"
    # and copying while WRONG must be worse than failing honestly
    wrong_copy, _ = compute_reward(
        copied.replace("<|answer|>500", "<|answer|>499"), "500",
        few_shot_exemplar=ex, **kw)
    wrong_honest, _ = compute_reward(
        own.replace("<|answer|>500", "<|answer|>499"), "500", **kw)
    assert wrong_copy < wrong_honest

    # special tags are stripped: they appear in BOTH sides by construction and
    # would otherwise guarantee a match that has nothing to do with copying.
    assert few_shot_echo_score("<|think|><|/think|><|answer|>1<|/answer|>", ex)[0] == 0


def test_format_tulu_single_vs_multi_turn():
    from osrt.sft_data import FORMAT_FN, format_tulu
    assert "tulu" in FORMAT_FN
    single = {"messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]}
    assert format_tulu(single) == ("What is 2+2?", "", "4")
    # multi-turn (>1 user turn) is skipped → empties → SFTStream drops it
    multi = {"messages": [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"},
    ]}
    assert format_tulu(multi) == ("", "", "")
    # malformed
    assert format_tulu({"messages": []}) == ("", "", "")


def test_sft_system_turn_masking():
    """The <|system|> turn joins the MASKED prefix; the response is trained."""
    import random

    from transformers import AutoTokenizer

    from osrt.sft_data import IGNORE_INDEX
    from osrt.system_prompts import sample_system_prompt
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    _, persona = sample_system_prompt(random.Random(0), "off")
    prompt = f"<|system|>{persona}<|user|>Q?<|assistant|>"
    resp = f"<|think|><|/think|><|answer|>A<|/answer|>{tok.eos_token}"
    pids = tok.encode(prompt, add_special_tokens=False)
    rids = tok.encode(resp, add_special_tokens=False)
    labels = [IGNORE_INDEX] * len(pids) + rids
    # the <|system|> token is in the masked prefix. Look the id up rather than
    # hardcoding it: the literal 13 was the v6 vocab's id and silently coupled
    # this test to one tokenizer (it broke on the G2 swap to the v7 vocab).
    sys_id = tok.convert_tokens_to_ids("<|system|>")
    assert sys_id != tok.unk_token_id, "<|system|> missing from the tokenizer"
    assert sys_id in pids
    assert all(x == IGNORE_INDEX for x in labels[:len(pids)])
    assert all(x != IGNORE_INDEX for x in labels[len(pids):])


def test_sftv1_config_values():
    from osrt.sft_data import FORMAT_FN
    from osrt.train_config import SFTv1Config
    c = SFTv1Config()
    assert c.pretrained_checkpoint.endswith("osrt_v5_midtrain_final.pt")
    assert c.seq_len == 2048
    assert c.stage_prefix == "sft_v1"
    assert c.system_tag == "<|system|>"
    assert c.min_response_tokens == 150
    assert c.hra_native is True
    assert c.hra_enabled is True
    # gradient checkpointing required at seq2048 for the v6 601M model
    # (un-checkpointed batch8 OOM'd on the sanity gate); eff batch stays 64.
    assert c.gradient_checkpointing is True
    assert c.batch_size * c.grad_accum_steps == 64
    ds = c.datasets
    assert abs(sum(d["weight"] for d in ds) - 1.0) < 1e-9
    # every dataset has a registered format + a valid reasoning_mode
    for d in ds:
        assert d["format"] in FORMAT_FN
        assert d["reasoning_mode"] in ("on", "off")
    # ~35% reasoning-on (math)
    on = sum(d["weight"] for d in ds if d["reasoning_mode"] == "on")
    assert 0.30 <= on <= 0.40


# ── SFT v2 tests (reasoning distillation via the rollout-loader path) ─────


def test_sftv2_config_values():
    from osrt.train_config import SFTv2Config
    c = SFTv2Config()
    # base = midtrain2 step_1750 (ppl 28.2) — the best INTACT midtrain2
    # artifact; the step-2000 final save was truncated on the volume.
    assert c.pretrained_checkpoint.endswith("osrt_v5_midtrain2_step_1750.pt")
    assert c.stage_prefix == "sft_v2"
    # the MOPD rollout-loader override → run_pretrain_extend uses sft_v2.jsonl
    assert c.rollout_dataset_path.endswith("sft_v2.jsonl")
    # native + trainable HRA (v6), checkpointing on at seq 4096
    assert c.hra_native is True and c.hra_frozen is False
    assert c.gradient_checkpointing is True
    # gentle SFT schedule — NOT midtrain's continued-pretrain 2e-4
    assert c.peak_lr == 1e-5 and c.min_lr == 1e-6
    assert c.total_steps == 1_000 and c.warmup_steps == 100
    assert c.lr_anchor_step == 0          # fresh cosine over the full run
    # recursive depth kept active during SFT
    assert c.aux_loop_loss_weight == 0.05 and c.loop_dropout_prob == 0.10
    # phase sizing the rollout path reads: seq 4096, eff batch 64
    ph = c.phases["extend"]
    assert ph["seq_len"] == 4096
    assert ph["batch_size"] * ph["grad_accum_steps"] == 64
    # in-loop perplexity eval disabled (reasoning eval runs offline on ckpts)
    assert c.eval_interval > 9_000
    assert c.ckpt_interval == 200


def test_midtrain_extend_config_values():
    from osrt.train_config import MidtrainExtendConfig
    c = MidtrainExtendConfig()
    # extended continued-pretrain from the clean midtrain base
    assert c.pretrained_checkpoint.endswith("osrt_v5_midtrain_final.pt")
    assert c.stage_prefix == "midtrain2"
    # GENTLE re-warm cosine (the 1e-4 was too hot: ppl rose 30→34, flat)
    assert c.peak_lr == 3e-5 and c.min_lr == 1e-5
    assert c.lr_anchor_step == 0
    # 2000-step cosine spanning two ~$30 Modal workspaces (chain at ~1000)
    assert c.total_steps == 2_000
    assert c.dataloader_num_workers == 1
    # native HRA + checkpointing carried from MidtrainConfig
    assert c.hra_native is True and c.gradient_checkpointing is True
    ph = c.phases["extend"]
    assert ph["seq_len"] == 4096
    assert abs(sum(d["weight"] for d in ph["datasets"]) - 1.0) < 1e-9
    # reasoning/instruction-heavy reweight: math+STEM-SFT+reasoning ≥ 0.70
    heavy = sum(d["weight"] for d in ph["datasets"] if d["name"] in {
        "nemotron-cc-math-4plus", "nemotron-stem",
        "nemotron-math-textbooks", "nemotron-reasoning",
    })
    assert heavy >= 0.70
    # NOT a rollout/SFT run — full-sequence streaming pretrain
    assert getattr(c, "rollout_dataset_path", None) is None


def test_midtrain_extend_sanity_writes_no_final():
    from osrt.train_config import MidtrainExtendSanityConfig
    c = MidtrainExtendSanityConfig()
    assert c.total_steps == 30
    assert c.save_final_checkpoint is False
    assert c.compile_enabled is False
    assert c.stage_prefix == "midtrain2-sanity"
    assert c.phases["extend"]["seq_len"] == 4096


def test_sftv2_sanity_writes_no_final():
    from osrt.train_config import SFTv2SanityConfig
    c = SFTv2SanityConfig()
    assert c.total_steps == 30
    assert c.save_final_checkpoint is False
    assert c.compile_enabled is False
    assert c.stage_prefix == "sft_v2-sanity"
    # inherits the real seq/rollout so the probe measures production VRAM
    assert c.phases["extend"]["seq_len"] == 4096
    assert c.rollout_dataset_path.endswith("sft_v2.jsonl")
