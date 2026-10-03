"""Tokenisation of (context, continuation) pairs in the lm-eval wrapper."""
from transformers import AutoTokenizer

from osrt.lm_eval_wrapper import OSRTLMEval


class _Stub(OSRTLMEval):
    """Only the tokenizer, no model: exercises the pure helpers."""

    def __init__(self):  # noqa: D107 — bypass model construction on purpose
        self._tok = AutoTokenizer.from_pretrained("tokenizer")

    def tok_encode(self, string, **kw):
        return self._tok.encode(string, add_special_tokens=False)


def test_pair_encoding_matches_joint_encoding_including_trailing_space():
    w = _Stub()
    for ctx, cont in [("He said: ", "hello"), ("The dog ran to the", " park"),
                      ("Answer:", " 42"), ("x = [1, 2,", " 3]")]:
        c, k = w._encode_pair(ctx, cont)
        assert c + k == w.tok_encode(ctx + cont), (ctx, cont)
        assert len(k) >= 1
        # The continuation ids are the tail of the joint encoding — scoring
        # them scores what the model sees in running text.
        assert c == w.tok_encode(ctx.rstrip()) or c == w.tok_encode(ctx)


def test_pair_encoding_keeps_empty_context_handling_to_caller():
    w = _Stub()
    c, k = w._encode_pair("", " park")
    assert c == [] and k == w.tok_encode(" park")


class _ChatStub(_Stub):
    def __init__(self):
        super().__init__()
        self._extract_answer_block = True


def test_chat_extraction_cuts_end_turn_and_surfaces_boxed_answers():
    w = _ChatStub()
    out = w._extract_answer(
        "First 3 apples, then 4: \\boxed{7}<|end_turn|><|user|>junk")
    assert out.endswith("\n#### 7") and "<|end_turn|>" not in out and "junk" not in out
    assert w._extract_answer("The answer is 12.<|end_turn|>") == "The answer is 12."


def test_chat_extraction_returns_code_fence_body_for_code_replies():
    w = _ChatStub()
    reply = "```python\ndef f(x):\n    return x + 1\n```\nThis adds one.<|end_turn|>"
    assert w._extract_answer(reply) == "def f(x):\n    return x + 1\n"
    prose = "Use a loop:\n```python\nfor i in x: pass\n```"
    assert w._extract_answer(prose) == prose  # fence not at the start: left alone


def test_chat_wrap_uses_render_chat_with_generation_prompt():
    w = _ChatStub()
    w._chat_format_generate = True
    w._chat_format_loglikelihood = False
    w._system_prompt = ""
    out = w._wrap_context("  Q: 2+2?\nA:  ", for_generate=True)
    assert out == "<|user|>Q: 2+2?\nA:<|assistant|>"
    w._system_prompt = "Be brief."
    assert w._wrap_context("hi", for_generate=True) == (
        "<|system|>Be brief.<|user|>hi<|assistant|>")
    assert w._wrap_context("hi", for_generate=False) == "hi"


def test_generate_helpers_group_stop_and_cut():
    w = _ChatStub()
    w._chat_format_generate = True
    w._max_temperature = 1.0
    w._default_temperature = 0.0
    w._default_top_p = 1.0
    w._default_top_k = 0
    w._default_repetition_penalty = 1.0
    w._max_gen_toks = 64
    st = w._gen_settings({"until": ["Question:", "\n\n"], "max_gen_toks": 32})
    assert st["until"] == ("Question:", "\n\n", "<|end_turn|>") and st["max_new"] == 32
    # Same kwargs -> same group key; a different cap -> a different group.
    k1 = tuple(sorted(w._gen_settings({"until": ["Question:", "\n\n"],
                                       "max_gen_toks": 32}).items()))
    assert k1 == tuple(sorted(st.items()))
    assert k1 != tuple(sorted(w._gen_settings({"until": ["Question:"]}).items()))
    stop_ids = w._stop_token_ids(st["until"])
    end_turn = w.tok_encode("<|end_turn|>")
    assert len(end_turn) == 1 and end_turn[0] in stop_ids
    assert all(len(w.tok_encode(s)) == 1 for s in st["until"]
               if w.tok_encode(s)[0] in stop_ids)
    eos = w.eot_token_id
    gen = [5, 6, end_turn[0], 7, eos, eos]
    assert w._cut_generated(gen, stop_ids) == [5, 6]
    assert w._cut_generated([5, 6, eos, eos], stop_ids) == [5, 6]
    assert w._cut_generated([5, 6], stop_ids) == [5, 6]
    assert w._postprocess("The answer is 4.\n\nQuestion: next", st["until"]) == (
        "The answer is 4.")
