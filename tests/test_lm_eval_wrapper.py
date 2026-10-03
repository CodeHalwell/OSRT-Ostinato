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
