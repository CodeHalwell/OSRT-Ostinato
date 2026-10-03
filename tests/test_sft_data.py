"""SFT stream: assistant-only loss mask, indentation normalisation,
formats."""
import torch
from transformers import AutoTokenizer

from osrt.chat_format import ASSISTANT, END_TURN, USER, encode_with_markers
from osrt.sft_data import (
    IGNORE_INDEX,
    assistant_labels,
    format_sft,
    normalise_python_indent,
)

TOK = AutoTokenizer.from_pretrained("tokenizer")
A_ID = TOK.convert_tokens_to_ids(ASSISTANT)
E_ID = TOK.convert_tokens_to_ids(END_TURN)


def test_labels_cover_exactly_the_assistant_spans():
    text = (f"{USER}What is 2+2?{ASSISTANT}4{END_TURN}"
            f"{USER}And 3+3?{ASSISTANT}6{END_TURN}")
    ids = torch.tensor(encode_with_markers(TOK, text) + [TOK.eos_token_id])
    labels, carried = assistant_labels(ids, A_ID, E_ID, False)
    assert carried is False
    kept = [int(t) for t, lab in zip(ids, labels) if lab != IGNORE_INDEX]
    # Each answer span is the answer tokens plus its <|end_turn|>; nothing
    # from the prompts, the <|assistant|> markers, or the trailing EOS.
    assert kept == encode_with_markers(TOK, f"4{END_TURN}") + encode_with_markers(
        TOK, f"6{END_TURN}")
    assert all(int(t) != A_ID for t, lab in zip(ids, labels) if lab != IGNORE_INDEX)
    assert labels[-1] == IGNORE_INDEX  # EOS after <|end_turn|> is not a target


def test_span_state_carries_across_window_boundaries():
    text = f"{USER}Q{ASSISTANT}a long answer that spans two windows{END_TURN}"
    ids = encode_with_markers(TOK, text)
    cut = ids.index(A_ID) + 3  # split inside the answer
    w1, w2 = torch.tensor(ids[:cut]), torch.tensor(ids[cut:])
    l1, state = assistant_labels(w1, A_ID, E_ID, False)
    assert state is True
    l2, state = assistant_labels(w2, A_ID, E_ID, state)
    assert state is False
    kept = [int(t) for t, lab in list(zip(w1, l1)) + list(zip(w2, l2))
            if lab != IGNORE_INDEX]
    assert kept == ids[ids.index(A_ID) + 1:]


def test_python_indent_normalisation_targets_python_only():
    fenced = ("Here:\n```python\ndef f():\n\treturn 1\n```\n"
              "and go:\n```go\n\tfmt.Println()\n```")
    out = normalise_python_indent(fenced)
    assert "def f():\n    return 1" in out and "\tfmt.Println()" in out
    raw = "def g(x):\n\tif x:\n\t\treturn x\n\treturn 0"
    assert normalise_python_indent(raw) == (
        "def g(x):\n    if x:\n        return x\n    return 0")
    prose = "Use\ttabs in prose freely."
    assert normalise_python_indent(prose) == prose


def test_format_sft_accepts_every_layout_and_normalises_assistant_only():
    msgs = {"messages": [{"role": "user", "content": "code\twith tab"},
                         {"role": "assistant", "content": "def f():\n\treturn 1"}]}
    out = format_sft(msgs)
    assert out == f"{USER}code\twith tab{ASSISTANT}def f():\n    return 1{END_TURN}"
    io = format_sft({"input": "q", "output": "a"})
    assert io == f"{USER}q{ASSISTANT}a{END_TURN}"
    om = format_sft({"problem": "p", "generated_solution": "s"})
    assert om == f"{USER}p{ASSISTANT}s{END_TURN}"
    assert format_sft({"text": "plain"}) == ""
    assert format_sft({"messages": [{"role": "tool", "content": "x"}]}) == ""
