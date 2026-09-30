"""The OSRT chat contract: `render_chat`, its Jinja twin, and marker-aware encoding.

Uses the real tokenizer under `tokenizer/`, so these also pin the shipped
`chat_template` and the marker ids of the contract.
"""
from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from osrt.chat_format import (
    CHAT_TEMPLATE,
    OSRT_MARKERS,
    encode_with_markers,
    render_chat,
)
from osrt.tokenizer_contract import OSTINATO_SPECIAL_TOKEN_IDS

U, A, S, E = "<|user|>", "<|assistant|>", "<|system|>", "<|end_turn|>"
SMOLLM2_CONTROL_IDS = set(range(17))   # <|endoftext|> ... <empty_output>


def u(c):
    return {"role": "user", "content": c}


def a(c):
    return {"role": "assistant", "content": c}


def s(c):
    return {"role": "system", "content": c}


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("tokenizer")


# ── render_chat ──────────────────────────────────────────────────────────

def test_single_pair_and_system_prefix():
    assert render_chat([u("hi"), a("hello")]) == f"{U}hi{A}hello{E}"
    assert render_chat([s("be brief"), u("hi"), a("hello")]) == \
        f"{S}be brief{U}hi{A}hello{E}"


def test_multi_turn_has_no_whitespace_between_markers():
    out = render_chat([s("sys"), u("q1"), a("a1"), u("q2"), a("a2")])
    assert out == f"{S}sys{U}q1{A}a1{E}{U}q2{A}a2{E}"
    assert "\n<|" not in out and "<|end_turn|>\n" not in out
    # Content keeps its internal newlines; only the edges are trimmed.
    out = render_chat([u(" q\nmore \n"), a("\nline1\nline2\n\n")])
    assert out == f"{U}q\nmore{A}line1\nline2{E}"


def test_role_aliases():
    sharegpt = [{"role": "human", "content": "q"}, {"role": "gpt", "content": "a"}]
    gemini = [{"role": "User", "content": "q"}, {"role": "model", "content": "a"}]
    assert render_chat(sharegpt) == render_chat(gemini) == f"{U}q{A}a{E}"


def test_trailing_user_turn_dropped_unless_generation_prompt():
    convo = [u("q1"), a("a1"), u("q2")]
    assert render_chat(convo) == f"{U}q1{A}a1{E}"
    assert render_chat(convo, add_generation_prompt=True) == f"{U}q1{A}a1{E}{U}q2{A}"
    assert render_chat([u("q")], add_generation_prompt=True) == f"{U}q{A}"
    assert render_chat([s("sys"), u("q")], add_generation_prompt=True) == \
        f"{S}sys{U}q{A}"


def test_rows_without_a_complete_pair_render_empty():
    assert render_chat([]) == ""
    assert render_chat([u("q")]) == ""
    assert render_chat([s("sys")]) == ""
    assert render_chat([s("sys"), u("q")]) == ""
    assert render_chat([a("answer with no question")]) == ""
    assert render_chat([], add_generation_prompt=True) == ""


def test_non_string_or_blank_content_renders_empty():
    assert render_chat([u(123), a("x")]) == ""
    assert render_chat([u("q"), a(None)]) == ""
    assert render_chat([u("q"), a([{"type": "text", "text": "parts"}])]) == ""
    assert render_chat([u("   "), a("x")]) == ""
    assert render_chat([u("q"), a("\n")]) == ""
    assert render_chat([s(""), u("q"), a("x")]) == ""


def test_unsupported_shapes_render_empty():
    assert render_chat([u("q"), {"role": "tool", "content": "42"}, a("x")]) == ""
    assert render_chat([u("q"), a("x"), s("late system")]) == ""
    assert render_chat([u("q"), "not a dict", a("x")]) == ""
    assert render_chat("not a list") == ""
    assert render_chat([{"content": "no role"}, a("x")]) == ""
    assert render_chat([{"role": None, "content": "x"}, a("x")]) == ""


# ── the Jinja twin ───────────────────────────────────────────────────────

def test_shipped_tokenizer_carries_the_template(tok):
    assert tok.chat_template == CHAT_TEMPLATE


@pytest.mark.parametrize("agp", [False, True])
def test_apply_chat_template_matches_render_chat_byte_for_byte(tok, agp):
    cases = [
        [u("hi"), a("hello")],
        [s("be brief "), u(" hi\n"), a("hello\n\n")],
        [u("q1"), a("a1"), u("q2"), a("a2")],
        [s("sys"), u("q1"), a("a1\nwith\nlines"), u("q2"), a("a2"), u("q3")],
        [u("q")],
        [s("sys"), u("q")],
        [u("unicode ✓ 日本語"), a("<|think|>t<|/think|><|answer|>x<|/answer|>")],
        # Aliased and malformed inputs: the twin must apply the same
        # normalisation and validation, not just agree on well-formed rows.
        [{"role": "human", "content": "q"}, {"role": "gpt", "content": "a"}],
        [{"role": "User", "content": "q"}, {"role": "model", "content": " a "}],
        [{"role": " Assistant", "content": "a"}],
        [a("answer with no question")],
        [u("q1"), u("q2"), a("a")],
        [u("q"), a("x"), u("q2"), u("q3")],
        [s("sys")],
        # (an empty list is covered on render_chat alone: transformers'
        # apply_chat_template indexes messages[0] before rendering)
        [u("   "), a("x")],
        [u("q"), a("\n")],
        [u("q"), a("")],
        [u(123), a("x")],
        [u("q"), a(None)],
        [u("q"), a([{"type": "text", "text": "parts"}])],
        [s(""), u("q"), a("x")],
        [u("q"), a("x"), s("late system")],
        [s("one"), s("two"), u("q"), a("x")],
    ]
    for messages in cases:
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=agp)
        assert rendered == render_chat(messages, add_generation_prompt=agp), messages


@pytest.mark.parametrize("bad", [
    {"role": "tool", "content": "42"},
    {"role": "function", "content": "x"},
    {"content": "no role"},
    {"role": None, "content": "x"},
])
def test_template_refuses_unsupported_roles(tok, bad):
    """The one deliberate divergence from `render_chat` (which returns "" so the
    data stream skips the row): at inference an error beats an empty prompt."""
    assert render_chat([u("q"), bad, a("x")]) == ""
    with pytest.raises(Exception, match="unsupported role"):
        tok.apply_chat_template([u("q"), bad, a("x")], tokenize=False)


# ── encode_with_markers ──────────────────────────────────────────────────

def test_markers_become_their_single_ids(tok):
    ids = encode_with_markers(tok, "x<|assistant|>y")
    assert ids == [104, 49164, 105]
    assert ids == tok.encode("x<|assistant|>y", add_special_tokens=False)
    assert encode_with_markers(tok, "a<|end_turn|><|user|>b")[1:3] == [49166, 49163]
    # All 32 contract markers, in id order, with nothing between them.
    assert encode_with_markers(tok, "".join(OSRT_MARKERS)) == list(range(49152, 49184))
    for marker, tid in OSTINATO_SPECIAL_TOKEN_IDS.items():
        assert encode_with_markers(tok, marker) == [tid], marker


def test_legacy_smollm2_specials_are_ordinary_text(tok):
    assert tok.encode("<|endoftext|>", add_special_tokens=False) == [0]   # the trap
    ids = encode_with_markers(tok, "<|endoftext|>")
    assert ids != [0] and not (set(ids) & SMOLLM2_CONTROL_IDS)
    assert tok.decode(ids) == "<|endoftext|>"
    web = "chat log <|im_start|>user hi<|im_end|>\n<file_sep>next<repo_name>x"
    ids = encode_with_markers(tok, web)
    assert not (set(ids) & SMOLLM2_CONTROL_IDS)
    assert tok.decode(ids) == web


def test_plain_text_round_trips_to_the_plain_encode(tok):
    for text in ("hello world", "  leading spaces\n\nand\tbreaks", "1234567 × 89",
                 "def f(x):\n    return x < 2 | 3", "日本語のテキスト", ""):
        assert encode_with_markers(tok, text) == \
            tok.encode(text, add_special_tokens=False), repr(text)


def test_rendered_chat_encodes_to_the_contract_ids_and_decodes_back(tok):
    text = render_chat([s("sys"), u("What is 2+2?"), a("<|answer|>4<|/answer|>")])
    ids = encode_with_markers(tok, text)
    order = [i for i in ids if i in (49165, 49163, 49164, 49161, 49162, 49166)]
    assert order == [49165, 49163, 49164, 49161, 49162, 49166]
    assert tok.decode(ids) == text
