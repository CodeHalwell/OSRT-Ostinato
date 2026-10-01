"""Per-dataset row gating and the phase-table weight check (data plan §1, §6)."""
import random

import pytest

from osrt.data import FORMAT_FN_PRETRAIN, row_passes
from osrt.train_config import PretrainConfig


def test_filter_is_an_allow_list_and_missing_field_rejects():
    cfg = {"filter": {"language": ["Python", "Rust"]}}
    rng = random.Random(0)
    assert row_passes(cfg, {"language": "Rust", "content": "x"}, rng)
    assert not row_passes(cfg, {"language": "COBOL", "content": "x"}, rng)
    assert not row_passes(cfg, {"content": "x"}, rng)          # no field → reject
    assert row_passes({"filter": {"lang": "python"}}, {"lang": "python"}, rng)


def test_subsample_is_an_acceptance_probability_with_default():
    cfg = {"subsample": {"language": {"Python": 0.25, "*": 1.0}}}
    rng = random.Random(1234)
    n = 20_000
    kept_py = sum(row_passes(cfg, {"language": "Python"}, rng) for _ in range(n))
    kept_go = sum(row_passes(cfg, {"language": "Go"}, rng) for _ in range(n))
    assert abs(kept_py / n - 0.25) < 0.02
    assert kept_go == n
    # p = 0 drops everything; a value absent from the table with no "*" keeps.
    assert not any(row_passes({"subsample": {"language": {"Java": 0.0}}},
                              {"language": "Java"}, rng) for _ in range(100))
    assert row_passes({"subsample": {"language": {"Java": 0.0}}},
                      {"language": "Go"}, rng)


def test_no_gate_keys_means_everything_passes():
    assert row_passes({"hf_id": "x", "weight": 1.0}, {}, random.Random(0))


def test_new_format_functions_produce_chat_shaped_text():
    om = FORMAT_FN_PRETRAIN["openmath_instruct2"]
    out = om({"problem": "2+2?", "generated_solution": "4"})
    assert out == "<|user|>2+2?<|assistant|>4<|end_turn|>"
    assert om({"problem": "", "generated_solution": "4"}) == ""
    io = FORMAT_FN_PRETRAIN["io_pair"]
    out = io({"input": "reverse a list", "output": "xs[::-1]"})
    assert out == "<|user|>reverse a list<|assistant|>xs[::-1]<|end_turn|>"
    assert io({"input": "q"}) == ""
    # Non-string fields are skipped (""), never raised (finding 4).
    assert io({"input": 3, "output": "x"}) == ""
    assert om({"problem": "q", "generated_solution": None}) == ""


def test_every_single_turn_formatter_ends_the_assistant_turn():
    """Each chat-shaped formatter must close with <|end_turn|> and open with
    <|user|>, so pretraining sees the same contract as inference."""
    rows = {
        "nemotron_sft": {
            "messages": [{"role": "user", "content": "q"},
                         {"role": "assistant", "content": "a"}],
            "reasoning": "r",
        },
        "openr1_math": {"problem": "p", "generations": ["<think>t</think>s"],
                        "answer": "7", "correctness_math_verify": [True]},
        "openmath_reasoning": {"problem": "p", "expected_answer": "7",
                               "generated_solution": "<think>t</think>s"},
        "openthoughts": {"conversations": [
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "<think>t</think>a"}]},
        "magicoder": {"instruction": "i", "response": "r"},
        "magicoder_oss": {"problem": "p", "solution": "s"},
        "bbh": {"input": "i", "target": "(A)"},
        "openmath_instruct2": {"problem": "p", "generated_solution": "s"},
        "io_pair": {"input": "i", "output": "o"},
    }
    for key, row in rows.items():
        out = FORMAT_FN_PRETRAIN[key](row)
        assert out.startswith("<|user|>"), (key, out)
        assert out.endswith("<|end_turn|>"), (key, out)
        assert "<|assistant|>" in out and "\n<|" not in out, (key, out)
        assert "<think>" not in out and "</think>" not in out, (key, out)
    r1 = FORMAT_FN_PRETRAIN["openr1_math"](rows["openr1_math"])
    assert r1 == ("<|user|>p<|assistant|><|think|>t<|/think|>s"
                  "<|answer|>7<|/answer|><|end_turn|>")
    assert FORMAT_FN_PRETRAIN["bbh"](rows["bbh"]) == \
        "<|user|>i<|assistant|><|answer|>(A)<|/answer|><|end_turn|>"


def test_phase_weights_must_sum_to_one_as_written():
    cfg = PretrainConfig()
    for name, ph in cfg.phases.items():
        assert abs(sum(d["weight"] for d in ph["datasets"]) - 1.0) < 1e-6, name
    bad = PretrainConfig()
    bad.phases["foundation"]["datasets"][0]["weight"] += 0.1
    with pytest.raises(ValueError, match="weights sum"):
        bad._resolve_phases()
    nokey = PretrainConfig()
    del nokey.phases["anneal"]["datasets"][0]["weight"]
    with pytest.raises(ValueError, match="hf_id and weight"):
        nokey._resolve_phases()


def test_every_phase_format_key_is_registered_and_gates_are_well_formed():
    cfg = PretrainConfig()
    for name, ph in cfg.phases.items():
        for d in ph["datasets"]:
            fmt = d.get("format")
            assert fmt is None or fmt in FORMAT_FN_PRETRAIN, (name, d["name"], fmt)
            for key in ("filter", "subsample"):
                assert isinstance(d.get(key, {}), dict), (name, d["name"], key)
            assert d.get("max_tokens", 1) > 0



def _repo(*files):
    return {"repo_path": "u/r", "files": [
        {"content_id": cid, "content": body, "file_path": path, "language": lang,
         "is_vendor": vendor, "size_bytes": str(len(body))}
        for cid, path, lang, body, vendor in files]}


def test_stack_v3_formatter_keeps_wanted_languages_and_skips_vendor():
    from osrt.data import _format_stack_v3, _format_stack_v3_nonpython
    row = _repo(
        ("00000000", "main.rs", "Rust", "fn main() {}", "False"),
        ("00000001", "lib/vendor.js", "JavaScript", "var x", "True"),      # vendor
        ("00000002", "a.py", "Python", "print(1)", "False"),
        ("00000003", "data.csv", "CSV", "a,b,c", "False"),                  # p = 0
        ("00000004", "low.cob", "COBOL", "DISPLAY-LOW", "False"),     # "*": u~0 kept
        ("ffffffff", "high.cob", "COBOL", "DISPLAY-HIGH", "False"),   # "*": u~1 dropped
    )
    out = _format_stack_v3(row)
    assert "# main.rs\nfn main() {}" in out and "print(1)" in out
    assert "var x" not in out and "a,b,c" not in out
    assert "DISPLAY-LOW" in out and "DISPLAY-HIGH" not in out
    assert "print(1)" not in _format_stack_v3_nonpython(row)
    assert "fn main" in _format_stack_v3_nonpython(row)


def test_stack_v3_language_table_has_a_default_and_explicit_junk_zeros():
    from osrt.data import STACK_V3_LANG_ACCEPT, _stack_v3_keep
    assert STACK_V3_LANG_ACCEPT["*"] == 0.25
    for junk in ("CSV", "TSV", "SVG", "XML", "Text"):
        assert STACK_V3_LANG_ACCEPT[junk] == 0.0, junk
    assert STACK_V3_LANG_ACCEPT["JSON"] == 0.1
    # An unlisted language is subject to the default, not silently p = 0 ...
    assert _stack_v3_keep({"content_id": "00000001", "language": "Zig",
                           "is_vendor": "False"}, STACK_V3_LANG_ACCEPT)
    # ... while a missing language is never kept.
    assert not _stack_v3_keep({"content_id": "00000001", "is_vendor": "False"},
                              STACK_V3_LANG_ACCEPT)


def test_stack_v3_acceptance_is_deterministic_by_content_id():
    from osrt.data import _stack_v3_keep
    accept = {"C#": 0.15}
    lo = {"content_id": "00000001", "language": "C#", "is_vendor": "False"}  # u ~ 0
    hi = {"content_id": "ffffffff", "language": "C#", "is_vendor": "False"}  # u ~ 1
    assert _stack_v3_keep(lo, accept) and not _stack_v3_keep(hi, accept)
    # The decision is a threshold on the content-id hash: 0x26666666 is just
    # below 0.15 * 0xFFFFFFFF, 0x26666667 just above.
    below = {"content_id": "26666666ab", "language": "C#", "is_vendor": "False"}
    above = {"content_id": "26666667ab", "language": "C#", "is_vendor": "False"}
    assert _stack_v3_keep(below, accept) and not _stack_v3_keep(above, accept)
    # A non-hex id falls back to u = 0.5: kept at p > 0.5, dropped at p < 0.5.
    odd = {"content_id": "not-hex!", "language": "C#", "is_vendor": "False"}
    assert _stack_v3_keep(odd, {"C#": 0.6}) and not _stack_v3_keep(odd, {"C#": 0.4})
    # Vendor files are dropped regardless of language probability.
    assert not _stack_v3_keep({**lo, "is_vendor": "True"}, {"C#": 1.0})
