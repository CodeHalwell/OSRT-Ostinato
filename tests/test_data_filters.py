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
    assert out == "<|user|>2+2?<|assistant|>4"
    assert om({"problem": "", "generated_solution": "4"}) == ""
    io = FORMAT_FN_PRETRAIN["io_pair"]
    assert io({"input": "reverse a list", "output": "xs[::-1]"}).startswith("<|user|>")
    assert io({"input": "q"}) == ""


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
        ("00000003", "weird.cob", "COBOL", "DISPLAY", "False"),             # p = 0
    )
    out = _format_stack_v3(row)
    assert "# main.rs\nfn main() {}" in out and "print(1)" in out
    assert "var x" not in out and "DISPLAY" not in out
    assert "print(1)" not in _format_stack_v3_nonpython(row)
    assert "fn main" in _format_stack_v3_nonpython(row)


def test_stack_v3_acceptance_is_deterministic_by_content_id():
    from osrt.data import _stack_v3_keep
    accept = {"C#": 0.15}
    lo = {"content_id": "00000001", "language": "C#", "is_vendor": "False"}  # u ~ 0
    hi = {"content_id": "ffffffff", "language": "C#", "is_vendor": "False"}  # u ~ 1
    assert _stack_v3_keep(lo, accept) and not _stack_v3_keep(hi, accept)
    assert _stack_v3_keep(lo, accept) == _stack_v3_keep(lo, accept)
