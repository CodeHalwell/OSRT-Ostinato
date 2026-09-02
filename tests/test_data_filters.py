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
