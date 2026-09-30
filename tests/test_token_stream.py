"""`TokenStream.__iter__` end to end on fake multi-shard datasets.

`datasets.load_dataset` is monkeypatched to serve in-memory
`Dataset.from_list(...).to_iterable_dataset(num_shards=N)` streams (or a
poison object that fails on read), `time.sleep` is a no-op, and the real
tokenizer under `tokenizer/` is used so marker ids are the contract's.
"""
from __future__ import annotations

import io
import time

import datasets
import pytest
import torch
from datasets import Dataset, IterableDataset

import osrt.data as data
from osrt.chat_format import encode_with_markers
from osrt.data import (
    DataSourceDead,
    TokenStream,
    make_loader,
    process_row,
    row_to_tokens,
)

TOK = "tokenizer"
EOS = 49153
U, A, S, E = 49163, 49164, 49165, 49166


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(data, "SHUFFLE_BUFFER_SIZE", 8)


@pytest.fixture(scope="module")
def tok():
    return data._load_tokenizer(TOK)


class _Poison:
    """Opens fine, fails on every read (revoked gate, broken shard, outage)."""

    def skip(self, n):
        return self

    def shuffle(self, **kw):
        return self

    def state_dict(self):
        return None

    def __iter__(self):
        raise ConnectionError("simulated HF outage")


def _install(monkeypatch, tables: dict, num_shards: int = 4) -> list[str]:
    """Serve `tables[hf_id]` (a list of rows, or a ready object) as the stream."""
    calls: list[str] = []

    def _load(hf_id, **kw):
        calls.append(hf_id)
        src = tables[hf_id]
        if isinstance(src, list):
            return Dataset.from_list(src).to_iterable_dataset(num_shards=num_shards)
        return src

    monkeypatch.setattr(datasets, "load_dataset", _load)
    return calls


def _take(stream, n):
    it = iter(stream)
    return [next(it) for _ in range(n)]


def _flat(chunks):
    return [t for x, _ in chunks for t in x.tolist()]


def _docs(tokens):
    """Complete documents (token lists between EOS); a trailing partial is dropped."""
    out, cur = [], []
    for t in tokens:
        if t == EOS:
            out.append(cur)
            cur = []
        else:
            cur.append(t)
    return out


def _text_rows(prefix, n, shards=4):
    return [{"text": f"{prefix} document {i} " + "word " * (i % 7)} for i in range(n)]


# ── packing ──────────────────────────────────────────────────────────────

def test_packing_is_lossless_with_one_eos_per_document(monkeypatch, tok):
    rows = _text_rows("alpha", 80)
    _install(monkeypatch, {"A": rows})
    stream = TokenStream([{"name": "A", "hf_id": "A", "weight": 1.0}], 16, TOK, seed=3)
    chunks = _take(stream, 30)
    for x, y in chunks:
        assert x.shape == (16,) and x.dtype == torch.long
        assert torch.equal(x, y), "labels must equal input_ids (the model shifts)"
    flat = _flat(chunks)
    expected = {tuple(encode_with_markers(tok, r["text"])) for r in rows}
    docs = [tuple(d) for d in _docs(flat)]
    assert len(docs) >= 20
    assert all(d in expected for d in docs), "a chunk boundary corrupted a document"
    assert len(set(docs)) == len(docs), "a document was packed twice"
    assert flat.count(EOS) == len(docs) + (1 if flat[-1] == EOS else 0)
    assert all(not (p == EOS and q == EOS) for p, q in zip(flat, flat[1:]))


# ── the chat contract reaches the token stream ───────────────────────────

def test_chat_rows_carry_the_contract_ids_through_the_iterator(monkeypatch, tok):
    tables = {
        "io": [{"input": f"q{i}", "output": f"a{i}"} for i in range(30)],
        "msgs": [{"messages": [{"role": "system", "content": "sys"},
                               {"role": "user", "content": f"q{i}"},
                               {"role": "assistant", "content": f"a{i}"}]}
                 for i in range(30)],
        "alpaca": [{"instruction": f"do {i}", "input": "ctx" if i % 2 else "",
                    "output": f"done {i}"} for i in range(30)],
        "share": [{"conversations": [{"from": "human", "value": f"q{i}"},
                                     {"from": "gpt", "value": f"a{i}"}]}
                  for i in range(30)],
    }
    _install(monkeypatch, tables)
    cfgs = [
        {"name": "io", "hf_id": "io", "weight": 0.25, "format": "io_pair"},
        {"name": "msgs", "hf_id": "msgs", "weight": 0.25},
        {"name": "alpaca", "hf_id": "alpaca", "weight": 0.25},
        {"name": "share", "hf_id": "share", "weight": 0.25},
    ]
    flat = _flat(_take(TokenStream(cfgs, 32, TOK, seed=0), 40))
    assert {U, A, S, E} <= set(flat)
    texts = [tok.decode(d) for d in _docs(flat)]
    assert len(texts) > 40
    for t in texts:
        assert t.startswith(("<|user|>", "<|system|>")), t
        assert t.endswith("<|end_turn|>"), t
        assert "\n<|" not in t and "role:" not in t
    assert any(t == "<|system|>sys<|user|>q1<|assistant|>a1<|end_turn|>" for t in texts)
    assert any(t == "<|user|>do 1\nctx<|assistant|>done 1<|end_turn|>" for t in texts)
    assert any(t == "<|user|>do 2<|assistant|>done 2<|end_turn|>" for t in texts)


# ── dead-source policy ───────────────────────────────────────────────────

def test_dead_stream_is_dropped_and_the_other_continues(monkeypatch, tok, capsys):
    calls = _install(monkeypatch, {"A": _text_rows("alpha", 60), "B": _Poison()})
    cfgs = [{"name": "A", "hf_id": "A", "weight": 0.5},
            {"name": "B", "hf_id": "B", "weight": 0.5}]
    stream = TokenStream(cfgs, 16, TOK, seed=0)
    assert stream.dead_sources == []
    chunks = _take(stream, 20)
    assert stream.dead_sources == ["B"]
    texts = [tok.decode(d) for d in _docs(_flat(chunks))]
    assert texts and all(t.startswith("alpha document") for t in texts)
    out = capsys.readouterr().out
    assert "SOURCE DEAD: B" in out and "live weights now: A=100.0%" in out
    assert calls.count("B") <= 1 + 3 * 8       # bounded reconnects, then dead
    sd = stream.state_dict()
    assert sd["dead"] == ["B"] and sd["tokens_seen"][1] == 0


def test_all_sources_dead_raises_within_bounded_picks(monkeypatch):
    calls = _install(monkeypatch, {"A": _Poison(), "B": _Poison()})
    cfgs = [{"name": "A", "hf_id": "A", "weight": 0.5},
            {"name": "B", "hf_id": "B", "weight": 0.5}]
    stream = TokenStream(cfgs, 16, TOK, seed=0, max_stream_failures=2)
    with pytest.raises(DataSourceDead) as info:
        next(iter(stream))
    assert sorted(info.value.sources) == ["A", "B"]
    assert isinstance(info.value, RuntimeError)
    assert sorted(stream.dead_sources) == ["A", "B"]
    assert len(calls) <= 2 * (1 + 2 * 8)


def test_source_rejecting_every_row_dies_after_the_threshold(monkeypatch, capsys):
    _install(monkeypatch, {
        "A": _text_rows("alpha", 60),
        "B": [{"text": "   "} for _ in range(12)],      # every row renders empty
    })
    cfgs = [{"name": "A", "hf_id": "A", "weight": 0.5},
            {"name": "B", "hf_id": "B", "weight": 0.5}]
    stream = TokenStream(cfgs, 16, TOK, seed=0, max_consecutive_rejections=50)
    _take(stream, 10)
    assert stream.dead_sources == ["B"]
    sd = stream.state_dict()
    assert sd["skipped"][1] == {"empty": 50}
    assert sd["streams"][1]["cycles"] >= 3           # it re-epoched while rejecting
    out = capsys.readouterr().out
    assert "SOURCE DEAD: B" in out and "50 consecutive rows rejected" in out
    assert "B: epoch 1 complete" in out


def test_rejection_streak_resets_on_an_accepted_row(monkeypatch):
    """A source that is merely heavily filtered must NOT be declared dead."""
    rows = [{"text": f"kept {i} " * 3 if i % 5 == 0 else "", "i": i}
            for i in range(100)]
    _install(monkeypatch, {"B": rows})
    stream = TokenStream([{"name": "B", "hf_id": "B", "weight": 1.0}], 16, TOK,
                         seed=1, max_consecutive_rejections=40)
    _take(stream, 15)
    assert stream.dead_sources == []
    sd = stream.state_dict()
    assert sd["skipped"][0]["empty"] > 40 and sd["rows_ok"][0] > 0


# ── malformed rows ───────────────────────────────────────────────────────

def test_non_string_fields_are_skipped_not_raised(monkeypatch, capsys):
    def bad_rows():
        yield {"messages": [{"role": "user", "content": 123},
                            {"role": "assistant", "content": "x"}]}
        yield {"messages": "not a list"}
        yield {"messages": [{"role": "user", "content": "q"}, "junk"]}
        yield {"conversations": [{"from": "human", "value": None},
                                 {"from": "gpt", "value": "a"}]}
        yield {"instruction": None, "output": "x"}
        yield {"instruction": "i", "input": ["x"], "output": "o"}      # input ignored
        yield {"text": ["a", "b"]}
        yield {"text": None}
        yield {"content": 5}
        yield {"input": 1, "output": 2}
        yield {"problem": "q", "generated_solution": {"nested": 1}}
        for i in range(30):
            yield {"text": f"fine document {i} " + "w " * i}

    def boom(example, state=None):
        raise KeyError("formatter bug")

    monkeypatch.setitem(data.FORMAT_FN_PRETRAIN, "boom", boom)
    _install(monkeypatch, {
        "mixed": IterableDataset.from_generator(bad_rows),
        "io": [{"input": 1, "output": 2}] * 5 + [{"input": 7, "output": 8}] * 5,
        "crash": [{"text": f"t{i}"} for i in range(10)],
    })
    cfgs = [
        {"name": "mixed", "hf_id": "mixed", "weight": 0.6},
        {"name": "io", "hf_id": "io", "weight": 0.2, "format": "io_pair"},
        {"name": "crash", "hf_id": "crash", "weight": 0.2, "format": "boom"},
    ]
    stream = TokenStream(cfgs, 16, TOK, seed=0, max_consecutive_rejections=25)
    chunks = _take(stream, 40)                       # > one epoch of "mixed"
    assert len(chunks) == 40
    sd = stream.state_dict()
    assert sd["skipped"][0]["empty"] >= 10           # the malformed rows, counted
    assert sd["skipped"][2] == {"error": 25}         # the crashing formatter
    assert sorted(stream.dead_sources) == ["crash", "io"]
    out = capsys.readouterr().out
    assert out.count("crash: row error KeyError") == 1   # logged once per source
    assert "row keys: ['text']" in out


def test_process_row_reports_path_reason_and_text(monkeypatch, tok):
    import random
    rng = random.Random(0)
    r = process_row({"format": "io_pair"}, {"input": "q", "output": "a"}, tok, rng)
    assert (r.reason, r.path) == ("ok", "format:io_pair")
    assert r.text == "<|user|>q<|assistant|>a<|end_turn|>" and r.tokens[-1] == E
    r = process_row({}, {"messages": [{"role": "user", "content": "q"},
                                      {"role": "assistant", "content": "a"}]}, tok, rng)
    assert (r.reason, r.path) == ("ok", "extract:messages")
    assert process_row({}, {"text": "hello"}, tok, rng).path == "extract:text"
    assert process_row({}, {"text": ""}, tok, rng).reason == "empty"
    r = process_row({"filter": {"k": 1}}, {"k": 2, "text": "x"}, tok, rng)
    assert r.reason == "filter"
    r = process_row({"max_tokens": 2}, {"text": "one two three four"}, tok, rng)
    assert r.reason == "max_tokens" and r.tokens is None and r.text
    assert process_row({}, "not a dict", tok, rng).reason == "error"
    assert row_to_tokens({}, {"text": "hi"}, tok, rng) == \
        tok.encode("hi", add_special_tokens=False)
    assert row_to_tokens({}, {"text": 5}, tok, rng) is None
    with pytest.raises(ValueError, match="Unknown pretrain format key"):
        row_to_tokens({"format": "nope"}, {"text": "x"}, tok, rng)
    with pytest.raises(ValueError, match="Unknown pretrain format key"):
        TokenStream([{"name": "x", "hf_id": "x", "weight": 1.0, "format": "nope"}],
                    16, TOK, seed=0)


# ── stack v3 ─────────────────────────────────────────────────────────────

def _repo(*files):
    lang = {"py": "Python", "csv": "CSV"}
    return {"repo_path": "u/r", "files": [
        {"content_id": cid, "content": body, "file_path": path,
         "language": lang[path.rsplit(".", 1)[-1]],
         "is_vendor": "False", "size_bytes": str(len(body))}
        for cid, path, body in files]}


def test_stack_v3_cap_is_checked_per_file_and_tally_is_per_instance():
    from osrt.data import STACK_V3_MAX_CHARS, _format_stack_v3
    huge = "x" * (STACK_V3_MAX_CHARS + 10)
    out = _format_stack_v3(_repo(("00000000", "huge.py", huge),
                                 ("00000001", "a.py", "print(1)"),
                                 ("00000002", "b.py", "print(2)")))
    assert "huge.py" not in out and "print(1)" in out and "print(2)" in out
    half = "y" * (STACK_V3_MAX_CHARS // 2 - 100)
    out = _format_stack_v3(_repo(("00000000", "one.py", half),
                                 ("00000001", "two.py", half),
                                 ("00000002", "three.py", half),
                                 ("00000003", "small.py", "print(3)")))
    assert "one.py" in out and "two.py" in out
    assert "three.py" not in out                     # would exceed the cap: skipped
    assert "print(3)" in out                         # ... the rest still fits
    assert len(out) <= STACK_V3_MAX_CHARS
    s1, s2 = {}, {}
    row = _repo(("00000000", "a.py", "print(1)"), ("00000001", "b.py", "print(2)"))
    _format_stack_v3(row, s1)
    _format_stack_v3(row, s1)
    _format_stack_v3(row, s2)
    assert s1["stack_v3_tally"] == {"Python": 4}
    assert s2["stack_v3_tally"] == {"Python": 2}
    assert not hasattr(data, "_stack_v3_tally")


def test_stack_v3_rows_stream_through_the_iterator(monkeypatch, tok):
    repos = [_repo((f"{i:08x}", f"f{i}.py", f"print({i})\n" * 5),
                   (f"{i + 1000:08x}", "notes.csv", "a,b,c")) for i in range(20)]
    _install(monkeypatch, {"S": repos})
    cfg = {"name": "S", "hf_id": "S", "weight": 1.0, "format": "stack_v3"}
    stream = TokenStream([cfg], 32, TOK, seed=0)
    texts = [tok.decode(d) for d in _docs(_flat(_take(stream, 8)))]
    assert texts and all(t.startswith("# f") and "a,b,c" not in t for t in texts)


# ── resume ───────────────────────────────────────────────────────────────

def _doc_ids(tok, chunks, prefix):
    ids = []
    for d in _docs(_flat(chunks)):
        t = tok.decode(d)
        if t.startswith(prefix):
            ids.append(int(t.split()[2]))
    return ids


def test_state_dict_is_none_before_iteration_and_a_plain_dict_after(monkeypatch, tok):
    _install(monkeypatch, {"A": _text_rows("alpha", 40)})
    stream = TokenStream([{"name": "A", "hf_id": "A", "weight": 1.0}], 16, TOK, seed=0)
    assert stream.state_dict() is None
    _take(stream, 5)
    sd = stream.state_dict()
    assert sd["version"] == 1 and sd["sources"] == ["A"] and sd["dead"] == []
    assert sd["streams"][0]["ds_state"] is not None
    assert sd["streams"][0]["shuffle_seed"] == 0
    assert isinstance(sd["rng"], list) and isinstance(sd["rng"][1], list)
    assert 0 < len(sd["buffer"]) < 16 + 200 and sd["rows_ok"] == [sd["rows_ok"][0]]
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    assert torch.load(buf, weights_only=True) == sd


def test_resume_continues_shard_progress_instead_of_row_zero(monkeypatch, tok, capsys):
    rows_a = _text_rows("alpha", 200)
    rows_b = [{"input": f"q{i}", "output": f"a{i}"} for i in range(40)]
    _install(monkeypatch, {"A": rows_a, "B": rows_b})
    cfgs = [{"name": "A", "hf_id": "A", "weight": 0.7},
            {"name": "B", "hf_id": "B", "weight": 0.3, "format": "io_pair"}]
    first = TokenStream(cfgs, 16, TOK, seed=1)
    half1 = _take(first, 40)
    sd = first.state_dict()
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    saved = torch.load(buf, weights_only=True)

    resumed = TokenStream(cfgs, 16, TOK, seed=999, resume_state=saved)
    half2 = _take(resumed, 40)
    out = capsys.readouterr().out
    assert "resumed data position — restored: A, B" in out

    a1, a2 = _doc_ids(tok, half1, "alpha"), _doc_ids(tok, half2, "alpha")
    assert len(a1) > 25 and len(a2) > 25
    # Not a restart: a fresh stream with the ORIGINAL seed replays half1's
    # opening docs; the resumed stream must not.
    fresh = _doc_ids(tok, _take(TokenStream(cfgs, 16, TOK, seed=1), 40), "alpha")
    assert fresh[:10] == a1[:10]
    assert a2[:10] != a1[:10]
    # Little overlap: only what the lost shuffle buffer can account for.
    assert len(set(a1) & set(a2)) <= data.SHUFFLE_BUFFER_SIZE + 2
    # The sampler state came along: token shares stay on target across the seam.
    sd2 = resumed.state_dict()
    total = sum(sd2["tokens_seen"])
    assert sd2["tokens_seen"][0] > sd["tokens_seen"][0]
    assert abs(sd2["tokens_seen"][0] / total - 0.7) < 0.05
    assert sd2["rows_ok"][0] > sd["rows_ok"][0] and sd2["streams"][0]["cycles"] == 0


def test_make_loader_passes_resume_state_and_exposes_state_dict(monkeypatch):
    _install(monkeypatch, {"A": _text_rows("alpha", 60)})
    cfgs = [{"name": "A", "hf_id": "A", "weight": 1.0}]
    loader = make_loader(cfgs, 16, TOK, batch_size=2, step_num=5, num_workers=0)
    assert loader.dataset.state_dict() is None
    x, y = next(iter(loader))
    assert x.shape == (2, 16) and torch.equal(x, y)
    sd = loader.dataset.state_dict()
    assert sd is not None and sd["sources"] == ["A"]
    loader2 = make_loader(cfgs, 16, TOK, batch_size=2, step_num=7, num_workers=0,
                          resume_state=sd)
    next(iter(loader2))
    assert loader2.dataset.state_dict()["rows_ok"][0] > sd["rows_ok"][0]
    assert loader2.dataset.dead_sources == []


def test_resume_state_for_other_sources_is_ignored_with_a_warning(monkeypatch, capsys):
    _install(monkeypatch, {"A": _text_rows("alpha", 40)})
    bogus = {"version": 1, "sources": ["X"], "streams": [], "buffer": [1, 2, 3]}
    stream = TokenStream([{"name": "A", "hf_id": "A", "weight": 1.0}], 16, TOK, seed=0,
                         resume_state=bogus)
    _take(stream, 2)
    assert "starting every source fresh" in capsys.readouterr().out
    assert stream.state_dict()["sources"] == ["A"]


def test_state_dict_is_none_without_datasets_support(monkeypatch, capsys):
    _install(monkeypatch, {"A": _text_rows("alpha", 40)})
    monkeypatch.delattr(datasets.IterableDataset, "state_dict")
    monkeypatch.setattr(data, "_STATE_DICT_UNSUPPORTED_WARNED", False)
    stream = TokenStream([{"name": "A", "hf_id": "A", "weight": 1.0}], 16, TOK, seed=0)
    _take(stream, 3)
    assert stream.state_dict() is None
    assert stream.state_dict() is None
    assert capsys.readouterr().out.count("no IterableDataset.state_dict") == 1


# ── the mix ──────────────────────────────────────────────────────────────

def test_mix_shares_hold_with_a_heavily_filtered_source(monkeypatch):
    rows_a = [{"text": f"alpha document {i} " + "word " * (5 + i % 9)}
              for i in range(150)]
    rows_b = [{"text": f"beta document {i} " + "token " * (3 + i % 5),
               "keep": int(i % 5 == 0)} for i in range(100)]
    _install(monkeypatch, {"A": rows_a, "B": rows_b})
    cfgs = [{"name": "A", "hf_id": "A", "weight": 0.7},
            {"name": "B", "hf_id": "B", "weight": 0.3, "filter": {"keep": 1}}]
    stream = TokenStream(cfgs, 32, TOK, seed=2)
    _take(stream, 150)
    sd = stream.state_dict()
    total = sum(sd["tokens_seen"])
    assert abs(sd["tokens_seen"][0] / total - 0.7) < 0.03
    assert abs(sd["tokens_seen"][1] / total - 0.3) < 0.03
    assert sd["skipped"][1]["filter"] > sd["rows_ok"][1]      # filtered, but alive
    assert stream.dead_sources == []
    assert sd["mix_counts"]["B"]["keep"] == {"1": sd["rows_ok"][1]}


def test_cycle_logging_is_throttled(monkeypatch, capsys):
    _install(monkeypatch, {"T": [{"text": f"t{i}"} for i in range(3)]}, num_shards=1)
    stream = TokenStream([{"name": "T", "hf_id": "T", "weight": 1.0}], 8, TOK, seed=0)
    _take(stream, 130)
    sd = stream.state_dict()
    assert sd["streams"][0]["cycles"] > 60
    out = capsys.readouterr().out
    cycles = sd["streams"][0]["cycles"]
    logged = sorted(int(line.split("epoch ")[1].split()[0])
                    for line in out.splitlines() if "T: epoch " in line)
    want = [c for c in (1, 2, 3, 5, 10, 20, 50, 100) if c <= cycles]
    want += list(range(200, cycles + 1, 100))
    assert logged == want
    assert f"epochs={logged[-1]}" in out
