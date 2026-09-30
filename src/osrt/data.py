"""Streaming data pipeline for OSRT pre-training.

Handles:
- Progressive seq_len (2048 → 4096 → 8192) across phases
- Multi-dataset weighted sampling within each phase (token-debt sampler)
- Code + text mixing from the start
- ONE chat contract for every chat-shaped row: `messages`, `conversations`,
  `instruction/output` and every `format=` formatter go through
  `osrt.chat_format.render_chat`, so `<|system|>`, `<|user|>`,
  `<|assistant|>` and `<|end_turn|>` appear in pretraining exactly as they
  will at inference.
- Marker-aware tokenisation (`encode_with_markers`): OSRT markers become
  their single ids; SmolLM2's legacy control strings in raw web text stay text.
- Resilient streaming: connection drops and corrupt shards are caught and
  retried. A source that keeps failing, or keeps rejecting every row, is
  declared DEAD, announced loudly and dropped from the sampler; when no
  source is left `DataSourceDead` is raised instead of spinning forever.
- Resumable: `TokenStream.state_dict()` captures every source's position
  (the `datasets` iterable state of the current shuffled epoch), the token
  buffer and the sampler state; `resume_state=` restores it.
- Optional `format` key on a dataset config to force a custom text
  extractor (Stack v3 repo rows, reasoning traces, io pairs, ...).
"""

from __future__ import annotations

import copy
import functools
import random
import time
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

from osrt.chat_format import encode_with_markers, render_chat

# ── Custom text extractors (the `format=` key) ─────────────────────────
# Used only when a dataset config sets `format=<key>`. Default (no format
# key) falls through to `_extract_text`, which handles plain text / code /
# chat-shaped columns generically.
#
# Every extractor has the signature `fn(example, state=None) -> str`, where
# `state` is a per-(stream, source) dict a formatter may use for its own
# bookkeeping (Stack v3 keeps its realised-language tally there). The
# returned string is already-formatted text ready for `encode_with_markers`,
# so the OSRT markers a formatter embeds end up as their single ids. Return
# "" for a malformed row and the stream skips it.


def _chat(user: str, assistant: str) -> str:
    """One user/assistant exchange in the canonical form (ends `<|end_turn|>`)."""
    return render_chat([
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ])


def _format_nemotron_sft_text(example: dict, state: dict | None = None) -> str:
    """Wrap Nemotron-Post-Training rows in the SFT chat schema.

    Used as a *rehearsal* signal during continued pretraining so the
    model keeps seeing (and predicting) the chat tags it learned in
    SFT — without that, ~1,800 steps of plain-text pretrain would
    erode the structured think/answer behaviour. Pretrain's loss is
    full-token (no masking), so the model is trained to predict every
    token in the formatted string including the chat tags.

    Pulls (question, reasoning, answer) from messages + the dedicated
    `reasoning` field, then wraps as
        <|user|>{q}<|assistant|><|think|>{r}<|/think|><|answer|>{a}<|/answer|><|end_turn|>
    Returns "" on malformed rows so the stream loop skips them.
    """
    msgs = example.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < 2:
        return ""
    question = ""
    answer = ""
    for m in msgs:
        if not isinstance(m, dict):
            return ""
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if not isinstance(content, str):
            return ""
        if role == "user" and not question:
            question = content
        elif role == "assistant" and question and not answer:
            answer = content
            break
    if not question.strip() or not answer.strip():
        return ""
    reasoning = example.get("reasoning") or ""
    if not isinstance(reasoning, str):
        return ""
    return _chat(
        question,
        f"<|think|>{reasoning.strip()}<|/think|>"
        f"<|answer|>{answer.strip()}<|/answer|>",
    )


def _format_stack_code(example: dict, state: dict | None = None) -> str:
    """The Stack v2 / the-stack-smol code rows.

    Schema varies across The Stack subsets — try `content`, `text`,
    and `code` in that order. Returns "" if none present so malformed
    rows skip cleanly rather than killing the worker.
    """
    for key in ("content", "text", "code"):
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _format_arxiv(example: dict, state: dict | None = None) -> str:
    """RedPajama-arxiv rows.

    RedPajama stores the full LaTeX paper body in `text`. Long but
    well-formatted; the tokenizer handles LaTeX as a sequence of
    short BPE pieces.
    """
    text = example.get("text", "")
    if isinstance(text, str) and text.strip():
        return text
    return ""


def _format_nemotron_math_pretrain(example: dict, state: dict | None = None) -> str:
    """Nemotron-CC-Math-v1 rows.

    The math-pretraining variant ships with `text` containing the
    math-rich web document, LaTeX equations preserved. Same shape as
    arxiv rows but tagged separately so the data-mix logging can
    distinguish them.
    """
    text = example.get("text", "")
    if isinstance(text, str) and text.strip():
        return text
    return ""


# ── Cold-start reasoning trace extractors (DeepSeek-R1 style) ───────────
# Per the R1 paper, exposure to long-form <think> traces during
# continued pretraining is the most efficient way to teach a small
# model to "think before answering".
#
# R1-style datasets natively emit `<think>...</think>` (HTML-style)
# while our model's special tokens are `<|think|>...<|/think|>`
# (pipe-delimited). We rewrite the tags during text extraction so the
# model trains on its own format consistently — otherwise the inner
# `<think>` would tokenise as raw BPE pieces and the model would learn
# a parallel, non-canonical reasoning format.


_R1_TAG_REWRITES = (
    ("<think>",  "<|think|>"),
    ("</think>", "<|/think|>"),
    # OpenThoughts uses these alternative tags for the same purpose.
    ("<|begin_of_thought|>", "<|think|>"),
    ("<|end_of_thought|>",   "<|/think|>"),
    ("<|begin_of_solution|>", "<|answer|>"),
    ("<|end_of_solution|>",   "<|/answer|>"),
)


def _rewrite_reasoning_tags(text: str) -> str:
    """Map R1/OpenThoughts inner tags to our canonical special tokens."""
    for src, dst in _R1_TAG_REWRITES:
        if src in text:
            text = text.replace(src, dst)
    return text


def _format_openr1_math(example: dict, state: dict | None = None) -> str:
    """open-r1/OpenR1-Math-220k rows — DeepSeek-R1 reasoning traces.

    Schema: `problem` + `generations` (list of R1 outputs, each with
    `<think>...</think>` wrappers) + `answer` + `correctness_math_verify`
    (list of bools). Picks the first verified-correct generation;
    skips the row entirely if none are correct, to avoid training the
    model on R1's mistakes.
    """
    problem = example.get("problem")
    generations = example.get("generations")
    answer = example.get("answer")
    verify = example.get("correctness_math_verify")
    if not (isinstance(problem, str) and problem.strip()):
        return ""
    if not (isinstance(generations, list) and generations):
        return ""
    pick = None
    if isinstance(verify, list) and len(verify) == len(generations):
        for i, ok in enumerate(verify):
            if ok and isinstance(generations[i], str) and generations[i].strip():
                pick = generations[i]
                break
    if pick is None:
        return ""
    pick = _rewrite_reasoning_tags(pick)
    answer_str = str(answer or "").strip()
    return _chat(problem, f"{pick}<|answer|>{answer_str}<|/answer|>")


def _format_openmath_reasoning(example: dict, state: dict | None = None) -> str:
    """nvidia/OpenMathReasoning/cot rows — DeepSeek-R1 math traces.

    Schema: `problem` + `generated_solution` (already wraps `<think>`)
    + `expected_answer`. Skips rows where the solution is n/a (a few
    edge-case rows in the cot split lack a generation).
    """
    problem = example.get("problem")
    solution = example.get("generated_solution")
    answer = example.get("expected_answer")
    if not (isinstance(problem, str) and problem.strip()):
        return ""
    if not (isinstance(solution, str) and solution.strip()) or solution == "n/a":
        return ""
    solution = _rewrite_reasoning_tags(solution)
    answer_str = str(answer or "").strip()
    return _chat(problem, f"{solution}<|answer|>{answer_str}<|/answer|>")


def _format_openthoughts(example: dict, state: dict | None = None) -> str:
    """open-thoughts/OpenThoughts-114k rows — multi-domain R1 traces.

    Schema: `conversations` (list of {from, value} dicts, alternating
    user/assistant). Takes the first user/assistant pair (one Q+A per
    row in practice) and rewrites the inner thought tags.
    """
    convs = example.get("conversations")
    if not (isinstance(convs, list) and len(convs) >= 2):
        return ""
    convs = [m for m in convs if isinstance(m, dict)]
    # OpenThoughts ships ShareGPT-format conversations, which use
    # "human"/"gpt" for the speaker; accept the "user"/"assistant" variant
    # too so neither tag convention silently drops every row.
    user_msg = next((m for m in convs if m.get("from") in ("user", "human")), None)
    assistant_msg = next(
        (m for m in convs if m.get("from") in ("assistant", "gpt")), None
    )
    if not user_msg or not assistant_msg:
        return ""
    q = user_msg.get("value", "")
    a = assistant_msg.get("value", "")
    if not (isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip()):
        return ""
    return _chat(q, _rewrite_reasoning_tags(a))


def _format_magicoder(example: dict, state: dict | None = None) -> str:
    """ise-uiuc/Magicoder-Evol-Instruct-110K rows — evolved coding tasks.

    Schema: `instruction` + `response`. Wrap as chat so the model
    sees the instruction-following pattern in pretraining context.
    """
    instr = example.get("instruction")
    resp = example.get("response")
    if not (isinstance(instr, str) and isinstance(resp, str)):
        return ""
    return _chat(instr, resp)


def _format_magicoder_oss(example: dict, state: dict | None = None) -> str:
    """ise-uiuc/Magicoder-OSS-Instruct-75K rows — multi-language OSS code.

    Schema: `problem` + `solution` (plus `lang`, `seed` we ignore).
    """
    prob = example.get("problem")
    sol = example.get("solution")
    if not (isinstance(prob, str) and isinstance(sol, str)):
        return ""
    return _chat(prob, sol)


def _format_bbh(example: dict, state: dict | None = None) -> str:
    """lukaemon/bbh rows — BIG-Bench Hard reasoning tasks.

    Schema: `input` (the puzzle) + `target` (short answer like '(A)').
    No `<think>` block — BBH targets are direct answers, so we keep
    the chat structure minimal. The R1/OpenMath streams cover the
    long-form CoT pattern; BBH covers the answer-precision pattern.
    """
    inp = example.get("input")
    tgt = example.get("target")
    if not (isinstance(inp, str) and inp.strip()):
        return ""
    if not (isinstance(tgt, str) and tgt.strip()):
        return ""
    return _chat(inp, f"<|answer|>{tgt.strip()}<|/answer|>")


def _format_openmath_instruct2(example: dict, state: dict | None = None) -> str:
    """nvidia/OpenMathInstruct-2 rows — problem / generated_solution (short CoT)."""
    q = example.get("problem")
    a = example.get("generated_solution")
    if not (isinstance(q, str) and isinstance(a, str)):
        return ""
    return _chat(q, a)


def _format_io_pair(example: dict, state: dict | None = None) -> str:
    """Rows with `input` / `output` columns (nvidia/OpenCodeInstruct)."""
    q = example.get("input")
    a = example.get("output")
    if not (isinstance(q, str) and isinstance(a, str)):
        return ""
    return _chat(q, a)


# Stack v3 (HuggingFaceCode/stack-v3-train) rows are whole repositories:
# {repo_path, repo_id, commit_id, github_metadata, num_files,
#  files: [{content_id, content, size_bytes, file_path, file_timestamp,
#           language, is_vendor, license_type, detected_licenses}]}
# (every field a str; verified by a one-row probe on 2026-09-02). Per-file
# language acceptance probabilities steer the mix toward data plan §1.4.
# They are NOT target shares: a 300-repo sample was C# 25%, Java 20%,
# JS 9%, Markdown 6%, Python 3.4%, Kotlin 3%, C 2.9%, Go/TS 2.3%, so the
# dominant languages are damped and the wanted ones kept whole. The
# formatter tallies the realised mix (per stream) and prints it every
# 5,000 files.
STACK_V3_LANG_ACCEPT: dict[str, float] = {
    "Python": 1.0, "Rust": 1.0, "Go": 1.0, "TypeScript": 1.0, "TSX": 1.0,
    "C": 1.0, "C++": 1.0, "SQL": 1.0, "Shell": 1.0, "Lua": 1.0,
    "JavaScript": 0.6, "Ruby": 0.8, "Kotlin": 0.5, "Swift": 0.5, "PHP": 0.4,
    "Java": 0.25, "C#": 0.15,
    "Markdown": 0.3, "YAML": 0.2, "TOML": 0.5, "JSON": 0.1, "Dockerfile": 1.0,
    "CMake": 0.5, "Makefile": 0.8,
    # Data and markup dumps are not code: explicit zero so they never ride
    # the default below.
    "CSV": 0.0, "TSV": 0.0, "SVG": 0.0, "XML": 0.0, "Text": 0.0,
    # "*" is the rate for every language NOT listed (Scala, Haskell, Elixir,
    # Zig, HTML, ...). Until 2026-09-30 an unlisted language was silently
    # p = 0, so the whole long tail was absent from the code mix.
    "*": 0.25,
}
STACK_V3_MAX_FILES = 24
STACK_V3_MAX_CHARS = 120_000


def _stack_v3_keep(f: dict, accept: dict[str, float]) -> bool:
    if str(f.get("is_vendor", "False")) == "True":
        return False
    lang = f.get("language")
    if not lang:
        return False
    p = accept.get(str(lang), accept.get("*", 0.0))
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    # Deterministic per file: hash the content id into [0, 1).
    cid = str(f.get("content_id", ""))[:8] or "0"
    u = int(cid, 16) / 0xFFFFFFFF if all(c in "0123456789abcdef" for c in cid) else 0.5
    return u < p


def _format_stack_v3_with(example: dict, accept: dict[str, float], tally: dict) -> str:
    files = example.get("files") or []
    if not isinstance(files, list):
        return ""
    parts: list[str] = []
    total = 0
    for f in files:
        if len(parts) >= STACK_V3_MAX_FILES or total >= STACK_V3_MAX_CHARS:
            break
        if not isinstance(f, dict) or not _stack_v3_keep(f, accept):
            continue
        content = f.get("content") or ""
        if not isinstance(content, str) or not content.strip():
            continue
        piece = f"# {f.get('file_path', '')}\n{content}"
        # Cap checked per file BEFORE appending: one oversized file is
        # skipped and the rest of the repo still fits. (Checking after the
        # append let a single huge file blow the cap and evict everything
        # behind it.)
        if len(piece) > STACK_V3_MAX_CHARS - total:
            continue
        lang = str(f.get("language"))
        tally[lang] = tally.get(lang, 0) + 1
        n = sum(tally.values())
        if n % 5000 == 0:
            top = sorted(tally.items(), key=lambda kv: -kv[1])[:14]
            print(
                f"[DataWorker] stack-v3 realised language mix after {n} files: "
                + ", ".join(f"{k}={c / n:.1%}" for k, c in top),
                flush=True,
            )
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


def _stack_tally(state: dict | None) -> dict:
    """The per-stream language tally (a throwaway dict when called bare)."""
    return state.setdefault("stack_v3_tally", {}) if state is not None else {}


def _format_stack_v3(example: dict, state: dict | None = None) -> str:
    """Stack v3 repo row -> concatenated kept files, language-steered (§1.4)."""
    return _format_stack_v3_with(example, STACK_V3_LANG_ACCEPT, _stack_tally(state))


def _format_stack_v3_nonpython(example: dict, state: dict | None = None) -> str:
    """Anneal variant: same table with Python dropped, so the other languages
    stay alive through the decay while Python comes from the curated sets."""
    return _format_stack_v3_with(
        example, {**STACK_V3_LANG_ACCEPT, "Python": 0.0}, _stack_tally(state)
    )


def row_passes(ds_cfg: dict, example: dict, rng: random.Random) -> bool:
    """Per-dataset row gate, driven by two optional keys on the dataset entry.

    `filter`: {field: value | [values]} — the row must match EVERY field;
        a missing field rejects the row. Used for allow-lists such as
        `{"language": ["Python", "Rust", ...]}` on Stack v3.
    `subsample`: {field: {value: p, ..., "*": p_default}} — the row is kept
        with probability p for its field value; unlisted values use "*"
        (default 1.0). Acceptance probabilities, not target shares: the
        realised mix is base-distribution x p, so it is logged (see
        TokenStream) rather than assumed.
    """
    flt = ds_cfg.get("filter")
    if flt:
        for field, allowed in flt.items():
            if field not in example:
                return False
            allowed = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
            if example[field] not in allowed:
                return False
    sub = ds_cfg.get("subsample")
    if sub:
        for field, table in sub.items():
            value = example.get(field)
            p = table.get(value, table.get("*", 1.0))
            if p < 1.0 and rng.random() >= p:
                return False
    return True


def _mix_fields(ds_cfg: dict) -> list[str]:
    """Fields whose realised value mix is worth logging (filter/subsample keys)."""
    fields: list[str] = []
    for key in ("filter", "subsample"):
        fields.extend(f for f in (ds_cfg.get(key) or {}) if f not in fields)
    return fields


FORMAT_FN_PRETRAIN = {
    "nemotron_sft": _format_nemotron_sft_text,
    "stack_code": _format_stack_code,
    "arxiv": _format_arxiv,
    "nemotron_math": _format_nemotron_math_pretrain,
    "openr1_math": _format_openr1_math,
    "openmath_reasoning": _format_openmath_reasoning,
    "openthoughts": _format_openthoughts,
    "magicoder": _format_magicoder,
    "magicoder_oss": _format_magicoder_oss,
    "bbh": _format_bbh,
    "openmath_instruct2": _format_openmath_instruct2,
    "io_pair": _format_io_pair,
    "stack_v3": _format_stack_v3,
    "stack_v3_nonpython": _format_stack_v3_nonpython,
}


# ── Generic extraction (no `format` key) ────────────────────────────────


def _messages_to_chat(msgs: object) -> str:
    """`messages` / `conversations` -> canonical chat text ("" if unusable).

    Accepts both {role, content} and ShareGPT {from, value} entries; role
    aliases (human/gpt/model) are resolved by `render_chat`.
    """
    if not isinstance(msgs, list):
        return ""
    norm: list[dict] = []
    for m in msgs:
        if not isinstance(m, dict):
            return ""
        norm.append({
            "role": m.get("role", m.get("from")),
            "content": m.get("content", m.get("value")),
        })
    return render_chat(norm)


def _extract_text(example: dict) -> tuple[str, str]:
    """Generic row -> (text, branch). Branch names the column layout taken.

    Non-string fields yield "" (the row is skipped) instead of raising.
    """
    if "messages" in example:
        return _messages_to_chat(example["messages"]), "messages"

    # Conversations format (OpenHermes, SlimOrca, ShareGPT)
    if "conversations" in example:
        return _messages_to_chat(example["conversations"]), "conversations"

    # Code format (content column)
    if "content" in example:
        c = example["content"]
        return (c if isinstance(c, str) else ""), "content"

    # Instruction/output format (Alpaca, Evol-Instruct-Code, OpenCoder)
    if "instruction" in example and "output" in example:
        instr, out = example["instruction"], example["output"]
        inp = example.get("input", "")
        if not (isinstance(instr, str) and isinstance(out, str)):
            return "", "instruction_output"
        user = instr
        if isinstance(inp, str) and inp.strip():
            user = f"{instr.rstrip()}\n{inp.strip()}"
        return _chat(user, out), "instruction_output"

    # Plain text
    if "text" in example:
        t = example["text"]
        return (t if isinstance(t, str) else ""), "text"

    return "", "none"


# ── The per-row pipeline: gate -> format -> encode -> cap ───────────────


@dataclass
class RowResult:
    """Outcome of one row. `tokens is None` means the row was skipped."""

    tokens: list[int] | None
    reason: str          # "ok" | "filter" | "empty" | "max_tokens" | "error"
    path: str            # "format:<key>" or "extract:<branch>"
    text: str            # rendered text ("" unless it was rendered)
    error: BaseException | None = None


def process_row(
    ds_cfg: dict,
    example: dict,
    tok,
    rng: random.Random,
    state: dict | None = None,
) -> RowResult:
    """Run one row through the exact pipeline the stream uses.

    `state` is the per-(stream, source) formatter scratch dict. A formatter
    or encoder exception is captured as `reason="error"` (with the exception
    in `.error`) rather than propagated, so one malformed row can never kill
    an iterator. An unknown `format` key is a config error and raises.
    """
    fmt = ds_cfg.get("format")
    fn = None
    if fmt:
        fn = FORMAT_FN_PRETRAIN.get(fmt)
        if fn is None:
            raise ValueError(
                f"Unknown pretrain format key '{fmt}' on dataset "
                f"{ds_cfg.get('name', ds_cfg.get('hf_id'))}. "
                f"Valid: {sorted(FORMAT_FN_PRETRAIN)}",
            )
    path = f"format:{fmt}" if fmt else "extract:?"
    try:
        if not isinstance(example, dict):
            return RowResult(None, "error", path, "",
                             TypeError(f"row is {type(example).__name__}, not dict"))
        if not row_passes(ds_cfg, example, rng):
            return RowResult(None, "filter", path, "")
        if fn is not None:
            text = fn(example, state)
        else:
            text, branch = _extract_text(example)
            path = f"extract:{branch}"
        if not isinstance(text, str) or not text.strip():
            return RowResult(None, "empty", path, "")
        tokens = encode_with_markers(tok, text)
        max_tok = ds_cfg.get("max_tokens")
        if max_tok and len(tokens) > max_tok:
            # e.g. cap Nemotron-Science MCQ traces at 2K (data plan §1.3)
            return RowResult(None, "max_tokens", path, text)
        return RowResult(tokens, "ok", path, text)
    except Exception as exc:  # noqa: BLE001 — one bad row must not kill the run
        return RowResult(None, "error", path, "", exc)


def row_to_tokens(
    ds_cfg: dict,
    example: dict,
    tok,
    rng: random.Random,
    state: dict | None = None,
) -> list[int] | None:
    """Row -> token ids, or None when skipped (filter/subsample, empty text,
    over `max_tokens`, or malformed). The single code path shared by
    `TokenStream` and `scripts/preflight_data.py`."""
    return process_row(ds_cfg, example, tok, rng, state).tokens


@functools.lru_cache(maxsize=8)
def _load_tokenizer(tok_name: str):
    tok = AutoTokenizer.from_pretrained(tok_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _to_plain(obj):
    """Restrict a state to dict/list/int/float/str/bool/None so the checkpoint
    loads under `torch.load(weights_only=True)`. Raises TypeError otherwise."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    item = getattr(obj, "item", None)          # numpy scalars
    if callable(item):
        v = item()
        if isinstance(v, (bool, int, float, str)):
            return v
    raise TypeError(f"unsupported type in data state: {type(obj).__name__}")


class DataSourceDead(RuntimeError):
    """Every configured source has been declared dead (see `TokenStream`)."""

    def __init__(self, sources: list[str]) -> None:
        self.sources: list[str] = list(sources)
        super().__init__(
            "every data source is dead: " + ", ".join(self.sources)
        )


SHUFFLE_BUFFER_SIZE = 5_000
# Epoch boundaries worth a log line: the first few, then every 100th.
_CYCLE_LOG_POINTS = frozenset({1, 2, 3, 5, 10, 20, 50, 100})
_REPORT_EVERY_ROWS = 5_000
_REJECT_WARN_EVERY = 10_000
_STATE_DICT_UNSUPPORTED_WARNED = False


class TokenStream(IterableDataset):
    """Streaming token dataset with multi-dataset weighted sampling.

    Streams from multiple HuggingFace datasets simultaneously, sampling
    according to configured weights (by realised tokens, not rows). Every
    chat-shaped row is rendered with `render_chat`; every text is tokenised
    with `encode_with_markers`; documents are packed back to back with one
    EOS between them into `seq_len` chunks. `labels == input_ids` — the
    model applies the one-position shift internally.

    Dead-source policy: when a stream cannot be opened at startup (after
    `load_dataset`'s bounded retries), when its fetch gives up (`_robust_next`
    exhausts its retries) `max_stream_failures` times in a row, or when it
    rejects `max_consecutive_rejections` rows in a row, it is marked
    dead — announced once, removed from the sampler (remaining weights are
    renormalised) and listed in `dead_sources`. When no live stream is left
    `DataSourceDead` is raised. A successfully fetched row resets the fetch
    counter; an accepted row resets the rejection counter.

    Resume: `state_dict()` (None before iteration starts, or in a DataLoader
    worker) captures each stream's `datasets` position within its current
    shuffled epoch, epoch count and shuffle seed, plus the token buffer,
    per-stream token counts, the sampler RNG, the dead set (recorded for the
    report; a new process retries every source) and the mix counts. Pass it
    back as `resume_state` to continue; shuffle-buffer contents are
    legitimately lost on resume (see `datasets` docs), shard and row
    progress are not.

    Args:
        dataset_configs: List of dataset config dicts with hf_id, weight, etc.
        seq_len: Sequence length for this phase.
        tok_name: HuggingFace tokenizer identifier.
        seed: Random seed for shuffling.
        resume_state: A previous `state_dict()` for the same source list.
        max_stream_failures: Consecutive fetch give-ups before a source is dead.
        max_consecutive_rejections: Consecutive rejected rows before a source
            is dead (warned every 10,000 with the reason histogram).
    """

    def __init__(
        self,
        dataset_configs: list[dict],
        seq_len: int,
        tok_name: str,
        seed: int,
        *,
        resume_state: dict | None = None,
        max_stream_failures: int = 3,
        max_consecutive_rejections: int = 200_000,
    ) -> None:
        self.dataset_configs = dataset_configs
        self.seq_len = seq_len
        self.tok_name = tok_name
        self.seed = seed
        self.resume_state = resume_state
        self.max_stream_failures = max_stream_failures
        self.max_consecutive_rejections = max_consecutive_rejections
        self.dead_sources: list[str] = []
        self._live: dict | None = None
        for cfg in dataset_configs:
            fmt = cfg.get("format")
            if fmt and fmt not in FORMAT_FN_PRETRAIN:
                raise ValueError(
                    f"Unknown pretrain format key '{fmt}' on dataset "
                    f"{cfg.get('name', cfg.get('hf_id'))}. "
                    f"Valid: {sorted(FORMAT_FN_PRETRAIN)}",
                )

    # Worker processes get a pickled copy: never ship live generators.
    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["_live"] = None
        return d

    def _name(self, i: int) -> str:
        cfg = self.dataset_configs[i]
        return cfg.get("name", cfg["hf_id"])

    # ── resume state ────────────────────────────────────────────────

    def state_dict(self) -> dict | None:
        """The stream's position, or None when it cannot report one (before
        iteration, or inside a DataLoader worker)."""
        global _STATE_DICT_UNSUPPORTED_WARNED
        live = self._live
        if live is None or torch.utils.data.get_worker_info() is not None:
            return None
        streams = []
        for st in live["streams"]:
            ds_state = None
            shuffled = st["shuffled"]
            if shuffled is not None:
                fn = getattr(shuffled, "state_dict", None)
                if fn is None:
                    if not _STATE_DICT_UNSUPPORTED_WARNED:
                        _STATE_DICT_UNSUPPORTED_WARNED = True
                        print(
                            "[DataWorker] this `datasets` version has no "
                            "IterableDataset.state_dict — data positions will "
                            "NOT be checkpointed (upgrade datasets >= 2.18)",
                            flush=True,
                        )
                    return None
                try:
                    ds_state = _to_plain(fn())
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[DataWorker] {st['name']}: position not captured: "
                        f"{type(exc).__name__}: {str(exc)[:120]}",
                        flush=True,
                    )
                    ds_state = None
            streams.append({
                "name": st["name"],
                "cycles": st["cycles"],
                "shuffle_seed": st["shuffle_seed"],
                "ds_state": ds_state,
            })
        rng_state = live["rng"].getstate()
        return {
            "version": 1,
            "sources": [st["name"] for st in live["streams"]],
            "seq_len": self.seq_len,
            "streams": streams,
            "tokens_seen": list(live["tokens_seen"]),
            "rows_ok": list(live["rows_ok"]),
            "skipped": copy.deepcopy(live["skipped"]),
            "buffer": list(live["buffer"]),
            "rng": [rng_state[0], list(rng_state[1]), rng_state[2]],
            "dead": [st["name"] for st in live["streams"] if st["dead"]],
            "mix_counts": copy.deepcopy(live["mix_counts"]),
        }

    def _apply_resume(self, live: dict, resume: dict) -> None:
        names = [st["name"] for st in live["streams"]]
        if resume.get("sources") != names:
            print(
                f"[DataWorker] resume_state is for sources {resume.get('sources')} "
                f"but this loader has {names}; starting every source fresh",
                flush=True,
            )
            return
        n = len(names)
        live["buffer"][:] = [int(t) for t in resume.get("buffer", [])]
        live["tokens_seen"][:] = list(resume.get("tokens_seen") or [0] * n)
        live["rows_ok"][:] = list(resume.get("rows_ok") or [0] * n)
        live["skipped"][:] = copy.deepcopy(
            resume.get("skipped") or [{} for _ in range(n)]
        )
        live["mix_counts"].update(copy.deepcopy(resume.get("mix_counts") or {}))
        rs = resume.get("rng")
        if rs:
            live["rng"].setstate((rs[0], tuple(rs[1]), rs[2]))
        dead = set(resume.get("dead") or [])
        restored, fresh = [], []
        for st, saved in zip(live["streams"], resume.get("streams") or []):
            st["cycles"] = int(saved.get("cycles", 0))
            st["shuffle_seed"] = int(saved.get("shuffle_seed", st["shuffle_seed"]))
            if saved.get("ds_state") is not None:
                st["pending_ds_state"] = saved["ds_state"]
                restored.append(st["name"])
            else:
                fresh.append(st["name"])
        # A source the previous process declared dead is NOT restored as
        # dead: a new process is a fresh chance (the outage or credentials may
        # have been fixed — that is the advertised "fix the data, then
        # re-run" recovery). It is reconnected like any other source below,
        # and declared dead again there if it still fails. Persisting the
        # all-dead set of a data-dead rescue would otherwise make every
        # re-run fail on the spot without a single connection attempt.
        print(
            "[DataWorker] resumed data position — restored: "
            + (", ".join(restored) or "none")
            + (f"; no saved position: {', '.join(fresh)}" if fresh else "")
            + (f"; previously dead, retrying in this process: "
               f"{', '.join(sorted(dead))}" if dead else "")
            + f"; buffer={len(live['buffer'])} tokens",
            flush=True,
        )

    # ── iteration ───────────────────────────────────────────────────

    def __iter__(self):  # noqa: ANN204
        from datasets import load_dataset

        tok = _load_tokenizer(self.tok_name)
        eos_id = tok.eos_token_id

        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed if worker_info is None else self.seed + worker_info.id
        rng = random.Random(seed)
        n = len(self.dataset_configs)
        streams: list[dict] = [
            {
                "idx": i,
                "name": self._name(i),
                "base": None,          # unshuffled datasets.IterableDataset
                "shuffled": None,      # the current epoch's shuffled view
                "gen": None,           # the infinite cycling generator
                "cycles": 0,           # completed epochs
                "shuffle_seed": seed,  # seed of the current epoch's shuffle
                "pending_ds_state": None,
                "dead": False,
                "fail_streak": 0,      # consecutive _robust_next give-ups
                "reject_streak": 0,    # consecutive rejected rows
                "reject_hist": {},     # reasons behind the current streak
                "fmt_state": {},       # formatter scratch (stack v3 tally)
                "error_logged": False,
            }
            for i in range(n)
        ]
        live = {
            "streams": streams,
            "tokens_seen": [0] * n,
            "rows_ok": [0] * n,
            "skipped": [{} for _ in range(n)],
            "buffer": [],
            "rng": rng,
            "mix_counts": {},
        }
        self.dead_sources = []
        if self.resume_state is not None:
            if worker_info is None:
                self._apply_resume(live, self.resume_state)
            else:
                print(
                    "[DataWorker] resume_state ignored: positions cannot be "
                    "restored into DataLoader worker processes (num_workers>0)",
                    flush=True,
                )
        self._live = live
        buffer: list[int] = live["buffer"]
        tokens_seen = live["tokens_seen"]
        rows_ok = live["rows_ok"]
        skipped = live["skipped"]
        mix_counts = live["mix_counts"]
        weights = [float(cfg["weight"]) for cfg in self.dataset_configs]

        def _open_base(i: int):
            """`load_dataset` with bounded retries. Used for the initial
            connect AND mid-run reconnects, so a transient HF Hub error
            during stream setup doesn't kill the whole run (the ablate stage
            hit exactly that: cell A worked, cell B's load_dataset raised
            "Cannot send a request, as the client has been closed")."""
            ds_cfg = self.dataset_configs[i]
            load_kwargs = {
                "split": ds_cfg.get("split", "train"),
                "streaming": True,
            }
            if ds_cfg.get("hf_config"):
                load_kwargs["name"] = ds_cfg["hf_config"]
            last_exc = None
            for attempt in range(1, 6):  # 5 attempts max
                try:
                    ds = load_dataset(ds_cfg["hf_id"], **load_kwargs)
                    skip_n = ds_cfg.get("skip", 0)
                    if skip_n > 0:
                        ds = ds.skip(skip_n)
                    return ds
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    print(
                        f"[DataWorker] {ds_cfg['hf_id']} setup attempt "
                        f"{attempt}/5 failed: {type(exc).__name__}: "
                        f"{str(exc)[:120]} — retrying...",
                        flush=True,
                    )
                    time.sleep(2 * attempt)
            raise RuntimeError(
                f"Failed to open stream for {ds_cfg['hf_id']} after 5 "
                f"attempts: {last_exc}",
            )

        def _skipped_str(i: int) -> str:
            parts = [f"{k}={v}" for k, v in sorted(skipped[i].items())]
            return ", ".join(parts) or "none"

        def _source_report(i: int) -> str:
            st = streams[i]
            total = sum(tokens_seen[j] for j in range(n) if not streams[j]["dead"])
            live_w = sum(weights[j] for j in range(n) if not streams[j]["dead"])
            share = tokens_seen[i] / total if total else 0.0
            target = weights[i] / live_w if live_w and not st["dead"] else 0.0
            return (
                f"[DataWorker] {st['name']}: rows={rows_ok[i]:,} epochs={st['cycles']} "
                f"token_share={share:.1%} (target {target:.1%}) "
                f"skipped[{_skipped_str(i)}]"
            )

        def _cycling_iter(st: dict):
            """Wrap an UNSHUFFLED streaming dataset in an infinite cycle.

            Small datasets like Magicoder (110K rows), OpenThoughts
            (114K), and BBH (250) exhaust their streaming iterators
            during a multi-thousand-step run, and the debt-based
            sampler then deadlocks: every worker thrashes on the same
            empty-stream reconnect because the unfulfilled deficit
            keeps the sampler picking it. Cycling internally means
            the stream is always ready to yield, so the deficit
            actually drains and the sampler moves on.

            Each epoch re-shuffles the UNSHUFFLED base with the next
            seed — NOT the previous epoch's shuffled view. Nesting
            ShuffledIterableDatasets every cycle blew the recursion limit
            on tiny datasets (v6 midtrain knowledge-mix probe). One shuffle
            layer, always. The current shuffled view stays on `st` so
            `state_dict()` can read its position; a `pending_ds_state`
            (resume or position-keeping reconnect) is loaded into the
            freshly built view before it is iterated.
            """
            while True:
                if st["shuffled"] is None:
                    st["shuffled"] = st["base"].shuffle(
                        buffer_size=SHUFFLE_BUFFER_SIZE, seed=st["shuffle_seed"]
                    )
                    pending = st.pop("pending_ds_state", None)
                    if pending is not None:
                        try:
                            st["shuffled"].load_state_dict(pending)
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"[DataWorker] {st['name']}: saved position not "
                                f"loadable ({type(exc).__name__}: {str(exc)[:100]}); "
                                f"starting the epoch from its first shard",
                                flush=True,
                            )
                for ex in st["shuffled"]:
                    yield ex
                st["cycles"] += 1
                st["shuffle_seed"] += 1
                st["shuffled"] = None
                c = st["cycles"]
                if c in _CYCLE_LOG_POINTS or c % 100 == 0:
                    print(
                        f"[DataWorker] {st['name']}: epoch {c} complete — "
                        f"re-shuffling with seed {st['shuffle_seed']} "
                        f"(small dataset, expected). {_source_report(st['idx'])}",
                        flush=True,
                    )

        def _saved_position(st: dict) -> dict | None:
            shuffled = st["shuffled"]
            fn = getattr(shuffled, "state_dict", None) if shuffled is not None else None
            if fn is None:
                return None
            try:
                return fn()
            except Exception:  # noqa: BLE001
                return None

        def _reconnect_stream(i: int, keep_position: bool) -> bool:
            """Rebuild one stream through the same retry-aware open path.

            `keep_position` (first retry after a transient error) re-shuffles
            with the SAME seed and reloads the last yielded position, so a
            network blip costs no data. Later retries fall back to a fresh
            random permutation, so a poisoned shard is not re-hit forever.
            Returns True on success; swallows failures (returns False) so a
            reconnect that itself errors can't propagate and kill the worker.
            """
            st = streams[i]
            pending = _saved_position(st) if keep_position else None
            try:
                st["base"] = _open_base(i)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[DataWorker] reconnect to {st['name']} failed: "
                    f"{type(exc).__name__}: {str(exc)[:120]}",
                    flush=True,
                )
                return False
            if pending is not None:
                st["pending_ds_state"] = pending
            else:
                st["shuffle_seed"] = seed + rng.randint(1, 100_000)
            st["shuffled"] = None
            st["gen"] = _cycling_iter(st)
            print(
                f"[DataWorker] Reconnected to {st['name']}"
                + (" (position kept)" if pending is not None else " (fresh shuffle)"),
                flush=True,
            )
            return True

        max_retries = 8

        def _robust_next(i: int):
            """Fetch the next example, surviving ANY failure.

            Catches EVERYTHING (StopIteration, connection resets, SSL
            errors, closed-client RuntimeErrors), reconnects with bounded
            exponential backoff, and returns None if it can't recover this
            attempt. The caller counts those give-ups towards the
            dead-source threshold instead of retrying the same stream
            forever.
            """
            st = streams[i]
            for attempt in range(1, max_retries + 1):
                try:
                    return next(st["gen"])
                except StopIteration:
                    # _cycling_iter makes this unreachable unless the
                    # generator died on an earlier error; rebuild it.
                    _reconnect_stream(i, keep_position=False)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[DataWorker] {st['name']}: {type(exc).__name__}: "
                        f"{str(exc)[:160]} — reconnect [{attempt}/{max_retries}]",
                        flush=True,
                    )
                    time.sleep(min(2 * attempt, 30))
                    _reconnect_stream(i, keep_position=(attempt == 1))
            print(
                f"[DataWorker] {st['name']}: giving up after {max_retries} "
                f"retries (failure {st['fail_streak'] + 1}/{self.max_stream_failures} "
                f"before the source is declared dead)",
                flush=True,
            )
            return None

        def _mark_dead(i: int, why: str) -> None:
            st = streams[i]
            st["dead"] = True
            self.dead_sources.append(st["name"])
            live_idx = [j for j in range(n) if not streams[j]["dead"]]
            live_w = sum(weights[j] for j in live_idx)
            remaining = ", ".join(
                f"{streams[j]['name']}={weights[j] / live_w:.1%}" for j in live_idx
            ) or "NONE"
            bar = "!" * 78
            print(
                f"\n{bar}\n[DataWorker] SOURCE DEAD: {st['name']} "
                f"({self.dataset_configs[i]['hf_id']}) — {why}\n"
                f"[DataWorker] removed from the sampler; "
                f"live weights now: {remaining}\n"
                f"[DataWorker] {_source_report(i)}\n{bar}\n",
                flush=True,
            )

        # Connect every stream. Every source gets a fresh attempt in a new
        # process, including ones a previous process declared dead.
        # A source that cannot be opened even after `_open_base`'s retries is
        # declared dead like a mid-run failure would be, so one unavailable
        # dataset does not take the healthy ones down with it; when none can
        # be opened the same DataSourceDead the sampler would raise is raised
        # here, before any compute is spent.
        for st in streams:
            if st["dead"]:
                continue
            print(f"[DataWorker] Connecting to {st['name']}...", flush=True)
            try:
                st["base"] = _open_base(st["idx"])
            except Exception as exc:  # noqa: BLE001 — reported via _mark_dead
                _mark_dead(
                    st["idx"],
                    f"unreachable at startup: {type(exc).__name__}: "
                    f"{str(exc)[:160]}",
                )
                continue
            st["gen"] = _cycling_iter(st)
            print(f"[DataWorker] Stream ready for {st['name']}", flush=True)
        if all(st["dead"] for st in streams):
            raise DataSourceDead(self.dead_sources)

        # Token-weighted sampling: pick the stream whose observed token
        # fraction is furthest behind its configured target. This makes
        # a "60% FineWeb / 40% CodeParrot" config produce the actual
        # token mix regardless of per-stream example lengths — otherwise
        # code streams with longer examples would dominate. Starts
        # weight-random during the bootstrap phase (no tokens seen yet).
        # Dead streams drop out and the remaining weights renormalise.
        def _pick_stream() -> int:
            live_idx = [j for j in range(n) if not streams[j]["dead"]]
            if not live_idx:
                raise DataSourceDead(self.dead_sources)
            live_w = sum(weights[j] for j in live_idx)
            total = sum(tokens_seen[j] for j in live_idx)
            if total == 0:
                return rng.choices(
                    live_idx, weights=[weights[j] / live_w for j in live_idx], k=1
                )[0]
            deficits = {
                j: weights[j] / live_w - tokens_seen[j] / total for j in live_idx
            }
            max_def = max(deficits.values())
            candidates = [j for j, d in deficits.items() if d >= max_def - 1e-6]
            return rng.choice(candidates)

        while True:
            idx = _pick_stream()
            st = streams[idx]
            ds_cfg_i = self.dataset_configs[idx]
            ds_name = st["name"]

            example = _robust_next(idx)
            if example is None:
                st["fail_streak"] += 1
                if st["fail_streak"] >= self.max_stream_failures:
                    _mark_dead(
                        idx,
                        f"{st['fail_streak']} consecutive fetch failures "
                        f"({max_retries} reconnects each)",
                    )
                continue
            st["fail_streak"] = 0

            res = process_row(ds_cfg_i, example, tok, rng, st["fmt_state"])
            if res.tokens is None:
                skipped[idx][res.reason] = skipped[idx].get(res.reason, 0) + 1
                st["reject_streak"] += 1
                st["reject_hist"][res.reason] = st["reject_hist"].get(res.reason, 0) + 1
                if res.error is not None and not st["error_logged"]:
                    st["error_logged"] = True
                    keys = (
                        sorted(example.keys()) if isinstance(example, dict)
                        else type(example).__name__
                    )
                    print(
                        f"[DataWorker] {ds_name}: row error "
                        f"{type(res.error).__name__}: {str(res.error)[:160]} "
                        f"(row keys: {keys}) — skipped; further errors on this "
                        f"source are counted, not printed",
                        flush=True,
                    )
                if st["reject_streak"] % _REJECT_WARN_EVERY == 0:
                    print(
                        f"[DataWorker] WARNING {ds_name}: {st['reject_streak']:,} "
                        f"consecutive rows rejected ({st['reject_hist']}); dead at "
                        f"{self.max_consecutive_rejections:,}",
                        flush=True,
                    )
                if st["reject_streak"] >= self.max_consecutive_rejections:
                    _mark_dead(
                        idx,
                        f"{st['reject_streak']:,} consecutive rows rejected "
                        f"({st['reject_hist']})",
                    )
                continue
            st["reject_streak"] = 0
            st["reject_hist"] = {}
            rows_ok[idx] += 1

            for field in _mix_fields(ds_cfg_i):
                tally = mix_counts.setdefault(ds_name, {}).setdefault(field, {})
                v = str(example.get(field))
                tally[v] = tally.get(v, 0) + 1
                cnt = sum(tally.values())
                if cnt % 5000 == 0:
                    top = sorted(tally.items(), key=lambda kv: -kv[1])[:12]
                    print(
                        f"[DataWorker] {ds_name} realised {field} mix after "
                        f"{cnt} rows: " + ", ".join(
                            f"{k}={c / cnt:.1%}" for k, c in top),
                        flush=True,
                    )

            tokens = res.tokens
            buffer.extend(tokens)
            buffer.append(eos_id)
            # Record token count for the debt-based sampler. We count
            # real tokens but not the structural EOS so code streams
            # with many short examples aren't artificially inflated.
            tokens_seen[idx] += len(tokens)
            if rows_ok[idx] % _REPORT_EVERY_ROWS == 0:
                print(_source_report(idx), flush=True)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                del buffer[: self.seq_len]     # in place: `buffer` IS the live state
                # Labels are aligned with input_ids — the model shifts
                # internally (model.py). Yielding chunk[1:] here would
                # double-shift, so position i would be trained to predict
                # token i+2 instead of i+1.
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                yield input_ids, input_ids.clone()


def make_loader(
    dataset_configs: list[dict],
    seq_len: int,
    tokenizer_name: str,
    batch_size: int,
    step_num: int,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    resume_state: dict | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a streaming DataLoader for a training phase.

    Args:
        dataset_configs: List of dataset config dicts for this phase.
        seq_len: Sequence length for this phase.
        tokenizer_name: HuggingFace tokenizer identifier.
        batch_size: Micro-batch size.
        step_num: Current step (used to vary shuffle seed).
        resume_state: A `TokenStream.state_dict()` saved by a loader over the
            same source list; restores every stream's position. With
            `num_workers=0` the trainer reads `loader.dataset.state_dict()`.

    Returns:
        DataLoader yielding (input_ids, labels) batches.
    """
    ds = TokenStream(
        dataset_configs, seq_len, tokenizer_name, seed=42 + step_num,
        resume_state=resume_state,
    )
    # num_workers>0 offloads HF streaming + BPE tokenisation to background
    # processes so the main training thread doesn't wait on them. Each
    # worker gets its own seed (TokenStream offsets by worker_info.id) so
    # they don't produce duplicate batches. persistent_workers keeps the
    # processes across phase transitions — new loaders still spawn fresh
    # workers, but within a phase we don't tear them down for every step's
    # reload. prefetch_factor keeps a small queue of ready batches per worker.
    # Positions cannot be checkpointed or restored across workers, so
    # state_dict() is None there; num_workers=0 (the trunk default) has it.
    #
    # multiprocessing_context="spawn" is critical. The default on Linux
    # is fork, which inherits the parent's threadpool state — tokenizers-rs
    # threads, torch's intra-op pool, wandb sync thread, HF datasets' xet
    # client — any of which holds a mutex at fork time and silently
    # deadlocks the child. Observed failure mode: worker stuck before its
    # first print, DataLoader iter() blocks forever. Spawn starts fresh
    # interpreters and re-imports modules cleanly; TokenStream's simple
    # fields (list[dict], int, str) serialise cleanly across the boundary.
    # num_workers default of 4 preserved for backwards compat; stages
    # with many small HF streams should lower this (each worker spawns
    # its own copy of every stream, so total HF connections grow as
    # num_workers × n_streams — at 4×9 = 36 we hit "Bad file descriptor"
    # and "Connection reset by peer" cascades that crash workers and
    # then the input). Pass num_workers=1 from the stage config to cut
    # the connection storm.
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
