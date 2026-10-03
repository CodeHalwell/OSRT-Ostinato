"""SFT data: the pretraining token stream plus a loss mask on assistant turns.

Everything about streaming, retrying, weighting and packing is `TokenStream`
(`osrt.data`); SFT adds two things on top:

1. `format="sft"` — any chat-shaped row (messages / conversations /
   instruction+output / input+output / problem+generated_solution) rendered
   with `render_chat`, with Python indentation normalised in the ASSISTANT
   text first (the trunk emits tab-indented bodies under space-indented
   signatures; see roadmap §20.5).
2. `SFTStream` — the same packed windows as `TokenStream`, but `labels` are
   -100 everywhere except inside assistant spans, i.e. the tokens AFTER an
   `<|assistant|>` marker up to and including the following `<|end_turn|>`.
   Spans that cross a window boundary are carried over, so a long answer is
   trained in full across windows.

Labels stay ALIGNED with input_ids (the model shifts internally, like the
pretraining loader): labels[j] is the target that position j-1 predicts.
"""

from __future__ import annotations

import re

import torch
from torch import Tensor

from osrt.chat_format import ASSISTANT, END_TURN, render_chat
from osrt.data import FORMAT_FN_PRETRAIN, TokenStream, _load_tokenizer

IGNORE_INDEX = -100

_FENCE_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<body>.*?)```", re.S)
_PY_LINE_START = re.compile(r"^(?:def |class |import |from \S+ import )", re.M)


def _tabs_to_spaces(code: str) -> str:
    """Leading tabs -> 4 spaces, line by line; nothing else is touched."""
    out = []
    for line in code.split("\n"):
        n = len(line) - len(line.lstrip("\t"))
        out.append("    " * n + line[n:] if n else line)
    return "\n".join(out)


def normalise_python_indent(text: str) -> str:
    """Normalise leading tabs to 4 spaces in Python code inside `text`.

    Applied to assistant turns only. Python is found as a ```python / ```py
    fence, or — when the turn has no fence at all — as plain text that starts
    lines with def/class/import (OpenCodeInstruct answers are raw code). Other
    languages are left alone: Go and Makefiles are tab-indented on purpose.
    """
    if "\t" not in text:
        return text
    if "```" in text:
        def _fix(m: re.Match) -> str:
            lang = m.group("lang").strip().lower()
            if lang in ("python", "py", "python3"):
                return f"```{m.group('lang')}\n{_tabs_to_spaces(m.group('body'))}```"
            return m.group(0)
        return _FENCE_RE.sub(_fix, text)
    if _PY_LINE_START.search(text):
        return _tabs_to_spaces(text)
    return text


def _as_messages(example: dict) -> list[dict] | None:
    """Row of any supported layout -> OpenAI-style messages, or None."""
    for key in ("messages", "conversations"):
        if key in example and isinstance(example[key], list):
            msgs = []
            for m in example[key]:
                if not isinstance(m, dict):
                    return None
                msgs.append({"role": m.get("role", m.get("from")),
                             "content": m.get("content", m.get("value"))})
            return msgs
    user = answer = None
    if "instruction" in example and "output" in example:
        user, answer = example["instruction"], example["output"]
        inp = example.get("input", "")
        if isinstance(inp, str) and inp.strip() and isinstance(user, str):
            user = f"{user.rstrip()}\n{inp.strip()}"
    elif "input" in example and "output" in example:
        user, answer = example["input"], example["output"]
    elif "problem" in example and "generated_solution" in example:
        user, answer = example["problem"], example["generated_solution"]
    elif "question" in example and "answer" in example:
        user, answer = example["question"], example["answer"]
    if isinstance(user, str) and isinstance(answer, str):
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": answer}]
        sys_prompt = example.get("system")
        if isinstance(sys_prompt, str) and sys_prompt.strip():
            msgs.insert(0, {"role": "system", "content": sys_prompt})
        return msgs
    return None


def format_sft(example: dict, state: dict | None = None) -> str:
    """`format="sft"`: chat row -> canonical chat text with assistant turns
    indentation-normalised. "" skips the row (the stream's convention)."""
    msgs = _as_messages(example)
    if not msgs:
        return ""
    for m in msgs:
        role = m.get("role")
        is_assistant = isinstance(role, str) and role.strip().lower() in (
            "assistant", "gpt", "model")
        if is_assistant and isinstance(m.get("content"), str):
            m["content"] = normalise_python_indent(m["content"])
    return render_chat(msgs)


FORMAT_FN_PRETRAIN["sft"] = format_sft


def assistant_labels(
    input_ids: Tensor, assistant_id: int, end_turn_id: int, in_assistant: bool,
) -> tuple[Tensor, bool]:
    """Labels for one packed window: the token itself inside assistant spans,
    IGNORE_INDEX elsewhere. Returns the carried-over span state."""
    ids = input_ids.tolist()
    labels = [IGNORE_INDEX] * len(ids)
    for j, t in enumerate(ids):
        if in_assistant:
            labels[j] = t
        if t == assistant_id:
            in_assistant = True
        elif t == end_turn_id:
            in_assistant = False
    return torch.tensor(labels, dtype=torch.long), in_assistant


class SFTStream(TokenStream):
    """`TokenStream` windows with assistant-only labels."""

    def __iter__(self):  # noqa: ANN204
        tok = _load_tokenizer(self.tok_name)
        a_id = int(tok.convert_tokens_to_ids(ASSISTANT))
        e_id = int(tok.convert_tokens_to_ids(END_TURN))
        in_assistant = False
        for input_ids, _ in super().__iter__():
            labels, in_assistant = assistant_labels(input_ids, a_id, e_id, in_assistant)
            yield input_ids, labels


def make_sft_loader(
    dataset_configs: list[dict], seq_len: int, tokenizer_name: str,
    batch_size: int, seed: int, resume_state: dict | None = None,
):
    """In-process (num_workers=0) loader of assistant-masked packed windows."""
    from torch.utils.data import DataLoader

    ds = SFTStream(dataset_configs, seq_len, tokenizer_name, seed=seed,
                   resume_state=resume_state)
    return DataLoader(ds, batch_size=batch_size, num_workers=0)
