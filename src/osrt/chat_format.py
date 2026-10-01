"""The OSRT chat contract: one renderer, one Jinja template, one encoder.

Three things used to disagree about what a chat looks like in pretraining:
the shipped tokenizer had no `chat_template` (so `_extract_text` fell back to
`role: content` text with a stray EOS), the `format=` formatters emitted
`<|user|>q<|assistant|>a` without `<|end_turn|>`, and `instruction/output`
rows rendered with no markers at all. `<|system|>` and `<|end_turn|>` never
appeared in pretraining. Everything now goes through `render_chat`, and
`CHAT_TEMPLATE` is the byte-identical Jinja form for inference-time
`apply_chat_template`.

The canonical form has NO newlines between markers — every marker is a single
token, so a newline would be an extra token the model has to learn to emit
and would make `<|end_turn|>` detection depend on whitespace:

    <|system|>{system}<|user|>{q1}<|assistant|>{a1}<|end_turn|><|user|>{q2}...

Content is stripped of surrounding whitespace (Jinja `| trim` == `str.strip`).
"""

from __future__ import annotations

import re

from osrt.tokenizer_contract import OSTINATO_SPECIAL_TOKENS

OSRT_MARKERS: tuple[str, ...] = OSTINATO_SPECIAL_TOKENS

SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END_TURN = "<|end_turn|>"

# Role spellings seen across the mix: ShareGPT (`human`/`gpt`), Gemini-style
# (`model`), and the plain OpenAI names. Anything else (`tool`, `function`,
# `observation`, ...) is not part of the pretraining contract -> the row is
# unusable and renders as "".
_ROLE_ALIASES: dict[str, str] = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
}

# Jinja twin of `render_chat` for transformers' apply_chat_template. It applies
# the same normalisation and validation as the Python renderer — role aliases
# (human -> user, gpt/model -> assistant, case-insensitive), content trimmed,
# no whitespace between markers, a trailing user turn dropped unless
# add_generation_prompt, and "" for a blank or non-string content, a system
# message anywhere but first, or a row with no adjacent user->assistant pair
# (with add_generation_prompt: no turns at all). The ONE deliberate
# difference: a role outside system/user/assistant (or a missing role) raises
# here, where a caller is building a prompt and an error beats an empty
# string, while `render_chat` returns "" so the data stream skips the row.
CHAT_TEMPLATE: str = "".join([
    "{%- set ns = namespace(ok=true, turns=[], system=none, "
    "pair=false, render=false) -%}",
    "{%- for m in messages -%}",
    "{%- set r = m['role'] if m['role'] is string else '' -%}",
    "{%- set r = r | trim | lower -%}",
    "{%- if r == 'human' -%}{%- set r = 'user' -%}",
    "{%- elif r == 'gpt' or r == 'model' -%}{%- set r = 'assistant' -%}",
    "{%- endif -%}",
    "{%- if r not in ['system', 'user', 'assistant'] -%}",
    "{{ raise_exception('OSRT chat template: unsupported role ' ~ m['role']) }}",
    "{%- endif -%}",
    "{%- set c = m['content'] if m['content'] is string else none -%}",
    "{%- if c is none or (c | trim) == '' -%}{%- set ns.ok = false -%}{%- endif -%}",
    "{%- if r == 'system' and not loop.first -%}{%- set ns.ok = false -%}{%- endif -%}",
    "{%- set ns.turns = ns.turns + [[r, (c | trim) if c is string else '']] -%}",
    "{%- endfor -%}",
    "{%- if ns.turns and ns.turns[0][0] == 'system' -%}",
    "{%- set ns.system = ns.turns[0][1] -%}",
    "{%- set ns.turns = ns.turns[1:] -%}",
    "{%- endif -%}",
    "{%- if not add_generation_prompt and ns.turns and ns.turns[-1][0] == 'user' -%}",
    "{%- set ns.turns = ns.turns[:-1] -%}",
    "{%- endif -%}",
    "{%- for i in range(ns.turns | length - 1) -%}",
    "{%- if ns.turns[i][0] == 'user' and ns.turns[i + 1][0] == 'assistant' -%}",
    "{%- set ns.pair = true -%}",
    "{%- endif -%}",
    "{%- endfor -%}",
    "{%- if add_generation_prompt -%}",
    "{%- set ns.render = ns.ok and (ns.turns | length) > 0 -%}",
    "{%- else -%}",
    "{%- set ns.render = ns.ok and ns.pair -%}",
    "{%- endif -%}",
    "{%- if ns.render -%}",
    "{%- if ns.system is not none -%}<|system|>{{ ns.system }}{%- endif -%}",
    "{%- for t in ns.turns -%}",
    "{%- if t[0] == 'user' -%}<|user|>{{ t[1] }}",
    "{%- else -%}<|assistant|>{{ t[1] }}<|end_turn|>",
    "{%- endif -%}",
    "{%- endfor -%}",
    "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}",
    "{%- endif -%}",
])


def _normalise(messages: object) -> list[tuple[str, str]] | None:
    """(role, content) turns, or None when any message makes the row unusable."""
    if not isinstance(messages, (list, tuple)):
        return None
    turns: list[tuple[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            return None
        raw_role = m.get("role")
        role = _ROLE_ALIASES.get(raw_role.strip().lower()) if isinstance(
            raw_role, str) else None
        if role is None:
            return None
        content = m.get("content")
        if not isinstance(content, str):
            return None
        content = content.strip()
        if not content:
            return None
        turns.append((role, content))
    return turns


def render_chat(messages: list[dict], *, add_generation_prompt: bool = False) -> str:
    """Render a `messages` list into the canonical OSRT chat string.

    * optional leading `<|system|>{content}` (a system message anywhere else
      is unsupported -> "");
    * user -> `<|user|>{content}`, assistant -> `<|assistant|>{content}<|end_turn|>`;
    * role aliases: human -> user, gpt/model -> assistant;
    * a trailing user turn with no reply is dropped, unless
      `add_generation_prompt`, in which case the string ends with `<|assistant|>`;
    * "" for rows with no complete user->assistant pair (the stream skips
      them), non-string or blank content, non-dict messages, or roles outside
      system/user/assistant (`tool` is not part of pretraining).
    """
    turns = _normalise(messages)
    if turns is None:
        return ""
    system: str | None = None
    if turns and turns[0][0] == "system":
        system = turns[0][1]
        turns = turns[1:]
    if any(role == "system" for role, _ in turns):
        return ""
    if not add_generation_prompt and turns and turns[-1][0] == "user":
        turns = turns[:-1]
    if not add_generation_prompt:
        has_pair = any(
            turns[i][0] == "user" and turns[i + 1][0] == "assistant"
            for i in range(len(turns) - 1)
        )
        if not has_pair:
            return ""
    elif not turns:
        return ""
    parts: list[str] = []
    if system is not None:
        parts.append(f"{SYSTEM}{system}")
    for role, content in turns:
        if role == "user":
            parts.append(f"{USER}{content}")
        else:
            parts.append(f"{ASSISTANT}{content}{END_TURN}")
    if add_generation_prompt:
        parts.append(ASSISTANT)
    return "".join(parts)


# Longest first so no marker can shadow another (none is a prefix of another
# today, but the sort makes that a non-assumption).
_MARKER_RE = re.compile(
    "|".join(re.escape(m) for m in sorted(OSRT_MARKERS, key=len, reverse=True))
)


def _marker_id(tok, marker: str) -> int:
    tid = tok.convert_tokens_to_ids(marker)
    if tid is None or (tid == tok.unk_token_id and marker != "<|unknown|>"):
        raise ValueError(
            f"tokenizer {getattr(tok, 'name_or_path', tok)!r} has no single id "
            f"for OSRT marker {marker!r}; it does not satisfy the tokenizer contract"
        )
    return int(tid)


def _encode_plain(tok, segment: str) -> list[int]:
    # split_special_tokens=True: SmolLM2's legacy control strings
    # (`<|endoftext|>` etc. — still `special` in tokenizer.json) become
    # ordinary text instead of collapsing to ids 0-16 when they occur in raw
    # web/code text.
    return tok.encode(segment, add_special_tokens=False, split_special_tokens=True)


def encode_with_markers(tok, text: str) -> list[int]:
    """Encode `text` so OSRT markers are single ids and nothing else is special.

    The text is split on the 32 OSRT marker strings; each plain segment is
    BPE-encoded with `split_special_tokens=True` and each marker contributes
    exactly its id. Plain text (no `<|`) is a single encode call and equals
    `tok.encode(text, add_special_tokens=False)`.
    """
    if "<|" not in text:
        return _encode_plain(tok, text)
    ids: list[int] = []
    pos = 0
    for m in _MARKER_RE.finditer(text):
        if m.start() > pos:
            ids.extend(_encode_plain(tok, text[pos:m.start()]))
        ids.append(_marker_id(tok, m.group(0)))
        pos = m.end()
    if pos < len(text):
        ids.extend(_encode_plain(tok, text[pos:]))
    return ids
