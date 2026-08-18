"""SFT data pipeline for OSRT with native token tags.

Chat format:
    <|user|>{prompt}<|assistant|><|think|>{reasoning}<|/think|><|answer|>{answer}<|/answer|><|end_of_text|>

Loss masking:
    - Everything from <|user|> to <|assistant|> (inclusive): IGNORE_INDEX
    - <|think|> through <|/answer|>: trained on (real labels)

All structural tags are single tokens in the v4 tokenizer.
"""

import random
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerFast

from osrt.system_prompts import sample_system_prompt

IGNORE_INDEX = -100


# ── Format functions ────────────────────────────────────────────────────
# Each returns (question, reasoning, answer) triple.
# The SFTStream handles wrapping with tags.


def format_gsm8k(example: dict) -> tuple[str, str, str]:
    """GSM8K: question + step-by-step reasoning + final numeric answer."""
    question = example["question"]
    answer_raw = example["answer"]

    if "####" in answer_raw:
        reasoning, final = answer_raw.rsplit("####", 1)
        return question, reasoning.strip(), final.strip()
    return question, answer_raw, answer_raw


def format_numina_math(example: dict) -> tuple[str, str, str]:
    """NuminaMath-CoT: problem + solution with reasoning."""
    import re
    question = example.get("problem", "")
    solution = example.get("solution", "")

    if not question or not solution:
        return "", "", ""

    lines = solution.strip().split("\n")
    if len(lines) > 1:
        reasoning = "\n".join(lines[:-1]).strip()
        final = lines[-1].strip()
        match = re.search(r"\\boxed\{(.+?)\}", final)
        if match:
            final = match.group(1)
    else:
        reasoning = solution.strip()
        final = solution.strip()

    return question, reasoning, final


def format_orca_math(example: dict) -> tuple[str, str, str]:
    """Orca-Math: question + worked solution."""
    question = example["question"]
    answer = example["answer"]

    lines = answer.strip().split("\n")
    if len(lines) > 1:
        reasoning = "\n".join(lines[:-1]).strip()
        final = lines[-1].strip()
    else:
        reasoning = answer.strip()
        final = answer.strip()

    return question, reasoning, final


def format_math_instruct(example: dict) -> tuple[str, str, str]:
    """MathInstruct: instruction + output."""
    question = example.get("instruction", "")
    output = example.get("output", "")

    if not question or not output:
        return "", "", ""

    lines = output.strip().split("\n")
    if len(lines) > 1:
        reasoning = "\n".join(lines[:-1]).strip()
        final = lines[-1].strip()
    else:
        reasoning = output.strip()
        final = output.strip()

    return question, reasoning, final


def format_evol_code(example: dict) -> tuple[str, str, str]:
    """Evol-Instruct-Code: instruction + code output."""
    question = example.get("instruction", "")
    output = example.get("output", "")

    if not question or not output:
        return "", "", ""

    parts = output.split("```")
    if len(parts) >= 3:
        # Leave empty when no natural prose precedes the code block —
        # fabricating "Let me write the code for this." trained v4 to
        # emit CoT filler on trivial prompts. Empty think is fine: the
        # model learns "no reasoning needed" for this example class.
        reasoning = parts[0].strip()
        code = "```" + "```".join(parts[1:])
        return question, reasoning, code

    paragraphs = output.strip().split("\n\n")
    if len(paragraphs) > 1:
        return question, paragraphs[0].strip(), "\n\n".join(paragraphs[1:]).strip()

    return question, "", output


def format_alpaca_code(example: dict) -> tuple[str, str, str]:
    """Alpaca-style code: instruction + optional input + output."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if not instruction or not output:
        return "", "", ""

    question = f"{instruction}\n{inp}".strip() if inp else instruction

    parts = output.split("```")
    if len(parts) >= 3:
        reasoning = parts[0].strip()
        code = "```" + "```".join(parts[1:])
        return question, reasoning, code

    lines = output.strip().split("\n")
    code_indicators = ("def ", "class ", "import ", "from ", "#", "if __name__")
    if lines and any(lines[0].strip().startswith(ind) for ind in code_indicators):
        return question, "", output

    if len(lines) > 3:
        return question, lines[0].strip(), "\n".join(lines[1:]).strip()

    return question, "", output


def format_alpaca(example: dict) -> tuple[str, str, str]:
    """Alpaca: instruction + optional input + output."""
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]

    question = f"{instruction}\n{inp}".strip() if inp else instruction

    sentences = output.strip().split(". ")
    if len(sentences) > 2:
        reasoning = ". ".join(sentences[:-1]).strip() + "."
        final = sentences[-1].strip()
        if not final.endswith("."):
            final += "."
        return question, reasoning, final

    return question, "", output


def format_openhermes(example: dict) -> tuple[str, str, str]:
    """OpenHermes: conversations list."""
    conversations = example.get("conversations", [])
    if len(conversations) < 2:
        return "", "", ""

    question = ""
    answer = ""
    for msg in conversations:
        role = msg.get("from", "")
        value = msg.get("value", "")
        if role == "human" and not question:
            question = value
        elif role == "gpt" and question and not answer:
            answer = value

    if not question or not answer:
        return "", "", ""

    paragraphs = answer.strip().split("\n\n")
    if len(paragraphs) > 1:
        reasoning = "\n\n".join(paragraphs[:-1]).strip()
        final = paragraphs[-1].strip()
        return question, reasoning, final

    return question, "", answer


def format_tulu(example: dict) -> tuple[str, str, str]:
    """Tülu-3 SFT mixture: `messages` chat-list schema.

    Schema: [{"role": "user"/"assistant"/"system", "content": str}, ...].
    SFT-v1 uses SINGLE-TURN only: take the first user message as the question
    and the following assistant message as the answer. Multi-turn rows (>1 user
    turn) are skipped (return empties → SFTStream drops them). reasoning_mode is
    'off' for tulu, so reasoning is left empty (the answer carries the content).
    """
    messages = example.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        return "", "", ""
    # count user turns; v1 keeps single-turn only
    user_turns = [m for m in messages if m.get("role") == "user"]
    if len(user_turns) != 1:
        return "", "", ""
    question = ""
    answer = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "user" and not question:
            question = content
        elif role == "assistant" and question and not answer:
            answer = content
    if not question or not answer:
        return "", "", ""
    return question, "", answer


def format_ifeval(example: dict) -> tuple[str, str, str]:
    """IFEval-like: prompt + response with constraints."""
    question = example.get("prompt", "")
    response = example.get("response", "")

    if not question or not response:
        return "", "", ""

    paragraphs = response.strip().split("\n\n")
    if len(paragraphs) > 1:
        reasoning = "\n\n".join(paragraphs[:-1]).strip()
        final = paragraphs[-1].strip()
        return question, reasoning, final

    return question, "", response


def format_longform(example: dict) -> tuple[str, str, str]:
    """LongForm: input + long output."""
    question = example["input"]
    output = example["output"]

    paragraphs = output.strip().split("\n\n")
    if len(paragraphs) > 1:
        reasoning = "\n\n".join(paragraphs[:-1]).strip()
        final = paragraphs[-1].strip()
        return question, reasoning, final

    return question, output.strip(), output.strip()


def format_nemotron(example: dict) -> tuple[str, str, str]:
    """Nemotron-Post-Training-Dataset-v1 (math, stem, code splits).

    Schema fields used:
      - messages: list of {role, content, tool_calls} — we take the
        first user message as `question` and the first assistant
        message as `answer`.
      - reasoning: top-level string field with the chain-of-thought.
        Maps directly to our `<|think|>{reasoning}<|/think|>` block —
        cleaner than the heuristic split-on-paragraph approach used by
        the other format functions.

    For tool_calling examples (where the assistant message has a
    tool_calls field), use `format_nemotron_tool_calling` instead.
    """
    msgs = example.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < 2:
        return "", "", ""

    question = ""
    answer = ""
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "user" and not question:
            question = content
        elif role == "assistant" and question and not answer:
            answer = content
            break

    if not question or not answer:
        return "", "", ""

    reasoning = (example.get("reasoning") or "").strip()
    return question, reasoning, answer


def format_nemotron_tool_calling(example: dict) -> tuple[str, str, str]:
    """Nemotron tool_calling split.

    Uses Hermes-style plain-text <tool_call>...</tool_call> tags inside
    the answer block to encode tool invocations. Avoids tokenizer
    surgery (no new special tokens) — the model just learns to emit a
    recognisable substring pattern that downstream serving code parses
    with a regex.

    Schema:
      - messages[i].tool_calls is a list of
        {id, type, function: {name, arguments}}
      - When tool_calls is non-empty, the assistant turn is "I'm going
        to call these tools" rather than a final answer.
      - The answer block here is the tool-call invocation only;
        post-tool-result continuation is a separate (multi-turn) example
        we don't try to capture in single-turn SFT.
    """
    import json

    msgs = example.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < 2:
        return "", "", ""

    question = ""
    assistant_content = ""
    tool_calls: list[dict] = []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "user" and not question:
            question = content
        elif (role == "assistant" and question
              and not assistant_content and not tool_calls):
            assistant_content = content
            tcs = m.get("tool_calls") or []
            if isinstance(tcs, list):
                tool_calls = tcs
            break

    if not question:
        return "", "", ""
    if not assistant_content and not tool_calls:
        return "", "", ""

    # Build the answer body. Plain-text <tool_call> tags carry the
    # function name + arguments in JSON form. If the assistant also
    # produced text alongside the tool calls, include both.
    parts: list[str] = []
    if assistant_content:
        parts.append(assistant_content)
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args_raw = fn.get("arguments")
        # `arguments` is sometimes already a JSON string, sometimes a
        # dict — normalise to a string we can drop into the tag body.
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": args_raw}
        else:
            args = args_raw or {}
        if not name:
            continue
        payload = json.dumps({"name": name, "arguments": args})
        parts.append(f"<tool_call>{payload}</tool_call>")
    answer = "\n".join(parts).strip()

    if not answer:
        return "", "", ""

    reasoning = (example.get("reasoning") or "").strip()
    return question, reasoning, answer


FORMAT_FN = {
    "gsm8k": format_gsm8k,
    "numina_math": format_numina_math,
    "orca_math": format_orca_math,
    "math_instruct": format_math_instruct,
    "evol_code": format_evol_code,
    "alpaca_code": format_alpaca_code,
    "alpaca": format_alpaca,
    "openhermes": format_openhermes,
    "tulu": format_tulu,
    "ifeval": format_ifeval,
    "longform": format_longform,
    "nemotron": format_nemotron,
    "nemotron_tool_calling": format_nemotron_tool_calling,
}


# ── SFT Stream ──────────────────────────────────────────────────────────


class SFTStream(IterableDataset):
    """Streaming SFT dataset with native token tags and loss masking.

    Formats each example as:
        <|user|>{question}<|assistant|><|think|>{reasoning}<|/think|><|answer|>{answer}<|/answer|><|end_of_text|>

    Loss masking:
        - <|user|> through <|assistant|> (inclusive): IGNORE_INDEX
        - Everything after <|assistant|>: real labels (think + answer)
    """

    def __init__(
        self,
        dataset_configs: list[dict],
        seq_len: int,
        tokenizer: PreTrainedTokenizerFast,
        seed: int,
        user_tag: str = "<|user|>",
        assistant_tag: str = "<|assistant|>",
        think_open: str = "<|think|>",
        think_close: str = "<|/think|>",
        answer_open: str = "<|answer|>",
        answer_close: str = "<|/answer|>",
        system_tag: str | None = None,
        min_response_tokens: int = 0,
    ) -> None:
        self.dataset_configs = dataset_configs
        self.seq_len = seq_len
        self.tok = tokenizer
        self.seed = seed
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.think_open = think_open
        self.think_close = think_close
        self.answer_open = answer_open
        self.answer_close = answer_close
        # When system_tag is set, each example is prefixed with a
        # <|system|>{persona} turn, the persona sampled per-example from the
        # reasoning-on/off pool selected by the dataset's `reasoning_mode`.
        # None → legacy behaviour (no system turn), so existing callers are
        # unaffected.
        self.system_tag = system_tag
        self.min_response_tokens = min_response_tokens

    def __iter__(self):  # noqa: ANN204
        from datasets import load_dataset

        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed if worker_info is None else self.seed + worker_info.id
        rng = random.Random(seed)

        # Mirror of data.py::_open_stream — bounded-retry stream open
        # used both for initial connect and mid-run reconnect. Without
        # this, a transient HF Hub error during stream setup
        # ("Cannot send a request, as the client has been closed")
        # kills the whole SFT run. Same exact failure that hit pretrain
        # before we patched data.py; sft_data.py needed the same fix.
        def _open_stream(stream_idx: int, seed_offset: int = 0):
            ds_cfg = self.dataset_configs[stream_idx]
            load_kwargs = {
                "split": ds_cfg.get("split", "train"),
                "streaming": True,
            }
            if ds_cfg.get("hf_config"):
                load_kwargs["name"] = ds_cfg["hf_config"]
            last_exc = None
            for attempt in range(1, 6):
                try:
                    ds = load_dataset(ds_cfg["hf_id"], **load_kwargs)
                    return ds.shuffle(
                        buffer_size=2_000, seed=seed + seed_offset,
                    )
                except Exception as exc:
                    last_exc = exc
                    print(
                        f"[SFTWorker] {ds_cfg['hf_id']} setup attempt "
                        f"{attempt}/5 failed: {type(exc).__name__}: "
                        f"{str(exc)[:120]} — retrying...",
                        flush=True,
                    )
                    time.sleep(2 * attempt)
            raise RuntimeError(
                f"Failed to open SFT stream for {ds_cfg['hf_id']} after "
                f"5 attempts: {last_exc}",
            )

        # Load all dataset streams
        streams = []
        weights = []
        for stream_idx, ds_cfg in enumerate(self.dataset_configs):
            print(f"[SFTWorker] Connecting to {ds_cfg['hf_id']}...")
            ds = _open_stream(stream_idx)
            streams.append(iter(ds))
            weights.append(ds_cfg["weight"])
            print(f"[SFTWorker] Stream ready for {ds_cfg['hf_id']}")

        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        format_fns = [FORMAT_FN[cfg["format"]] for cfg in self.dataset_configs]

        # Token-weighted sampling (same debt-based scheme as pretrain
        # data.py). SFT response lengths vary by 5-10x across datasets
        # — NuminaMath-CoT rationales are long, Alpaca answers are
        # short — so pure example-weighted sampling silently drifts
        # the trained-token mix away from the configured weights.
        # Counts only response tokens (prompt is IGNORE_INDEX-masked
        # and contributes zero loss signal), so this balances what the
        # model actually trains on.
        tokens_seen: list[int] = [0] * len(streams)

        def _pick_stream() -> int:
            total = sum(tokens_seen)
            if total == 0:
                return rng.choices(range(len(streams)), weights=weights, k=1)[0]
            deficits = [
                weights[i] - tokens_seen[i] / total
                for i in range(len(streams))
            ]
            max_def = max(deficits)
            candidates = [
                i for i, d in enumerate(deficits) if d >= max_def - 1e-6
            ]
            return rng.choice(candidates)

        # Packing buffer: accumulate multiple examples per sequence
        pack_ids: list[int] = []
        pack_labels: list[int] = []

        max_retries = 5

        while True:
            idx = _pick_stream()
            ds_cfg_i = self.dataset_configs[idx]
            ds_name = ds_cfg_i.get("name", ds_cfg_i["hf_id"])

            try:
                example = next(streams[idx])
            except StopIteration:
                # Mid-run reconnect uses the same retry-aware open path.
                # Catch ALL exceptions from the post-reconnect next() —
                # it's a fresh HF stream that may itself hit httpx
                # errors. sft_math run 1 died at step ~625 because
                # the inner next() raised RuntimeError ("Cannot send
                # a request, as the client has been closed") which
                # only the OUTER except Exception was meant to catch.
                # Falling through to `continue` skips the bad sample;
                # the next loop iteration will pick a stream and try
                # again, hitting the outer retry logic if the failure
                # repeats.
                try:
                    ds = _open_stream(
                        idx, seed_offset=rng.randint(1, 10000),
                    )
                    streams[idx] = iter(ds)
                    example = next(streams[idx])
                except StopIteration:
                    continue
                except Exception as inner_exc:
                    print(
                        f"[SFTWorker] {ds_name}: post-reconnect "
                        f"{type(inner_exc).__name__}: "
                        f"{str(inner_exc)[:120]} — skipping example",
                        flush=True,
                    )
                    continue
            except Exception as exc:
                # Mid-iteration failures: HF httpx "client has been
                # closed", connection drops, corrupt parquet shards,
                # rate-limit responses. Mirrors data.py::TokenStream's
                # retry block (this file used to only catch
                # StopIteration, which let the run die on the first
                # transient HTTP error — caught during sft_refresh
                # run 3 on codhe-hugging-mcp where the workspace
                # top-up triggered an httpx client reset mid-run).
                example = None
                for attempt in range(1, max_retries + 1):
                    print(
                        f"[SFTWorker] {ds_name}: {type(exc).__name__}: "
                        f"{str(exc)[:120]} — reconnecting "
                        f"[{attempt}/{max_retries}]",
                        flush=True,
                    )
                    time.sleep(2 * attempt)
                    try:
                        ds = _open_stream(
                            idx, seed_offset=rng.randint(1, 100_000),
                        )
                        streams[idx] = iter(ds)
                        example = next(streams[idx])
                        print(
                            f"[SFTWorker] {ds_name}: reconnected.",
                            flush=True,
                        )
                        break
                    except Exception as retry_exc:
                        exc = retry_exc
                if example is None:
                    print(
                        f"[SFTWorker] {ds_name}: giving up after "
                        f"{max_retries} retries, skipping example",
                        flush=True,
                    )
                    continue

            try:
                question, reasoning, answer = format_fns[idx](example)
            except (KeyError, TypeError):
                continue

            if not question or not answer:
                continue

            # Build token sequence with native tags.
            # Prompt (masked): [<|system|>{persona}]<|user|>{question}<|assistant|>
            # Response (trained):
            #   <|think|>{reasoning}<|/think|>
            #   <|answer|>{answer}<|/answer|><|end_of_text|>
            # Optional <|system|> turn: persona sampled per-example from the
            # reasoning-on/off pool chosen by this dataset's reasoning_mode.
            # It joins the MASKED prefix, so the model learns to FOLLOW the
            # system prompt but is never trained to echo it.
            system_prefix = ""
            if self.system_tag is not None:
                mode = self.dataset_configs[idx].get("reasoning_mode", "on")
                _, persona = sample_system_prompt(rng, mode)
                system_prefix = f"{self.system_tag}{persona}"
            prompt_text = (
                f"{system_prefix}{self.user_tag}{question}{self.assistant_tag}"
            )
            response_text = (
                f"{self.think_open}{reasoning}{self.think_close}"
                f"{self.answer_open}{answer}{self.answer_close}"
                f"{self.tok.eos_token}"
            )

            prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)
            response_ids = self.tok.encode(response_text, add_special_tokens=False)

            # "Not too short": skip examples whose trained response is below the
            # floor (response_ids includes the format tags + eos, so the floor
            # is on the full response block).
            if len(response_ids) < self.min_response_tokens:
                continue

            ex_ids = prompt_ids + response_ids
            ex_labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids

            # Skip examples longer than seq_len or with no room for response
            if len(ex_ids) > self.seq_len or len(prompt_ids) >= len(ex_ids) - 5:
                continue

            # If this example won't fit, flush the buffer first
            if pack_ids and len(pack_ids) + len(ex_ids) > self.seq_len:
                pad_len = self.seq_len - len(pack_ids)
                yield (
                    torch.tensor(
                        pack_ids + [self.tok.pad_token_id] * pad_len,
                        dtype=torch.long,
                    ),
                    torch.tensor(
                        pack_labels + [IGNORE_INDEX] * pad_len,
                        dtype=torch.long,
                    ),
                )
                pack_ids = []
                pack_labels = []

            # Append example to buffer and record its trained-token
            # contribution for the debt-based sampler.
            pack_ids.extend(ex_ids)
            pack_labels.extend(ex_labels)
            tokens_seen[idx] += len(response_ids)


def make_sft_loader(
    dataset_configs: list[dict],
    seq_len: int,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    seed: int,
    user_tag: str = "<|user|>",
    assistant_tag: str = "<|assistant|>",
    think_open: str = "<|think|>",
    think_close: str = "<|/think|>",
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
    system_tag: str | None = None,
    min_response_tokens: int = 0,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a streaming DataLoader for SFT.

    Args:
        dataset_configs: List of dataset config dicts.
        seq_len: Sequence length.
        tokenizer: The v4 tokenizer.
        batch_size: Micro-batch size.
        seed: Random seed.
        user_tag: User turn tag.
        assistant_tag: Assistant turn tag.
        think_open: Think open tag.
        think_close: Think close tag.
        answer_open: Answer open tag.
        answer_close: Answer close tag.

    Returns:
        DataLoader yielding (input_ids, labels) batches.
    """
    ds = SFTStream(
        dataset_configs, seq_len, tokenizer, seed,
        user_tag=user_tag,
        assistant_tag=assistant_tag,
        think_open=think_open,
        think_close=think_close,
        answer_open=answer_open,
        answer_close=answer_close,
        system_tag=system_tag,
        min_response_tokens=min_response_tokens,
    )
    # Single-threaded loader. We tried num_workers=4 + persistent_workers
    # + prefetch_factor=4 for SFT-ultralong launch on 2026-05-08 — workers
    # silently deadlocked at the very first __iter__() call, never even
    # printing the "[SFTWorker] Connecting to..." line. 38 min of
    # GPU-idle wait before manual kill, ~$2.50 of budget burned.
    #
    # Failure surface: 4 workers × 7 HF dataset streams × spawn-context
    # re-import = 28 simultaneous load_dataset() calls during worker
    # bootstrap. TOKENIZERS_PARALLELISM=false (set in the Modal image)
    # prevents the tokenizers-rs mutex deadlock, but the HF datasets
    # streaming auth + first-shard fetch per worker has its own race
    # surface that we have not isolated. Until that's diagnosed,
    # num_workers=0 is the only known-safe configuration.
    #
    # Throughput cost: ~30-40 % vs the prefetched setup. SFT-long ran
    # successfully at num_workers=0 (20-23 sec/step on the fast
    # windows), so this is not a regression — it is the previous
    # working configuration.
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
