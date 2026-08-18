"""Streaming data pipeline for OSRT pre-training.

Handles:
- Progressive seq_len (2048 → 4096 → 8192) across phases
- Multi-dataset weighted sampling within each phase
- Code + text mixing from the start
- Instruction format handling (messages column)
- Resilient streaming: connection drops and corrupt shards are caught
  and retried instead of killing the training run.
- Optional `format` key on a dataset config to force a custom text
  extractor — used by the continued-pretraining ("extend") mix to
  wrap Nemotron post-training rows in the SFT chat schema for
  rehearsal-against-forgetting (see PretrainExtendConfig).
"""

import random
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

# ── Custom text extractors for the extend-stage mix ─────────────────────
# Used only when a dataset config sets `format=<key>`. Default
# (no format key) falls through to TokenStream._extract_text which
# handles plain text / code / chat-messages columns generically.
#
# Each extractor takes a raw HF row dict and returns a string of
# already-formatted text ready for the tokenizer. The tokenizer's
# special-token IDs end up baked in, which is the whole point for
# the SFT-rehearsal extractors.


def _format_nemotron_sft_text(example: dict) -> str:
    """Wrap Nemotron-Post-Training rows in the SFT chat schema.

    Used as a *rehearsal* signal during continued pretraining so the
    model keeps seeing (and predicting) the chat tags it learned in
    SFT — without that, ~1,800 steps of plain-text pretrain would
    erode the structured think/answer behaviour. Pretrain's loss is
    full-token (no masking), so the model is trained to predict every
    token in the formatted string including the chat tags.

    Pulls (question,
    reasoning, answer) from messages + the dedicated `reasoning`
    field, then wraps as
        <|user|>{q}<|assistant|><|think|>{r}<|/think|><|answer|>{a}<|/answer|>
    Returns "" on malformed rows so the stream loop skips them.
    """
    msgs = example.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < 2:
        return ""
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
        return ""
    reasoning = (example.get("reasoning") or "").strip()
    return (
        f"<|user|>{question}<|assistant|>"
        f"<|think|>{reasoning}<|/think|>"
        f"<|answer|>{answer}<|/answer|>"
    )


def _format_stack_code(example: dict) -> str:
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


def _format_arxiv(example: dict) -> str:
    """RedPajama-arxiv rows.

    RedPajama stores the full LaTeX paper body in `text`. Long but
    well-formatted; the tokenizer handles LaTeX as a sequence of
    short BPE pieces.
    """
    text = example.get("text", "")
    if isinstance(text, str) and text.strip():
        return text
    return ""


def _format_nemotron_math_pretrain(example: dict) -> str:
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
# Used by PretrainExtend2Config to inject high-density reasoning data.
# Per the R1 paper, exposure to long-form <think> traces during
# continued pretraining is the most efficient way to teach a small
# model to "think before answering". The HRA delta from prior SFT/GRPO
# stays frozen so the gain accumulates as a base-weight knowledge
# update without disturbing the GRPO-tuned answer format.
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


def _format_openr1_math(example: dict) -> str:
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
    return (
        f"<|user|>{problem}<|assistant|>{pick}"
        f"<|answer|>{answer_str}<|/answer|>"
    )


def _format_openmath_reasoning(example: dict) -> str:
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
    return (
        f"<|user|>{problem}<|assistant|>{solution}"
        f"<|answer|>{answer_str}<|/answer|>"
    )


def _format_openthoughts(example: dict) -> str:
    """open-thoughts/OpenThoughts-114k rows — multi-domain R1 traces.

    Schema: `conversations` (list of {from, value} dicts, alternating
    user/assistant). Takes the first user/assistant pair (one Q+A per
    row in practice) and rewrites the inner thought tags.
    """
    convs = example.get("conversations")
    if not (isinstance(convs, list) and len(convs) >= 2):
        return ""
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
    a = _rewrite_reasoning_tags(a)
    return f"<|user|>{q}<|assistant|>{a}"


def _format_magicoder(example: dict) -> str:
    """ise-uiuc/Magicoder-Evol-Instruct-110K rows — evolved coding tasks.

    Schema: `instruction` + `response`. Wrap as chat so the model
    sees the instruction-following pattern in pretraining context.
    """
    instr = example.get("instruction")
    resp = example.get("response")
    if not (isinstance(instr, str) and instr.strip()):
        return ""
    if not (isinstance(resp, str) and resp.strip()):
        return ""
    return f"<|user|>{instr}<|assistant|>{resp}"


def _format_magicoder_oss(example: dict) -> str:
    """ise-uiuc/Magicoder-OSS-Instruct-75K rows — multi-language OSS code.

    Schema: `problem` + `solution` (plus `lang`, `seed` we ignore).
    """
    prob = example.get("problem")
    sol = example.get("solution")
    if not (isinstance(prob, str) and prob.strip()):
        return ""
    if not (isinstance(sol, str) and sol.strip()):
        return ""
    return f"<|user|>{prob}<|assistant|>{sol}"


def _format_bbh(example: dict) -> str:
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
    return (
        f"<|user|>{inp}<|assistant|>"
        f"<|answer|>{tgt}<|/answer|>"
    )


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
}


class TokenStream(IterableDataset):
    """Streaming token dataset with multi-dataset weighted sampling.

    Streams from multiple HuggingFace datasets simultaneously,
    sampling according to configured weights. Handles both plain text
    and instruction-format (messages column) datasets.

    Args:
        dataset_configs: List of dataset config dicts with hf_id, weight, etc.
        seq_len: Sequence length for this phase.
        tok_name: HuggingFace tokenizer identifier.
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        dataset_configs: list[dict],
        seq_len: int,
        tok_name: str,
        seed: int,
    ) -> None:
        self.dataset_configs = dataset_configs
        self.seq_len = seq_len
        self.tok_name = tok_name
        self.seed = seed

    def __iter__(self):  # noqa: ANN204
        from datasets import load_dataset

        tok = AutoTokenizer.from_pretrained(self.tok_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed if worker_info is None else self.seed + worker_info.id
        rng = random.Random(seed)

        # Build (or rebuild) one stream with bounded retries. Used both
        # for the initial connect AND mid-run reconnect so a transient
        # HF Hub error during stream setup doesn't kill the whole run.
        # The ablate stage hit this exact failure: cell A worked but
        # cell B's load_dataset raised "Cannot send a request, as the
        # client has been closed" before yielding a single batch.
        def _cycling_iter(base_ds, ds_name: str, ds_seed: int):
            """Wrap an UNSHUFFLED streaming dataset in an infinite cycle.

            Small datasets like Magicoder (110K rows), OpenThoughts
            (114K), and BBH (250) exhaust their streaming iterators
            during a multi-thousand-step run, and the debt-based
            sampler then deadlocks: every worker thrashes on the same
            empty-stream reconnect because the unfulfilled deficit
            keeps the sampler picking it. Cycling internally means
            the stream is always ready to yield, so the deficit
            actually drains and the sampler moves on.

            On each cycle we re-shuffle with a new seed so the same
            underlying rows arrive in a different order — better for
            small datasets where repeated exposure to the same
            template is the whole point (e.g. BBH's 250 logic puzzles
            teach reasoning structure, not specific puzzles).

            CRITICAL: we shuffle the UNSHUFFLED `base_ds` fresh each
            cycle — NOT the previous cycle's shuffled view. The old code
            did `ds = ds.shuffle(...)`, wrapping the already-shuffled
            dataset in another ShuffledIterableDataset every cycle. A
            tiny dataset (e.g. nemotron-math-textbooks) cycles many times
            per deficit-fill, nesting wrappers until iteration blew the
            Python recursion limit — "RecursionError: maximum recursion
            depth exceeded" → reconnect → repeat forever, never reaching
            a forward pass. Shuffling the base keeps exactly ONE shuffle
            layer no matter how many cycles. (Caught by the v6 midtrain
            knowledge-mix probe; foundation's mix never streamed these
            small Specialized-v1 configs. See scripts/repro_cycling_recursion.py.)
            """
            cycles = 0
            current_seed = ds_seed
            while True:
                shuffled = base_ds.shuffle(buffer_size=5_000, seed=current_seed)
                for ex in shuffled:
                    yield ex
                cycles += 1
                if cycles == 1:
                    print(
                        f"[DataWorker] {ds_name} stream exhausted — "
                        f"cycling with re-shuffle (small dataset, "
                        f"expected for Magicoder/OpenThoughts/BBH).",
                        flush=True,
                    )
                current_seed += 1

        def _open_stream(stream_idx: int, seed_offset: int = 0):
            ds_cfg = self.dataset_configs[stream_idx]
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
                    ds_seed = seed + seed_offset
                    # Pass the UNSHUFFLED ds — _cycling_iter shuffles the
                    # base fresh each cycle (one shuffle layer, always).
                    # Shuffling here first and passing `shuffled` would make
                    # the first cycle double-shuffled and reintroduce the
                    # nesting that the cycle loop now avoids.
                    # Return an infinite-cycle wrapper so small
                    # streaming datasets (Magicoder, OpenThoughts,
                    # BBH) never trigger the StopIteration → reconnect
                    # path that deadlocks the debt-based sampler.
                    return _cycling_iter(ds, ds_cfg["hf_id"], ds_seed)
                except Exception as exc:
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

        # Load all dataset streams
        streams = []
        weights = []
        for stream_idx, ds_cfg in enumerate(self.dataset_configs):
            print(f"[DataWorker] Connecting to {ds_cfg['hf_id']}...")
            ds = _open_stream(stream_idx)
            streams.append(iter(ds))
            weights.append(ds_cfg["weight"])
            print(f"[DataWorker] Stream ready for {ds_cfg['hf_id']}")

        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        # Token-weighted sampling: pick the stream whose observed token
        # fraction is furthest behind its configured target. This makes
        # a "60% FineWeb / 40% CodeParrot" config produce the actual
        # token mix regardless of per-stream example lengths — otherwise
        # code streams with longer examples would dominate. Starts
        # weight-random during the bootstrap phase (no tokens seen yet).
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

        buffer: list[int] = []

        def _reconnect_stream(stream_idx: int) -> bool:
            # Mid-run reconnect uses the same retry-aware open path as
            # initial setup, plus a randomised seed offset so the
            # reshuffle doesn't replay identical examples. Returns True on
            # success; swallows failures (returns False) so a reconnect
            # that itself errors can't propagate and kill the worker.
            try:
                ds = _open_stream(
                    stream_idx, seed_offset=rng.randint(1, 100_000),
                )
                streams[stream_idx] = iter(ds)
                print(
                    f"[DataWorker] Reconnected to "
                    f"{self.dataset_configs[stream_idx]['hf_id']}",
                    flush=True,
                )
                return True
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[DataWorker] reconnect to "
                    f"{self.dataset_configs[stream_idx]['hf_id']} failed: "
                    f"{type(exc).__name__}: {str(exc)[:120]}",
                    flush=True,
                )
                return False

        max_retries = 8

        def _robust_next(stream_idx: int, ds_name: str):
            """Fetch the next example, surviving ANY failure.

            The previous code only caught StopIteration on the
            post-reconnect retry, so a reconnected stream that immediately
            raised (e.g. httpx 'Cannot send a request, as the client has
            been closed' after a mass HF disconnect) propagated out and
            killed the DataLoader worker — taking the whole multi-hour run
            with it. This catches EVERYTHING (StopIteration, connection
            resets, SSL errors, closed-client RuntimeErrors), reconnects
            with bounded exponential backoff, and returns None if it can't
            recover this attempt so the caller just skips and continues.
            """
            for attempt in range(1, max_retries + 1):
                try:
                    return next(streams[stream_idx])
                except StopIteration:
                    # _cycling_iter should make this unreachable, but if a
                    # stream genuinely ends, rebuild it.
                    _reconnect_stream(stream_idx)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[DataWorker] {ds_name}: {type(exc).__name__}: "
                        f"{str(exc)[:160]} — reconnect [{attempt}/{max_retries}]",
                        flush=True,
                    )
                    time.sleep(min(2 * attempt, 30))
                    _reconnect_stream(stream_idx)
            print(
                f"[DataWorker] {ds_name}: giving up after {max_retries} "
                f"retries, skipping batch",
                flush=True,
            )
            return None

        while True:
            idx = _pick_stream()
            ds_cfg_i = self.dataset_configs[idx]
            ds_name = ds_cfg_i.get("name", ds_cfg_i["hf_id"])

            example = _robust_next(idx, ds_name)
            if example is None:
                continue

            # Extract text from example. Per-stream `format` config
            # (used by extend-stage rehearsal) overrides the generic
            # _extract_text path so we can wrap Nemotron rows in the
            # SFT chat schema and pull non-standard fields cleanly.
            fmt = ds_cfg_i.get("format")
            if fmt:
                fn = FORMAT_FN_PRETRAIN.get(fmt)
                if fn is None:
                    raise ValueError(
                        f"Unknown pretrain format key '{fmt}' on dataset "
                        f"{ds_name}. Valid: {sorted(FORMAT_FN_PRETRAIN)}",
                    )
                text = fn(example)
            else:
                text = self._extract_text(example, tok)
            if not text or not text.strip():
                continue

            tokens = tok.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer.append(tok.eos_token_id)
            # Record token count for the debt-based sampler. We count
            # real tokens but not the structural EOS so code streams
            # with many short examples aren't artificially inflated.
            tokens_seen[idx] += len(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                # Labels are aligned with input_ids — the model shifts
                # internally (model.py:895-897). Yielding chunk[1:] here
                # would double-shift, so position i would be trained to
                # predict token at i+2 instead of i+1. SFT yields aligned
                # labels too, so this matches the rest of the stack.
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                yield input_ids, input_ids.clone()

    def _extract_text(self, example: dict, tok) -> str:
        """Extract text from various dataset formats."""
        # Instruction format (messages column)
        if "messages" in example:
            try:
                return tok.apply_chat_template(example["messages"], tokenize=False)
            except Exception:
                parts: list[str] = []
                for m in example["messages"]:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    parts.append(f"{role}: {content}")
                    if role == "assistant":
                        parts.append(tok.eos_token)
                return "\n".join(parts)

        # Conversations format (OpenHermes, SlimOrca)
        if "conversations" in example:
            parts = []
            for m in example["conversations"]:
                role = m.get("from", m.get("role", "user"))
                value = m.get("value", m.get("content", ""))
                parts.append(f"{role}: {value}")
            return "\n".join(parts)

        # Code format (content column)
        if "content" in example:
            return example["content"]

        # Instruction/output format (Alpaca, Evol-Instruct-Code)
        if "instruction" in example and "output" in example:
            inp = example.get("input", "")
            if inp:
                return f"{example['instruction']}\n{inp}\n{example['output']}"
            return f"{example['instruction']}\n{example['output']}"

        # Plain text
        if "text" in example:
            return example["text"]

        return ""


def make_loader(
    dataset_configs: list[dict],
    seq_len: int,
    tokenizer_name: str,
    batch_size: int,
    step_num: int,
    num_workers: int = 4,
    prefetch_factor: int = 4,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a streaming DataLoader for a training phase.

    Args:
        dataset_configs: List of dataset config dicts for this phase.
        seq_len: Sequence length for this phase.
        tokenizer_name: HuggingFace tokenizer identifier.
        batch_size: Micro-batch size.
        step_num: Current step (used to vary shuffle seed).

    Returns:
        DataLoader yielding (input_ids, labels) batches.
    """
    ds = TokenStream(
        dataset_configs, seq_len, tokenizer_name, seed=42 + step_num
    )
    # num_workers=2 offloads HF streaming + BPE tokenisation to two
    # background processes so the main training thread doesn't wait on
    # them. Each worker gets its own seed (TokenStream reads
    # worker_info.id at line 55 and offsets) so they don't produce
    # duplicate batches. persistent_workers keeps the processes across
    # phase transitions — new loaders still spawn fresh workers, but
    # within a phase we don't tear them down for every step's reload.
    # prefetch_factor=2 keeps a small queue of ready batches per worker.
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


# ═════════════════════════════════════════════════════════════════════
# MOPD ROLLOUT DATASET — teacher-distillation from local JSONL
# ═════════════════════════════════════════════════════════════════════
# Reads a JSONL file of Gemini rollouts (one record per line with prompt /
# thinking / response fields), tokenises into the osrt chat template,
# and yields (input_ids, labels) where labels mask the user prompt with
# -100 so the model is trained ONLY on producing the assistant response
# (thinking + answer), not on repeating the user message.
