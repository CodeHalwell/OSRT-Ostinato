"""lm-evaluation-harness wrapper for OSRT v7 (ported from the v6 wrapper).

Implements the `lm_eval.api.model.LM` interface so the standard
benchmark harness (gsm8k, IFEval, MMLU, etc.) can score our custom
OSRTForCausalLM. Two methods cover the bulk of HF-style benchmarks:

  loglikelihood(requests)  — multiple-choice scoring
                              (MMLU, ARC, hellaswag, BoolQ, ...)
  generate_until(requests) — open-ended generation with stop strings
                              (gsm8k, IFEval, HumanEval, ...)

`loglikelihood_rolling` (rolling perplexity) is not implemented — none
of our target benchmarks use it. If that changes, plumb it like
`loglikelihood` but with sliding-window context.

v7 lane: the model is built from the OSRT_V7 preset exactly as the trunk
built it (vocab padded to a multiple of 128, HRA off), and the checkpoint is
loaded STRICTLY. `base_model=True` (the default) scores the pretrained trunk:
no chat wrapping, greedy decoding, no answer-block extraction. Pass
`base_model=False` for an SFT checkpoint to get the v6 chat-schema behaviour.

Usage (scripts/lm_eval_trunk.py):

    wrapper = OSRTLMEval(ckpt_path="/vol/trunk/osrt_final.pt",
                         tokenizer_path="/root/tokenizer", batch_size=8)
    results = lm_eval.simple_evaluate(model=wrapper, tasks=["hellaswag", "gsm8k"])
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance


class OSRTLMEval(LM):
    """LM-eval-harness adapter over OSRTForCausalLM.

    Loads the model + tokenizer once at construction. Subsequent
    .loglikelihood / .generate_until calls run against the same
    in-memory model — important for benchmark suites that issue
    thousands of requests per task.

    Batching strategy:
      - loglikelihood: right-pad each batch of requests to the max
        sequence length, run a single forward, gather per-token
        log-probs and ignore padded positions. Default batch=8 fits
        comfortably on a single H100.
      - generate_until: batched at request level — each request gets
        its own .generate call. Generation is autoregressive and
        sequence lengths diverge fast on stop-string termination, so
        true batching gains very little here. KV cache inside
        .generate keeps per-step cost O(1).

    Numerical correctness:
      - `loglikelihood` returns log-probs in fp32 (after up-casting
        from bf16 logits). The bf16 → fp32 cast happens inside
        F.log_softmax to preserve precision when computing sums over
        long continuations.
      - Logits are clipped to `real_vocab_size` (32768) so probability
        mass on padding/unused vocab tail (when `vocab_size` was
        rounded up for tensor-core alignment) doesn't leak in. v5 has
        vocab_size == real_vocab_size already so the clip is a no-op,
        but the guard stays for portability.
    """

    # Targeted at the only failure mode the model+wrapper combination
    # has shown that's NOT pure math/IF capability: tool-call
    # hallucination bleeding in from Nemotron-tool_calling SFT
    # examples. Smoke L5b confirmed this is the dominant IFEval
    # failure — 2 of 3 IFEval prompts derail into invented function
    # names like "itinerary_for_a_play" or "get_full_salary_and_
    # placeholder" instead of producing the requested response.
    #
    # CRITICAL: do NOT include literal <|think|>, <|answer|>, etc. in
    # this prompt. An earlier version did, and the resulting tokens
    # (IDs 7-10) in the prompt got suppressed by repetition_penalty
    # during generation, breaking the trained think/answer structure
    # entirely. Use only natural language references.
    # v7 chat mode: no system turn by default — the SFT backbone rows mostly
    # carry none, and render_chat omits the marker when the prompt is empty.
    DEFAULT_SYSTEM_PROMPT = ""

    def __init__(
        self,
        ckpt_path: str,
        tokenizer_path: str,
        batch_size: int = 8,
        max_length: int = 8192,
        device: str = "cuda",
        base_model: bool = True,
        chat_format_generate: bool | None = None,
        chat_format_loglikelihood: bool = False,
        system_prompt: str | None = None,
        default_temperature: float | None = None,
        max_temperature: float = 1.0,
        default_top_p: float = 0.9,
        default_top_k: int = 50,
        default_repetition_penalty: float | None = None,
        extract_answer_block: bool | None = None,
        max_gen_toks: int = 256,
        gen_prefix: str = "",
        legacy_gates: bool = True,
    ) -> None:
        """
        Eval-time prompt + sampling controls
        ────────────────────────────────────
        chat_format_generate (default True): wraps generate_until contexts
            in the model's trained chat schema:
              <|system|>{system_prompt}<|user|>{context}<|assistant|>
            The model was SFT'd to emit <|think|>...<|/think|><|answer|>
            ...<|/answer|> after <|assistant|>. Helpful for gsm8k / IFEval
            / open-ended generation.

        chat_format_loglikelihood (default False): wraps loglikelihood
            contexts the same way. Disabled by default because empirical
            smoke runs showed it breaks MMLU scoring — the model is asked
            to score " A"/" B"/" C"/" D" continuations after <|assistant|>
            where it expects <|think|>, producing essentially-random
            (~25 %) results. Raw context for loglikelihood lets the
            model's pretrained next-token distribution rank multiple-choice
            options in the expected way.

        Sampling defaults are tuned for an undertrained 363M MoE that
        collapses into repetition loops at low temperature. Local probes
        (probe3.py) showed:
          - temp 0.2  → degenerate loops ("17*23 = 17*(2*23) = 17*(2*23)..."
                        for 30+ iterations, never closes <|/think|>)
          - temp 0.7  + repetition_penalty 1.0 → enters bizarre nested
                        states (<|/think|><|answer|><think>...)
          - temp 0.7, top_p 0.9, top_k 50, rep_penalty 1.2 → coherent
                        attempts that close all three tags in <300 tokens
                        with structurally-correct (if mathematically wrong)
                        reasoning.
        Future better-trained checkpoints should drop temperature and
        repetition_penalty back toward 0.2 / 1.0 as they stop repeating.

        max_temperature caps benchmark-passed temperatures at 1.0 to
        avoid pure-noise sampling.

        extract_answer_block (default True): post-processes generations
            to return only the contents of the first <|answer|>...
            <|/answer|> block. Without this, lm-eval's gsm8k extractor
            sees "<|think|>... long reasoning ... <|/think|><|answer|>42"
            and fails to find the numeric answer because its regex
            patterns target "#### 42", "\\boxed{42}", or "the answer is 42"
            — none of which are present. Returning just "42" lets the
            standard extractor work.
        """
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._max_length = max_length
        self._max_gen_toks = int(max_gen_toks)
        # Prepended to every generate_until context (e.g. "```python\n" so a
        # chat-annealed base continues code instead of closing a fence).
        self._gen_prefix = gen_prefix
        # base_model=True: raw prompts, greedy, no answer extraction (a
        # pretrained trunk has no chat schema to put it "in distribution").
        # base_model=False: the v6 SFT defaults (chat wrap, temp 0.7, rp 1.2).
        self._chat_format_generate = (
            (not base_model) if chat_format_generate is None else chat_format_generate
        )
        self._chat_format_loglikelihood = chat_format_loglikelihood
        self._system_prompt = (
            system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT
        )
        self._default_temperature = float(
            (0.0 if base_model else 0.7)
            if default_temperature is None else default_temperature
        )
        self._max_temperature = float(max_temperature)
        self._default_top_p = float(default_top_p)
        self._default_top_k = int(default_top_k)
        self._default_repetition_penalty = float(
            (1.0 if base_model else 1.2)
            if default_repetition_penalty is None else default_repetition_penalty
        )
        self._extract_answer_block = (
            (not base_model) if extract_answer_block is None else extract_answer_block
        )

        # Imports here to avoid loading torch dependencies at module
        # import time when this file might be parsed but not used.
        from transformers import AutoTokenizer

        from osrt.model import OSRTForCausalLM
        from osrt.presets import OSRT_V7, build_config
        from osrt.train import load_model_state_or_raise

        print(f"[lm_eval] Loading tokenizer from {tokenizer_path}", flush=True)
        self._tok = AutoTokenizer.from_pretrained(tokenizer_path)
        real = len(self._tok)
        if real != OSRT_V7["real_vocab_size"]:
            raise ValueError(
                f"tokenizer has {real} tokens, preset expects "
                f"{OSRT_V7['real_vocab_size']}"
            )
        padded = ((real + 127) // 128) * 128  # exactly as app.py::trunk built it
        # legacy_gates=True reproduces the trunk's routing function: gates from
        # the bias-adjusted selection distribution (`router_bias_in_gates`),
        # which is how every checkpoint written before 2026-09-30 was trained.
        # Verified against the pinned implementation by
        # scripts/check_trunk_loader.py.
        cfg = build_config(
            vocab_size=padded, real_vocab_size=real,
            bos_token_id=self._tok.bos_token_id,
            eos_token_id=self._tok.eos_token_id,
            pad_token_id=self._tok.pad_token_id,
            fused_cross_entropy_chunks=8,
            router_bias_in_gates=legacy_gates,
        )
        self._cfg = cfg
        self.resolved_config = {"legacy_gates": legacy_gates, "ckpt_path": ckpt_path,
                                "tokenizer_path": tokenizer_path,
                                "base_model": base_model, "max_length": max_length,
                                "max_gen_toks": max_gen_toks, "gen_prefix": gen_prefix,
                                "batch_size": batch_size}
        print("[lm_eval] Constructing OSRT_V7 model...", flush=True)
        model = OSRTForCausalLM(cfg).to(self._device)

        print(f"[lm_eval] Loading weights from {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # STRICT load — a key/shape mismatch must FAIL loudly, not silently
        # produce a garbage eval.
        load_model_state_or_raise(model, state_dict, context=f"lm_eval {ckpt_path}")
        del ckpt, state_dict

        model.train(False)  # disables MoE capacity drops, enables KV-cache path
        self._model = model
        print("[lm_eval] Model ready.", flush=True)

    # ── Required LM interface ──────────────────────────────────────

    @property
    def eot_token_id(self) -> int:
        return int(self._tok.eos_token_id)

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def world_size(self) -> int:
        return 1

    @property
    def rank(self) -> int:
        return 0

    def tok_encode(self, string: str, **kwargs) -> list[int]:  # noqa: ANN003
        return self._tok.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, **kwargs) -> str:  # noqa: ANN001, ANN003
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._tok.decode(tokens, skip_special_tokens=False)

    # ── Chat-format helper ─────────────────────────────────────────

    def _wrap_context(self, context: str, *, for_generate: bool) -> str:
        """Wrap raw lm-eval context in the SFT-trained chat schema.

        Schema (matches sft_data.py SFTStream output):
            <|system|>{system_prompt}<|user|>{context}<|assistant|>

        Two independent gates:
          - generate_until paths use chat_format_generate (default True
            — gsm8k / IFEval need it to put the model in distribution)
          - loglikelihood paths use chat_format_loglikelihood (default
            False — MMLU scoring is broken by chat wrap, see __init__
            docstring)
        """
        gate = (
            self._chat_format_generate
            if for_generate
            else self._chat_format_loglikelihood
        )
        if not gate:
            return context
        prefix = ""
        if self._system_prompt:
            prefix = f"<|system|>{self._system_prompt}"
        return f"{prefix}<|user|>{context}<|assistant|>"

    def _extract_answer(self, text: str) -> str:
        """Chat-mode post-processing for the v7 contract (no think/answer tags).

        * cut at `<|end_turn|>` (the trained end of an assistant turn);
        * a `\\boxed{X}` answer (OpenMathInstruct-2 style) gets "\n#### X"
          appended so lm-eval's gsm8k strict-match filter finds it, while the
          flexible filter still sees the last number;
        * a reply that is a fenced code block (HumanEval under chat) returns
          the body of the first ```python / ``` fence, so the harness appends
          runnable code to the prompt instead of markdown.
        Off when extract_answer_block=False (base-model mode).
        """
        if not self._extract_answer_block:
            return text
        cut = text.find("<|end_turn|>")
        if cut != -1:
            text = text[:cut]
        text = text.strip()
        fence = re.search(
            r"```(?:python|py|python3)?[ \t]*\n(.*?)(?:```|$)", text, re.S)
        if fence is not None and text.lstrip().startswith("```"):
            return fence.group(1).rstrip() + "\n"
        boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
        if boxed:
            ans = boxed[-1].strip().replace(",", "").replace("$", "")
            return f"{text}\n#### {ans}"
        return text

    def _encode_pair(
        self, context: str, continuation: str,
    ) -> tuple[list[int], list[int]]:
        """Tokenise (context, continuation) the way the upstream harness does.

        Encoding the two strings separately can produce different ids from
        encoding them jointly at the boundary (a trailing space on the context
        is the classic case: "He said: " + "hello" tokenises differently from
        "He said: hello"). The harness moves trailing whitespace onto the
        continuation, encodes the whole string once and splits at the
        context's token length, so the scored ids are exactly the ids the
        model would see in running text.
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole = self.tok_encode(context + continuation)
        ctx = self.tok_encode(context)
        if whole[: len(ctx)] == ctx:
            return ctx, whole[len(ctx):]
        # The boundary merged into the context's last token; the harness
        # accepts the same imprecision here. Fall back to separate encoding.
        return ctx, self.tok_encode(continuation)

    # ── loglikelihood (multiple-choice scoring) ────────────────────

    @torch.no_grad()
    def loglikelihood(
        self,
        requests: list[Instance],
    ) -> list[tuple[float, bool]]:
        """Score continuations given context.

        For each (context, continuation) pair, returns (sum_log_p,
        is_greedy) where:
          sum_log_p = sum over continuation tokens of log P(token | prev tokens)
          is_greedy = True iff every continuation token equals argmax of
                      its position's logits (no sampling needed)

        Used by MMLU / ARC / hellaswag / etc. — score the multiple
        choice options against each other and pick the one with
        highest sum_log_p.
        """
        results: list[tuple[float, bool]] = []
        real_vocab = self._cfg.real_vocab_size
        pad_id = int(self._tok.pad_token_id if self._tok.pad_token_id is not None
                     else self.eot_token_id)

        # Pre-tokenize all requests so we can sort by length for
        # better batch padding efficiency. Track original index so we
        # can return results in the original order.
        items = []
        for orig_idx, req in enumerate(requests):
            context, continuation = req.args
            # Loglikelihood by default does NOT wrap (smoke run showed
            # chat-wrap drops MMLU scoring to ~22 % — below random — by
            # asking the model to score " A"/" B"/" C"/" D" continuations
            # after <|assistant|> where it expects <|think|>). Toggle via
            # chat_format_loglikelihood if testing.
            ctx_ids, cont_ids = self._encode_pair(
                self._wrap_context(context, for_generate=False), continuation,
            )
            # lm-eval can pass an empty context (unconditional scoring). The
            # logit slice below conditions the first continuation token on
            # position ctx_len-1, so an empty context would index -1 (the
            # wrong/last position). Seed a single BOS/EOS so ctx_len >= 1 and
            # the first token has a well-defined conditioning position.
            if not ctx_ids:
                ctx_ids = [self.eot_token_id]
            # Truncate from the left if context+continuation exceeds
            # max_length. We keep the continuation intact and lop off
            # the front of the context.
            full = ctx_ids + cont_ids
            if len(full) > self._max_length:
                drop = len(full) - self._max_length
                ctx_ids = ctx_ids[drop:]
                full = ctx_ids + cont_ids
            items.append((orig_idx, ctx_ids, cont_ids, full))

        # Sort by total length for tighter batches.
        items.sort(key=lambda x: len(x[3]))

        # Process in batches.
        out: dict[int, tuple[float, bool]] = {}
        for batch_start in range(0, len(items), self._batch_size):
            batch = items[batch_start : batch_start + self._batch_size]
            max_len = max(len(it[3]) for it in batch)
            input_ids = torch.full(
                (len(batch), max_len),
                pad_id,
                dtype=torch.long,
                device=self._device,
            )
            for i, (_, _, _, full) in enumerate(batch):
                input_ids[i, : len(full)] = torch.tensor(
                    full,
                    dtype=torch.long,
                    device=self._device,
                )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = self._model(input_ids).logits  # (B, T, V)

            # Slice to real_vocab and cast to fp32 for stable sums.
            logits = logits[..., :real_vocab].float()
            log_probs = F.log_softmax(logits, dim=-1)

            # For each sample: continuation lives at positions
            # [len(ctx), len(ctx)+len(cont)). The token at position p
            # is predicted by the logits at position p-1.
            for i, (orig_idx, ctx_ids, cont_ids, full) in enumerate(batch):
                cont_len = len(cont_ids)
                ctx_len = len(ctx_ids)
                if cont_len == 0:
                    out[orig_idx] = (0.0, True)
                    continue
                # Logits predicting continuation tokens.
                cont_logits = logits[i, ctx_len - 1 : ctx_len - 1 + cont_len, :]
                cont_log_probs = log_probs[i, ctx_len - 1 : ctx_len - 1 + cont_len, :]
                cont_target = torch.tensor(
                    cont_ids,
                    dtype=torch.long,
                    device=self._device,
                )
                token_log_probs = cont_log_probs.gather(
                    1,
                    cont_target.unsqueeze(1),
                ).squeeze(1)
                argmax = cont_logits.argmax(dim=-1)
                is_greedy = bool((argmax == cont_target).all().item())
                out[orig_idx] = (
                    float(token_log_probs.sum().item()),
                    is_greedy,
                )

        # Reassemble in original order.
        for orig_idx in range(len(requests)):
            results.append(out[orig_idx])
        return results

    def loglikelihood_rolling(
        self,
        requests: list[Instance],
    ) -> list[float]:
        """Not used by gsm8k / IFEval / MMLU. Implement if a benchmark
        we care about (e.g. WikiText perplexity) needs it."""
        raise NotImplementedError(
            "loglikelihood_rolling not implemented — none of the v5 "
            "target benchmarks need it.",
        )

    # ── generate_until (open-ended generation) ─────────────────────

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate continuations up to a stop string or max_gen_toks.

        gsm8k passes `until=["</s>", "Question:", "\n\n"]` (or similar);
        IFEval passes `until=["\n\n"]`. We use the model's KV-cached
        .generate, then post-process to truncate at the first stop
        string encountered.

        Sampling defaults are tuned for the undertrained 363M MoE
        (see __init__ docstring): temp 0.7, top_p 0.9, top_k 50,
        repetition_penalty 1.2. Per-request gen_kwargs override these
        but temperature is hard-capped at max_temperature (1.0).
        """
        results: list[str] = []
        for req in requests:
            context, gen_kwargs = req.args
            until = list(gen_kwargs.get("until") or [])
            if isinstance(until, str):
                until = [until]
            # Always stop at the model's trained end-of-answer tag —
            # the answer block is the target output for every benchmark
            # we run, and continuing past it just generates noise.
            if self._chat_format_generate and "<|end_turn|>" not in until:
                until.append("<|end_turn|>")
            max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            requested_temp = float(
                gen_kwargs.get("temperature", self._default_temperature),
            )
            temperature = min(requested_temp, self._max_temperature)
            top_p = float(gen_kwargs.get("top_p", self._default_top_p))
            top_k = int(gen_kwargs.get("top_k", self._default_top_k))
            repetition_penalty = float(
                gen_kwargs.get(
                    "repetition_penalty",
                    self._default_repetition_penalty,
                ),
            )

            # Wrap with SFT chat schema. The generation will start
            # immediately after <|assistant|>, exactly where the model
            # learned to emit <|think|>...<|/think|><|answer|>... .
            ctx_ids = self.tok_encode(
                self._gen_prefix + self._wrap_context(context, for_generate=True),
            )
            # Leave room for max_new tokens within the model's
            # max_position_embeddings. Drop oldest context if needed.
            keep = self._max_length - max_new
            if len(ctx_ids) > keep:
                ctx_ids = ctx_ids[-keep:]
            ctx_tensor = torch.tensor(
                [ctx_ids],
                dtype=torch.long,
                device=self._device,
            )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_ids = self._model.generate(
                    ctx_tensor,
                    max_new_tokens=max_new,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=self.eot_token_id,
                )

            gen_ids = out_ids[0, len(ctx_ids) :].tolist()
            text = self.tok_decode(gen_ids)

            # Truncate at first occurrence of any stop string. The
            # <|/answer|> auto-stop above means we cut cleanly at the
            # answer-block close in the common case.
            min_idx = math.inf
            for stop in until:
                idx = text.find(stop)
                if idx != -1 and idx < min_idx:
                    min_idx = idx
            if min_idx != math.inf:
                text = text[: int(min_idx)]
            # Strip the <|think|>...<|/think|> wrapper and return only
            # the answer-block contents so lm-eval's gsm8k extractor
            # (which looks for "#### X", "\\boxed{X}", or last number)
            # can find our numeric answer without us reformatting it.
            text = self._extract_answer(text)
            results.append(text)
        return results
