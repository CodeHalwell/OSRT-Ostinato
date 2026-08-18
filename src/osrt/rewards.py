"""Verifiable reward functions for GRPO training.

Uses rule-based rewards (no learned reward model):
- Correctness: extract numeric answer and compare to ground truth
- Format: check for proper <think>...</think> usage
- Reasoning quality: reward multi-step thinking when correct
- Truncation penalty: penalize hitting near seq_len (degenerate loops)
- Empty thinking penalty: penalize gaming format with no actual reasoning
"""

import re


def extract_numeric_answer(
    text: str,
    think_close: str = "</think>",
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
) -> str | None:
    """Extract the final numeric answer from model output.

    Priority order:
        1. LAST number inside answer_open / answer_close tags
           (many correct responses say e.g. "After 3 steps, the answer
           is 12" — returning the first number would score 3, wrong.
           The last number inside the answer tag is the model's final
           committed answer.)
        2. Last number after think_close (v3 format)
        3. Last number in the entire text (fallback)
    """
    # Strategy 1: look inside explicit answer tags. Use rindex on the
    # opening tag so a model that emits multiple <|answer|>...<|/answer|>
    # pairs (e.g. self-correcting mid-completion) gets credit for its
    # FINAL committed answer, not the first attempt. With rindex, end is
    # always the first close after the last open — correct for nested or
    # repeated tag pairs.
    if answer_open in text and answer_close in text:
        start = text.rindex(answer_open) + len(answer_open)
        end = (
            text.index(answer_close, start)
            if answer_close in text[start:] else len(text)
        )
        inside = text[start:end]
        numbers = re.findall(r"-?[\d,]+\.?\d*", inside)
        if numbers:
            return numbers[-1].replace(",", "")

    # Strategy 2: last number after the think_close tag
    if think_close and think_close in text:
        after_think = text.split(think_close, 1)[1].strip()
        numbers = re.findall(r"-?[\d,]+\.?\d*", after_think)
        if numbers:
            return numbers[-1].replace(",", "")

    # Strategy 3: fallback to last number in entire text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def extract_numeric_answer_strict(
    text: str,
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
) -> tuple[str | None, str]:
    """STRICT numeric answer extractor — closes the "last-number-wins" loophole.

    The loose extract_numeric_answer above lets a model spam multiple
    numbers and win on whichever happens to be last. GRPO exploits this:
    "I tried 50, then 32, but actually 18" → reward thinks model said 18.

    This stricter version returns (answer, confidence) where confidence
    is one of:
      - "single_number":     answer block contains exactly ONE number
      - "boxed":             answer wrapped in **N** or `N` or boxed{N}
      - "concluding":        last sentence is "= N" / "is N" / "answer: N"
      - "ambiguous":         multiple numbers, no clear conclusion → None
      - "no_extraction":     no number found at all → None

    Only "single_number", "boxed", "concluding" yield a non-None answer.
    GRPO can no longer dump numbers and hope.

    Use this in `check_answer_score` when a config flag is set; keep the
    loose version for inference scoring / display so a verbose-but-correct
    free-form answer still gets partial credit at probe-time.
    """
    if answer_open not in text or answer_close not in text:
        return None, "no_answer_block"

    start = text.rindex(answer_open) + len(answer_open)
    end_rel = text[start:].find(answer_close)
    if end_rel < 0:
        return None, "no_answer_close"
    inside = text[start:start + end_rel].strip()

    # All numbers in the answer block (allow integers, decimals, neg, commas)
    numbers = re.findall(r"-?\d+(?:[,.]\d+)*", inside)
    if not numbers:
        return None, "no_extraction"

    cleaned = [n.replace(",", "") for n in numbers]

    # Confidence tier 1: exactly one number in the entire answer block
    if len(cleaned) == 1:
        return cleaned[0], "single_number"

    # Confidence tier 2: a "boxed" or marked-up number — many appear,
    # but ONE is clearly the answer (e.g. **18**, `18`, \\boxed{18}, *18*)
    boxed = re.findall(
        r"(?:\*\*|`|\\boxed\{|\$\\boxed\{|\*)"
        r"(-?\d+(?:[,.]\d+)*)"
        r"(?:\*\*|`|\}|\$|\*)",
        inside,
    )
    if len(boxed) == 1:
        return boxed[0].replace(",", ""), "boxed"

    # Confidence tier 3: concluding phrase — last clause says "answer is N"
    # or "= N" or "Therefore, N". Catches well-formatted reasoning that
    # explicitly commits to the final answer.
    concluding = re.search(
        r"(?:answer\s+is|=\s*|therefore[,:]?\s*|so\s+the\s+answer\s+is\s*|"
        r"final\s+answer[\s:]*|=\s*\\boxed\{)"
        r"\s*\$?(-?\d+(?:[,.]\d+)*)\$?\s*[.\s]*$",
        inside,
        flags=re.IGNORECASE,
    )
    if concluding:
        return concluding.group(1).replace(",", ""), "concluding"

    # Multiple numbers, no clear marker — model is hedging. Reject.
    return None, "ambiguous"


def extract_gsm8k_answer(answer_text: str) -> str | None:
    """Extract the ground truth answer from GSM8K #### format."""
    if "####" in answer_text:
        final = answer_text.rsplit("####", 1)[1].strip()
        return final.replace(",", "").strip()
    return None


def extract_thinking(
    text: str,
    think_open: str = "<think>",
    think_close: str = "</think>",
) -> str:
    """Extract the content between think tags."""
    if think_open in text and think_close in text:
        start = text.index(think_open) + len(think_open)
        end = text.index(think_close)
        return text[start:end].strip()
    return ""


def check_format(
    text: str,
    think_open: str = "<think>",
    think_close: str = "</think>",
) -> bool:
    """Check if the completion uses proper thinking format."""
    return think_open in text and think_close in text


# ─────────────────────────────────────────────────────────────────────
# Composable reward functions (Unsloth-pattern)
# Each takes a single completion string + optional ground truth and
# returns a scalar score. Designed to be summed by a GRPO loop, mirror-
# ing TRL's `reward_funcs=[fn1, fn2, ...]` pattern. Each function is
# also exposed independently so we can ablate which signals help.
# ─────────────────────────────────────────────────────────────────────

def _format_regex(
    think_open: str,
    think_close: str,
    answer_open: str,
    answer_close: str,
) -> "re.Pattern":
    """Compile the exact-format regex for a chat template.

    Matches: think_open ... think_close (...) answer_open (CAPTURE) answer_close
    Returns the answer content in group(1).
    """
    return re.compile(
        re.escape(think_open) + r".*?"
        + re.escape(think_close) + r"\s*"
        + re.escape(answer_open) + r"(.+?)"
        + re.escape(answer_close) + r"\s*$",
        flags=re.MULTILINE | re.DOTALL,
    )


def match_format_exactly_score(
    text: str,
    think_open: str = "<|think|>",
    think_close: str = "<|/think|>",
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
    reward: float = 3.0,
) -> float:
    """Big binary reward (+`reward`) when the completion matches the
    full expected template exactly (think then answer, answer-close at
    end). Mirrors Unsloth's `match_format_exactly`."""
    pat = _format_regex(think_open, think_close, answer_open, answer_close)
    return reward if pat.search(text) is not None else 0.0


def match_format_approximately_score(
    text: str,
    think_open: str = "<|think|>",
    think_close: str = "<|/think|>",
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
    per_tag_pos: float = 0.5,
    per_tag_neg: float = -1.0,
) -> tuple[float, dict]:
    """Per-tag count-based signal: +per_tag_pos if a tag appears EXACTLY
    once, per_tag_neg otherwise. Sum across the 4 chat tags gives a
    fine-grained gradient even when the strict regex doesn't match.
    Mirrors Unsloth's `match_format_approximately`.

    Note the asymmetric default: positive 0.5 vs negative 1.0 — the
    penalty is intentionally stronger so the model can't game the
    reward by emitting two of every tag.
    """
    score = 0.0
    breakdown: dict[str, int] = {}
    for tag_name, tag in [
        ("think_open", think_open),
        ("think_close", think_close),
        ("answer_open", answer_open),
        ("answer_close", answer_close),
    ]:
        count = text.count(tag)
        breakdown[tag_name] = count
        score += per_tag_pos if count == 1 else per_tag_neg
    return score, breakdown


def check_answer_score(
    text: str,
    ground_truth: str | None,
    think_open: str = "<|think|>",
    think_close: str = "<|/think|>",
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
    strict_extraction: bool = False,
    ambiguous_penalty: float = -0.5,
) -> tuple[float, str]:
    """Tiered correctness reward using the format-aware extractor.
    Matches Unsloth's `check_answer` tier schedule but adapted to our
    tags. Returns (score, tier_label).

        exact string             +3.0
        whitespace-stripped      +1.5
        within  5 % (ratio)      +1.0
        within 20 % (ratio)      +0.5
        wrong / unparseable      -1.5
        no extract (no answer)   -0.0  (penalty handled by match_format)
        ambiguous (strict only)  ambiguous_penalty (-0.5)

    Why a separate function from `correctness_partial_credit`?
    The Unsloth schedule has TIGHTER positive rewards (max 3.0 vs 5.0)
    which composes better with multi-component reward stacks — keeps
    the math signal from drowning out the format signal.

    strict_extraction=True closes the "last-number-wins" hack the v1
    GRPO run revealed: model dumps multiple numbers in the answer
    block, hoping the last one matches GT. Strict mode uses
    extract_numeric_answer_strict which only returns an answer when
    extraction is high-confidence (single number / boxed / concluding
    phrase) and applies `ambiguous_penalty` when the answer block
    contains multiple unmarked numbers.
    """
    if strict_extraction:
        guess, confidence = extract_numeric_answer_strict(
            text, answer_open=answer_open, answer_close=answer_close,
        )
        if confidence == "ambiguous":
            # Model is hedging — small negative reward instead of just zero
            # so GRPO actively learns to commit to a single answer.
            return ambiguous_penalty, "ambiguous"
    else:
        # Use the loose format-aware numeric extractor — picks the LAST
        # number inside the answer tag. Backwards-compatible default.
        guess = extract_numeric_answer(
            text, think_close=think_close,
            answer_open=answer_open, answer_close=answer_close,
        )

    if guess is None or guess == "":
        return 0.0, "no_extract"
    if ground_truth is None or ground_truth == "":
        return 0.0, "no_ground_truth"

    gt = ground_truth.strip()
    if guess == gt:
        return 3.0, "exact"
    if guess.strip() == gt.strip():
        return 1.5, "stripped"

    try:
        gv = float(gt.replace(",", ""))
        pv = float(guess.replace(",", ""))
    except (ValueError, TypeError):
        return -1.5, "non_numeric_wrong"

    if gv == 0:
        return (3.0, "exact_numeric") if pv == 0 else (-1.5, "zero_gt_wrong")

    ratio = pv / gv
    if 0.95 <= ratio <= 1.05:
        return 1.0, "within_5pct"
    if 0.80 <= ratio <= 1.20:
        return 0.5, "within_20pct"
    return -1.5, "wrong"


def check_numbers_score(
    text: str,
    ground_truth: str | None,
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
    reward_match: float = 1.5,
    penalty_wrong: float = -0.5,
) -> tuple[float, str]:
    """Strict float-equality double-check on the LAST number inside the
    answer block. Mirrors Unsloth's `check_numbers` — a second,
    smaller reward that fires only on exact numeric match. Useful
    composed alongside `check_answer_score` so the model gets a clean
    binary "you got the number right" gradient in addition to the
    tiered match score."""
    if ground_truth is None:
        return 0.0, "no_ground_truth"
    guess = extract_numeric_answer(
        text, think_close="</think>",  # fallback, doesn't matter here
        answer_open=answer_open, answer_close=answer_close,
    )
    if guess is None:
        return 0.0, "no_number"
    try:
        gv = float(ground_truth.replace(",", "").strip())
        pv = float(guess.replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0, "non_numeric"
    return (reward_match, "match") if gv == pv else (penalty_wrong, "miss")


# ─────────────────────────────────────────────────────────────────────
# EMA reward tracker — process-global, throttled diagnostic logging.
# Mirrors Unsloth's `_EMA_REWARD` + PRINT_EVERY_STEPS pattern.
# ─────────────────────────────────────────────────────────────────────


class RewardEMA:
    """Track exponentially-smoothed mean reward across GRPO steps.

    Cheap signal-quality metric — drift in the EMA tells you if GRPO
    is actually learning vs single-batch noise. Updates per-batch with
    `update(mean_reward)`, reports current state with `state()`.

    Optionally throttles a `print_fn` call so the per-step log isn't
    flooded — by default emits every PRINT_EVERY_N_CALLS update.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        print_every_n_calls: int = 0,
        print_fn=print,
    ) -> None:
        self.alpha = alpha
        self.value: float | None = None
        self.n_calls: int = 0
        self.print_every_n_calls = print_every_n_calls
        self.print_fn = print_fn

    def update(self, batch_mean: float, **extras) -> float:
        """Update EMA with this batch's mean reward. Returns new EMA.
        extras (e.g. `exact_hits=2, parse_fails=1`) get included in the
        throttled diagnostic print."""
        self.value = (
            batch_mean if self.value is None
            else self.alpha * batch_mean + (1.0 - self.alpha) * self.value
        )
        self.n_calls += 1
        if (
            self.print_every_n_calls > 0
            and self.n_calls % self.print_every_n_calls == 0
        ):
            extras_str = "  ".join(f"{k}={v}" for k, v in extras.items())
            self.print_fn(
                f"  [reward_ema #{self.n_calls}] batch_mean={batch_mean:+.3f}  "
                f"ema={self.value:+.3f}  {extras_str}",
                flush=True,
            )
        return self.value

    def state(self) -> dict:
        return {"ema": self.value, "n_calls": self.n_calls}


def strict_template_score(
    text: str,
    think_open: str = "<|think|>",
    think_close: str = "<|/think|>",
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
    user_marker: str = "<|user|>",
) -> tuple[float, dict]:
    """Score how cleanly the completion follows the chat template.

    Returns (score in [-1.0, 1.0], breakdown).

    Designed against the failure modes observed in post-MOPD inference
    (2026-06-06):
      - model generates `<|answer|>...<|/answer|><|answer|>...<|/answer|>`
        (multiple answer blocks in a single completion)
      - model starts a new turn mid-completion with `<|user|>...`
      - model emits answer without preceding think
      - model emits think without following answer (no commitment)
      - model wraps answer in extra structural fluff after `<|/answer|>`

    Scoring:
        +1.0   exactly one think+answer pair, ends at `<|/answer|>`
        +0.6   exactly one think+answer pair (trailing content allowed)
        +0.3   answer block present but no/extra think OR open-but-no-close
        -0.3   multiple `<|answer|>` opens (most common bug)
        -0.6   `<|user|>` appears in completion (trying to start new turn)
        -1.0   no `<|answer|>` at all (no commitment to a final answer)

    The +1.0 / +0.6 split is intentional: the model should learn to emit
    `<|/answer|>` then stop, not keep going. With stop-token enforcement
    at inference time the difference matters less, but during training
    a positive gradient on stopping cleanly is still useful.
    """
    breakdown = {}

    n_answer_open = text.count(answer_open)
    n_answer_close = text.count(answer_close)
    n_think_open = text.count(think_open)
    n_think_close = text.count(think_close)
    has_user = user_marker in text

    breakdown.update({
        "n_answer_open": n_answer_open,
        "n_answer_close": n_answer_close,
        "n_think_open": n_think_open,
        "n_think_close": n_think_close,
        "user_in_completion": has_user,
    })

    # Hardest failure: tried to start a new turn
    if has_user:
        return -0.6, {**breakdown, "verdict": "user_marker_in_completion"}

    # Critical failure: no committed answer
    if n_answer_open == 0 or n_answer_close == 0:
        return -1.0, {**breakdown, "verdict": "no_answer"}

    # Multiple answer blocks — the MOPD failure mode
    if n_answer_open > 1 or n_answer_close > 1:
        return -0.3, {**breakdown, "verdict": "multiple_answers"}

    # Single answer pair from here. Check think structure.
    has_one_think = (n_think_open == 1 and n_think_close == 1)
    extra_think = (n_think_open > 1 or n_think_close > 1)
    if extra_think:
        return 0.3, {**breakdown, "verdict": "multiple_thinks"}

    # Check ordering: think_open < think_close < answer_open < answer_close
    if has_one_think:
        to = text.index(think_open)
        tc = text.index(think_close)
        ao = text.index(answer_open)
        ac = text.index(answer_close)
        if not (to < tc < ao < ac):
            return 0.3, {**breakdown, "verdict": "out_of_order"}

    # Single clean answer. Check for trailing content after `<|/answer|>`.
    trailing = text[text.index(answer_close) + len(answer_close):].strip()
    breakdown["trailing_chars"] = len(trailing)
    if trailing:
        return 0.6, {**breakdown, "verdict": "clean_with_trailing"}

    return 1.0, {**breakdown, "verdict": "clean"}


def numeric_match(predicted: str | None, ground_truth: str | None) -> bool:
    """Check if two numeric answers match (handles float comparison)."""
    if predicted is None or ground_truth is None:
        return False

    try:
        pred_val = float(predicted)
        gt_val = float(ground_truth)
        if pred_val == gt_val:
            return True
        if abs(gt_val) > 1e-8:
            return abs(pred_val - gt_val) / abs(gt_val) < 1e-4
        return abs(pred_val - gt_val) < 1e-8
    except (ValueError, OverflowError):
        return predicted.strip() == ground_truth.strip()


def count_reasoning_steps(thinking: str) -> int:
    """Count reasoning steps in the thinking block.

    Looks for numbered steps, newlines with content, or sentence breaks
    that indicate multi-step reasoning.
    """
    if not thinking:
        return 0

    # Count numbered steps (1. 2. 3. or Step 1, Step 2)
    numbered = re.findall(
        r"(?:^|\n)\s*(?:\d+[\.\):]|step\s+\d+)",
        thinking, re.IGNORECASE,
    )
    if len(numbered) >= 2:
        return len(numbered)

    # Fallback: count non-empty lines as steps
    lines = [line.strip() for line in thinking.split("\n") if line.strip()]
    return len(lines)


def correctness_partial_credit(
    predicted: str | None,
    ground_truth: str | None,
) -> tuple[float, str]:
    """Tiered correctness reward. ONLY an exact answer scores positive.

    Returns (score, tier_label). Score ranges from +5.0 (exact) to
    -2.5 (no extractable number).

    Tier schedule:
        exact match     +5.0     ← the ONLY positive tier
        0.5x – 2.0x     -0.5     ← wrong, but same order of magnitude
        wrong number    -2.0     ← off by an order of magnitude+
        no number       -2.5     ← couldn't even parse one

    Why the graded NEGATIVE tiers, and why no positive ones
    ──────────────────────────────────────────────────────
    The negative tiers exist to keep variance in a group's rewards when no
    rollout is exactly right. At ~5 % accuracy with group_size=8, binary
    scoring left most groups with zero correct rollouts → uniform rewards →
    group-normalised advantage of all zeros → no gradient → frozen policy.
    Spreading wrong answers across -0.5 / -2.0 / -2.5 fixes that without ever
    paying for a wrong answer.

    This originally followed the Unsloth tutorial's *proximity* schedule, which
    also paid within_5_pct +3.5, within_10_pct +2.5, within_20_pct +1.5. Those
    were REMOVED on 2026-08-10 because they measurably degraded the model.
    A wrong answer earning +3.5 against exact's +5.0 is 70 % of the reward for
    being wrong, and because advantages are group-normalised, a near-miss in a
    group with no exact hit becomes the BEST rollout and gets a large positive
    advantage — training the policy toward plausible numbers of roughly the
    right magnitude rather than correct ones. GRPO v6 wave 1 lost 7.5pp of
    held-out GSM8K accuracy in 100 steps this way (soup 20.0 % → step 100
    12.5 %, n=200), degrading in both reasoning modes and under both personas,
    with format decay and length inflation alongside.

    Note that on arithmetic word problems "within 5 %" is not partial
    understanding — it is simply wrong, so there is nothing to give credit for.
    Proximity credit only makes sense where the target is a genuine estimate.
    """
    if predicted is None or predicted == "":
        return -2.5, "no_extraction"
    if ground_truth is None or ground_truth == "":
        return 0.0, "no_ground_truth"

    # String-strip exact match before numeric coercion (handles "53"
    # vs " 53" vs "53.0" cases the comma-strip in numeric_match
    # already handles).
    if str(predicted).strip() == str(ground_truth).strip():
        return 5.0, "exact"

    try:
        gt = float(str(ground_truth).strip().replace(",", ""))
        pr = float(str(predicted).strip().replace(",", ""))
    except (ValueError, TypeError):
        return -2.0, "non_numeric"

    if gt == 0:
        # Special-case division-by-zero: only exact match is valid
        return (5.0, "exact") if pr == 0 else (-2.0, "zero_gt_wrong")

    if pr == gt:
        return 5.0, "exact_numeric"

    ratio = pr / gt
    # NO positive tier for a wrong answer — see the docstring. Anything within
    # 20 % is already inside the 0.5x-2.0x band, so the former within_5/10/20
    # tiers fall through to wrong_same_order (-0.5) with no new constants.
    if 0.50 <= ratio <= 2.00:
        return -0.5, "wrong_same_order"
    return -2.0, "wrong_far_off"


_SPECIAL_TAG = re.compile(r"<\|[^|]*\|>")


def few_shot_echo_score(
    completion: str,
    exemplar: str,
    n: int = 12,
) -> tuple[int, int]:
    """Verbatim overlap between a completion and a few-shot exemplar.

    Returns (matching_ngrams, total_completion_ngrams) over WORD n-grams.

    Why n-grams and not a similarity ratio: we want to catch a long verbatim
    RUN, not stylistic resemblance. A shared 12-word span is copying; a shared
    4-word span ("let C be the") is idiom that good reasoning legitimately
    reuses, and penalising it would punish the very pattern the exemplar is
    there to teach. n=12 is long enough that the exemplar's generic phrasing
    does not trip it.

    Special tags are stripped from BOTH sides first. Every well-formed
    completion necessarily contains `<|think|>` and `<|answer|>`, and so does
    the exemplar, so leaving them in would produce guaranteed matches that have
    nothing to do with copying.
    """
    if not exemplar.strip():
        return 0, 0

    def grams(text: str) -> list[tuple[str, ...]]:
        words = _SPECIAL_TAG.sub(" ", text).split()
        if len(words) < n:
            return []
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    ex = set(grams(exemplar))
    comp = grams(completion)
    if not ex or not comp:
        return 0, len(comp)
    return sum(1 for g in comp if g in ex), len(comp)


def length_ramp_penalty(completion_tokens: int, max_tokens: int) -> float:
    """Smooth length penalty that ramps as the completion approaches the
    output budget.

    Designed to discourage the model from padding its <|think|> block to
    fill the entire generation budget — observed in math reasoning
    models that learn "longer reasoning = more reward" without a
    counter-pressure. The ramp targets the *use of the budget*, not
    the absolute token count, so it scales with whatever max_gen_len
    is configured.

    Schedule (linear between breakpoints):
        ≤ 80 % of max_tokens : 0.0   (think freely, no penalty)
          80 %               : -0.5
          90 %               : -0.75
          100 %              : -1.0  (caps; model hit the cap)

    Returns 0.0 if max_tokens is 0 or completion_tokens is 0 (used as
    a no-op when GRPO doesn't pass token info — e.g. unit tests).
    """
    if max_tokens <= 0 or completion_tokens <= 0:
        return 0.0
    pct = completion_tokens / max_tokens
    if pct < 0.80:
        return 0.0
    if pct >= 1.00:
        return -1.0
    # Linear ramp from -0.5 at 80 % to -1.0 at 100 %:
    #   slope = (−1.0 − (−0.5)) / (1.00 − 0.80) = −2.5
    # Verifies: 0.80→-0.5, 0.90→-0.75, 1.00→-1.0
    return -0.5 + (pct - 0.80) * (-2.5)


def compute_reward(
    completion: str,
    ground_truth_answer: str,
    correctness_weight: float = 1.0,
    format_weight: float = 0.2,
    strict_template_weight: float = 0.0,
    length_penalty: float = 0.0,
    think_open: str = "<think>",
    think_close: str = "</think>",
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
    user_marker: str = "<|user|>",
    max_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_bonus: float = 0.3,
    truncation_penalty: float = -0.5,
    empty_think_penalty: float = -0.1,
    few_shot_exemplar: str = "",
    few_shot_echo_penalty: float = -3.0,
) -> tuple[float, dict]:
    """Compute the total reward for a single completion.

    Reward components:
        1. Correctness (+1.0): correct final answer
        2. Format (+0.2): proper think_open/think_close tags present
        3. Reasoning bonus (+0.3): multi-step thinking when correct
        4. Truncation penalty (-0.5): output hit 90% of max tokens
        5. Empty thinking penalty (-0.1): thinking block has no content
        5b. Few-shot echo penalty (-3.0): a 12-word verbatim run from the
            supplied exemplar appears in the completion (inactive when no
            exemplar is passed)
        6. Length penalty (0.0 default): disabled, reasoning needs room

    IMPORTANT: pass the actual tag strings used during training (e.g. v4 uses
    single-token native tags '<|think|>' and '<|/think|>'). The defaults match
    v3's plain-string tags and will silently fail for v4 if not overridden.

    Returns:
        (total_reward, reward_breakdown_dict)
    """
    reward = 0.0
    breakdown = {}

    # 1. Correctness reward (core signal) — tiered, but only an EXACT answer
    # scores positive. The graded negative tiers replace the original binary
    # correctness (+1 if exact, 0 otherwise) which caused GRPO collapse on this
    # model: at ~5 % gsm8k baseline acc with group_size=8, most groups had zero
    # correct rollouts → uniform rewards → zero advantage → frozen updates. See
    # correctness_partial_credit() for the tier schedule (+5.0 exact down to
    # -2.5 no-extract) and for why the positive proximity tiers were removed.
    # correctness_weight scales the ENTIRE tier output (default 1.0).
    predicted = extract_numeric_answer(
        completion,
        think_close=think_close,
        answer_open=answer_open,
        answer_close=answer_close,
    )
    gt = extract_gsm8k_answer(ground_truth_answer)
    if gt is None:
        gt = ground_truth_answer.replace(",", "").strip()

    tier_score, tier_label = correctness_partial_credit(predicted, gt)
    correctness_r = tier_score * correctness_weight
    correct = (tier_label in ("exact", "exact_numeric"))
    reward += correctness_r
    breakdown["correct"] = correct
    breakdown["correctness_tier"] = tier_label
    breakdown["correctness_reward"] = correctness_r

    # 2. Format reward (boolean: think tags present)
    has_format = check_format(completion, think_open, think_close)
    format_r = format_weight if has_format else 0.0
    reward += format_r
    breakdown["has_format"] = has_format
    breakdown["format_reward"] = format_r

    # 2b. Strict template reward (graded: catches multi-answer, no-answer,
    # user-marker-in-completion, out-of-order). When strict_template_weight
    # is 0 (default) this is a no-op for backwards compat with the original
    # math-only GRPO config — flip it on (e.g. 0.5) for the multi-env
    # rewards where template adherence is a learnable target.
    if strict_template_weight > 0.0:
        strict_score, strict_breakdown = strict_template_score(
            completion,
            think_open=think_open,
            think_close=think_close,
            answer_open=answer_open,
            answer_close=answer_close,
            user_marker=user_marker,
        )
        strict_r = strict_template_weight * strict_score
        reward += strict_r
        breakdown["strict_template_score"] = strict_score
        breakdown["strict_template_reward"] = strict_r
        breakdown["strict_template_verdict"] = strict_breakdown.get(
            "verdict", "")
    else:
        breakdown["strict_template_score"] = 0.0
        breakdown["strict_template_reward"] = 0.0

    # 3. Reasoning quality bonus (only when correct — reward thinking that works)
    thinking = extract_thinking(completion, think_open, think_close)
    n_steps = count_reasoning_steps(thinking)
    breakdown["reasoning_steps"] = n_steps
    breakdown["thinking_length"] = len(thinking.split()) if thinking else 0

    if correct and n_steps >= 2:
        # Graduated bonus: more steps = more reward, capped at 3+ steps
        step_bonus = reasoning_bonus * min(n_steps / 3.0, 1.0)
        reward += step_bonus
        breakdown["reasoning_bonus"] = step_bonus
    else:
        breakdown["reasoning_bonus"] = 0.0

    # 3b. Few-shot regurgitation penalty. Only active when an exemplar is
    # supplied, so unhinted/unexemplified prompts are untouched.
    #
    # A few-shot prompt the policy learns to ECHO is worse than none: it stops
    # being a pattern to follow and becomes a template to reproduce. The direct
    # incentive is small (copied prose collects the +0.2 format term and, if the
    # answer happens to be right, the +0.3 step bonus) but the failure mode is
    # exactly the one already on record — a trace that looks like reasoning and
    # does none. The penalty is deliberately LARGE relative to what echoing can
    # earn, so copying is never the cheapest route to a well-formed trace.
    echo_hits, echo_total = few_shot_echo_score(completion, few_shot_exemplar)
    echo_pen = few_shot_echo_penalty if echo_hits > 0 else 0.0
    reward += echo_pen
    breakdown["few_shot_echo_hits"] = echo_hits
    breakdown["few_shot_echo_ngrams"] = echo_total
    breakdown["few_shot_echo_penalty"] = echo_pen

    # 4. Length-ramp penalty (smooth, scales toward output token budget)
    # Replaces the old binary truncation_penalty (which only fired at
    # 90 %). Now: 0 below 80 %, then linear ramp from -0.5 at 80 %
    # → -0.75 at 90 % → -1.0 at 100 %. Discourages padding the think
    # block to fill the budget without harshly penalising completions
    # that legitimately need 60-75 % of the room.
    # `truncation_penalty` config field is now unused (kept for
    # backwards compat with saved configs); the ramp is hardcoded
    # in length_ramp_penalty() above.
    length_pen = length_ramp_penalty(completion_tokens, max_tokens)
    reward += length_pen
    breakdown["length_ramp_penalty"] = length_pen
    breakdown["length_ramp_pct"] = (
        completion_tokens / max_tokens if max_tokens > 0 else 0.0
    )
    # Keep `truncated` flag for compatibility with downstream logging
    # — true iff we hit the cap (≥ 100 %).
    breakdown["truncated"] = (
        max_tokens > 0 and completion_tokens >= max_tokens
    )
    breakdown["truncation_penalty"] = 0.0  # deprecated; see length_ramp_penalty

    # 5. Empty thinking penalty (gaming format with no content)
    if has_format and len(thinking.split()) < 3:
        reward += empty_think_penalty
        breakdown["empty_thinking"] = True
        breakdown["empty_think_penalty"] = empty_think_penalty
    else:
        breakdown["empty_thinking"] = False
        breakdown["empty_think_penalty"] = 0.0

    # 6. Legacy length penalty (disabled by default)
    n_words = len(completion.split())
    length_r = length_penalty * n_words
    reward += length_r
    breakdown["n_words"] = n_words
    breakdown["length_reward"] = length_r

    breakdown["total_reward"] = reward
    return reward, breakdown


# ─────────────────────────────────────────────────────────────────────
# Env-specific rewards for multi-env GRPO
# ─────────────────────────────────────────────────────────────────────


def extract_answer_text(
    completion: str,
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
) -> str | None:
    """Extract the FIRST answer-block content as a string. Returns the
    content inside the first <|answer|>...<|/answer|> pair, or None if
    no such block exists. For multi-env tasks where the answer isn't
    necessarily numeric (chat replies, code, instruction-following)."""
    if answer_open not in completion or answer_close not in completion:
        return None
    start = completion.index(answer_open) + len(answer_open)
    end_rel = completion[start:].find(answer_close)
    if end_rel < 0:
        return None
    return completion[start:start + end_rel].strip()


def ifeval_constraint_reward(
    completion: str,
    instruction_id_list: list[str] | None,
    kwargs_list: list[dict] | None,
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
    reward_per_constraint: float = 1.0,
    penalty_no_answer: float = -1.0,
) -> tuple[float, dict]:
    """Verifiable reward for Google's IFEval-style instruction-following.

    IFEval prompts come with a list of constraint identifiers (e.g.
    "length_constraints:number_words", "keywords:include_keywords")
    and per-constraint kwargs (e.g. {"num_words": 200}, {"keywords":
    ["sustainability"]}). This function evaluates a SUBSET of those
    that are cheap to verify locally without IFEval's full eval harness.

    Returns (score, breakdown). Score = sum of per-constraint rewards
    if the answer block exists, else `penalty_no_answer`. Constraints
    we don't know how to verify locally are skipped (no reward, no
    penalty) — better than reporting fake satisfaction.

    Supported constraints (V1):
      - length_constraints:number_words      ({num_words})
      - length_constraints:number_sentences  ({num_sentences})
      - keywords:include_keywords            ({keywords})
      - keywords:forbidden_words             ({forbidden_words})
      - startswith / endswith                ({start_phrase, end_phrase})
      - punctuation:no_comma                 ({})

    Coverage from a typical IFEval batch: ~50-70% of constraints.
    """
    answer = extract_answer_text(completion, answer_open, answer_close)
    if answer is None:
        return penalty_no_answer, {"verdict": "no_answer"}

    if not instruction_id_list:
        # Some IFEval rows have no constraints — just reward for having
        # an answer at all (caller handles format rewards separately).
        return 0.0, {"verdict": "no_constraints"}

    kwargs_list = kwargs_list or [{}] * len(instruction_id_list)
    hits, misses, skipped = 0, 0, 0
    for inst_id, kw in zip(instruction_id_list, kwargs_list):
        kw = kw or {}
        # number_words
        if inst_id == "length_constraints:number_words":
            target = kw.get("num_words")
            if target is not None:
                actual = len(answer.split())
                # Allow ±5% tolerance
                if abs(actual - target) <= max(int(target * 0.05), 2):
                    hits += 1
                else:
                    misses += 1
            else:
                skipped += 1
        # number_sentences
        elif inst_id == "length_constraints:number_sentences":
            target = kw.get("num_sentences")
            if target is not None:
                actual = len([s for s in re.split(r"[.!?]+", answer) if s.strip()])
                if abs(actual - target) <= 1:
                    hits += 1
                else:
                    misses += 1
            else:
                skipped += 1
        # include_keywords
        elif inst_id == "keywords:include_keywords":
            keywords = kw.get("keywords") or []
            if keywords:
                lower = answer.lower()
                if all(k.lower() in lower for k in keywords):
                    hits += 1
                else:
                    misses += 1
            else:
                skipped += 1
        # forbidden_words
        elif inst_id == "keywords:forbidden_words":
            forbidden = kw.get("forbidden_words") or []
            if forbidden:
                lower = answer.lower()
                if not any(f.lower() in lower for f in forbidden):
                    hits += 1
                else:
                    misses += 1
            else:
                skipped += 1
        # startswith
        elif inst_id == "startswith:response":
            phrase = kw.get("start_phrase")
            if phrase:
                if answer.lower().startswith(phrase.lower()):
                    hits += 1
                else:
                    misses += 1
            else:
                skipped += 1
        # endswith
        elif inst_id == "endswith:response":
            phrase = kw.get("end_phrase")
            if phrase:
                if answer.lower().rstrip(".!?").endswith(phrase.lower()):
                    hits += 1
                else:
                    misses += 1
            else:
                skipped += 1
        # no_comma
        elif inst_id == "punctuation:no_comma":
            if "," not in answer:
                hits += 1
            else:
                misses += 1
        else:
            skipped += 1

    total = (hits - misses) * reward_per_constraint
    return total, {
        "verdict": "evaluated",
        "constraints_hit": hits,
        "constraints_miss": misses,
        "constraints_skipped": skipped,
    }


def mbpp_test_reward(
    completion: str,
    test_list: list[str] | None,
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
    reward_pass: float = 2.0,
    reward_partial: float = 1.0,
    penalty_fail: float = -1.5,
    timeout_sec: float = 3.0,
    allow_unsafe_exec: bool = False,
    python_executable: str = "/usr/bin/python3",
) -> tuple[float, dict]:
    """Verifiable reward for MBPP code generation via SANDBOXED subprocess.

    Extracts the answer block, attempts to run it as Python with the
    test_list assertions appended, returns reward based on test pass rate.

    Tries to extract a code block (```python ... ```) from the answer;
    falls back to the whole answer if no code fence.

    SAFETY model — model output is UNTRUSTED. This function:
      1. Refuses to run unless `allow_unsafe_exec=True` is explicitly
         set by the caller. Defaults to off so an accidental import
         can't suddenly execute training-data code.
      2. Strips inherited env vars before exec — only PATH, LC_ALL,
         LANG. Critically excludes HF_TOKEN, WANDB_API_KEY,
         GEMINI_API_KEY, DEEPSEEK_API_KEY, AWS_*, OPENAI_API_KEY,
         and any other secret that might be present on Modal.
      3. Runs in a fresh CWD (tempdir, cleaned up) so the script
         can't read sibling files or write anywhere that persists.
      4. New process session + on-timeout kills the whole process
         group (handles fork-bombs / async writes).
      5. Caps stdout/stderr reads (combined ≤ 64 KiB) so an OOM
         loop in the child can't blow up the parent.
      6. Pins absolute python path (`/usr/bin/python3` by default,
         override via arg) so PATH manipulation can't divert exec.

    For multi-tenant or production deployment, use Modal Sandbox or
    nsjail/bubblewrap instead — these in-process hardenings are best-
    effort defence-in-depth, not a true sandbox.
    """
    if not test_list:
        return 0.0, {"verdict": "no_tests"}

    answer = extract_answer_text(completion, answer_open, answer_close)
    if answer is None:
        return penalty_fail, {"verdict": "no_answer"}

    if not allow_unsafe_exec:
        # Surface a verdict instead of silently scoring 0 so the caller
        # knows their MBPP env is being skipped.
        return 0.0, {"verdict": "exec_disabled_set_allow_unsafe_exec"}

    # Pull a python code block out of the answer if fenced
    fence = re.search(r"```python\s*\n(.*?)```", answer, re.DOTALL)
    code = fence.group(1) if fence else answer

    # Build a test program that runs each assertion INDEPENDENTLY and
    # prints the pass count. The previous implementation concatenated
    # all assertions into one script body; Python exits non-zero on the
    # FIRST failing assertion, so a 3-of-4-pass completion scored the
    # same as a 0-pass completion. This silently destroyed partial
    # reward signal in GRPO (per review/deep-dive 2026-06-08 P2).
    #
    # Each test runs inside `try/exec/except` in the SAME interpreter
    # so the code's defined functions are visible via globals().
    # Sandboxing (env strip, tempdir cwd, timeout, group-kill) is
    # unchanged — the outer subprocess still runs everything once.
    _tests_repr = repr(list(test_list))
    test_program = (
        code
        + "\n\n"
        + "__mbpp_tests = " + _tests_repr + "\n"
        + "__mbpp_passed = 0\n"
        + "for __mbpp_t in __mbpp_tests:\n"
        + "    try:\n"
        + "        exec(__mbpp_t)\n"
        + "        __mbpp_passed += 1\n"
        + "    except Exception:\n"
        + "        pass\n"
        + "print(f'__MBPP_PASSED__{__mbpp_passed}', flush=True)\n"
    )

    # ── sandboxed exec ──
    import os
    import signal
    import subprocess
    import tempfile

    # Minimal env. Strip all secrets — explicit allowlist is safer than
    # a blocklist (we'd inevitably miss some var name in a blocklist).
    sandbox_env = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONIOENCODING": "utf-8",
        # No HOME / TMP — let Python use defaults inside the tempdir cwd.
    }

    tmpdir = tempfile.mkdtemp(prefix="mbpp_eval_")
    proc: subprocess.Popen | None = None
    try:
        # start_new_session=True puts the child into a new process group
        # so a timeout kill takes down forks too.
        proc = subprocess.Popen(
            [python_executable, "-c", test_program],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmpdir,
            env=sandbox_env,
            start_new_session=True,
        )
        try:
            stdout_bytes, stderr_bytes = proc.communicate(
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            # Kill the entire process group, not just the leader.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            # Drain the pipes so the FD doesn't leak.
            try:
                proc.communicate(timeout=2.0)
            except Exception:
                pass
            return penalty_fail, {"verdict": "timeout"}

        # Cap captured output so we don't accidentally log megabytes of
        # model stdout (and any secrets it might have printed if the
        # sandbox somehow leaked).
        _stdout = stdout_bytes[:65536]
        _stderr = stderr_bytes[:65536]
        del stdout_bytes, stderr_bytes  # release memory
        # Parse the per-test pass count emitted by the wrapped script.
        # Marker is bytes-literal so it survives a non-UTF8 stdout.
        # A missing marker means the script crashed before reaching the
        # final print (e.g. model code raised at import or the sandbox
        # was killed) — that's an honest all-fail, NOT all-pass on rc==0.
        _marker_re = re.compile(rb"__MBPP_PASSED__(\d+)")
        _m = _marker_re.search(_stdout)
        passed = int(_m.group(1)) if _m else 0
    except Exception as e:  # noqa: BLE001
        return penalty_fail, {"verdict": f"exec_error:{type(e).__name__}"}
    finally:
        # Best-effort cleanup of the cwd tempdir.
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    total = len(test_list)
    if passed == total:
        return reward_pass, {
            "verdict": "all_pass", "passed": passed, "total": total,
        }
    if passed > 0:
        # Partial credit. Linear in pass rate so 1-of-4 and 3-of-4 give
        # meaningfully different rewards — restores the per-test
        # gradient signal that GRPO needs to escape uniform-reward
        # groups (LEARNINGS.md "uniform reward groups" failure mode).
        return reward_partial * (passed / total), {
            "verdict": "partial", "passed": passed, "total": total,
        }
    return penalty_fail, {
        "verdict": "all_fail", "passed": 0, "total": total,
    }


def regurgitation_penalty(
    system_prompt: str | None,
    completion: str,
    ngram_size: int = 5,
    free_threshold: float = 0.10,
    saturation_threshold: float = 0.40,
    max_penalty: float = -5.0,
) -> tuple[float, dict]:
    """Heavy penalty when the model regurgitates the system prompt.

    Common failure mode for newly system-prompt-trained models: they
    quote chunks of the system prompt back at the user instead of
    answering. We detect this with token n-gram overlap.

    Algorithm:
      1. Tokenize system prompt and completion to word-level n-grams
         (n = ngram_size, default 5).
      2. Compute fraction of system n-grams that appear in the
         completion.
      3. If overlap < free_threshold (10 %): no penalty (legitimate
         echoing — "the answer the user asked for is X").
      4. Linearly scale penalty from 0 at free_threshold to max_penalty
         at saturation_threshold (40 %). Beyond 40 % overlap, full
         max_penalty applies.

    Why n=5? 5-grams are long enough to filter out incidental matches
    on common phrases like "the answer is" but short enough that
    paraphrased regurgitation still triggers.

    Why max -5.0? With check_answer_score peaking at +3.0 and
    format rewards at +5.0, a -5.0 regurgitation penalty makes
    regurgitating strictly WORSE than producing a clean wrong answer
    — exactly the gradient we want.

    Returns (penalty, breakdown). Returns 0 if system prompt is None
    or too short.
    """
    if system_prompt is None or not system_prompt.strip():
        return 0.0, {"reason": "no_system_prompt"}

    # Word-level tokenization (avoids re-tokenizing through the model
    # tokenizer; n-grams of words are robust to whitespace differences).
    sys_words = system_prompt.split()
    if len(sys_words) < ngram_size:
        return 0.0, {"reason": "system_too_short"}

    comp_words = completion.split()
    if len(comp_words) < ngram_size:
        return 0.0, {"reason": "completion_too_short"}

    # Build n-gram sets. Lowercase so capitalization differences don't
    # hide regurgitation.
    sys_ngrams = {
        tuple(w.lower() for w in sys_words[i:i + ngram_size])
        for i in range(len(sys_words) - ngram_size + 1)
    }
    comp_ngrams = {
        tuple(w.lower() for w in comp_words[i:i + ngram_size])
        for i in range(len(comp_words) - ngram_size + 1)
    }
    if not sys_ngrams:
        return 0.0, {"reason": "no_sys_ngrams"}

    overlap = sys_ngrams & comp_ngrams
    overlap_frac = len(overlap) / len(sys_ngrams)

    if overlap_frac < free_threshold:
        return 0.0, {
            "overlap_frac": overlap_frac,
            "n_matches": len(overlap),
            "n_sys_ngrams": len(sys_ngrams),
            "verdict": "clean",
        }

    # Scale: penalty goes from 0 at free_threshold to max_penalty at
    # saturation_threshold. Beyond saturation, full penalty.
    span = max(saturation_threshold - free_threshold, 1e-6)
    scaled = min(1.0, (overlap_frac - free_threshold) / span)
    penalty = max_penalty * scaled

    return penalty, {
        "overlap_frac": overlap_frac,
        "n_matches": len(overlap),
        "n_sys_ngrams": len(sys_ngrams),
        "verdict": "regurgitating",
    }


def compose_template_rewards(
    completion: str,
    ground_truth_answer: str | None,
    think_open: str = "<|think|>",
    think_close: str = "<|/think|>",
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
    # Per-component weights — tune per-stage. Defaults match the
    # Unsloth tutorial values which compose to a max reward of about
    # +7.5 (3.0 exact-format + 2.0 all-tags-once + 3.0 answer-exact +
    # 1.5 number-strict-match) and a min of about -4.0 (no format,
    # all tags wrong count, answer parse-fail, number miss).
    exact_format_reward: float = 3.0,
    approx_format_pos: float = 0.5,
    approx_format_neg: float = -1.0,
    answer_check: bool = True,
    number_check_reward: float = 1.5,
    number_check_penalty: float = -0.5,
    user_marker: str = "<|user|>",
    strict_template_weight: float = 0.0,
    strict_extraction: bool = False,
    ambiguous_penalty: float = -0.5,
    system_prompt: str | None = None,
    regurgitation_weight: float = 1.0,
    regurgitation_max_penalty: float = -5.0,
) -> tuple[float, dict]:
    """Run all template + correctness rewards as a list and sum.

    Mirrors TRL's `reward_funcs=[fn1, fn2, ...]` pattern but as a
    single call so the GRPO loop just stores one number per rollout +
    a debug breakdown. Returns (total_reward, per_component_dict).

    Components:
        match_format_exactly         hard binary, big positive
        match_format_approximately   smooth per-tag signal
        check_answer (if gt given)   tiered correctness
        check_numbers (if gt given)  strict numeric double-check
        strict_template_score (opt)  multi-answer / user-marker penalties
    """
    total = 0.0
    bd: dict[str, float | str] = {}

    # 1. Exact format
    exact = match_format_exactly_score(
        completion,
        think_open=think_open, think_close=think_close,
        answer_open=answer_open, answer_close=answer_close,
        reward=exact_format_reward,
    )
    total += exact
    bd["r_exact_format"] = exact

    # 2. Per-tag approximate format
    approx, approx_bd = match_format_approximately_score(
        completion,
        think_open=think_open, think_close=think_close,
        answer_open=answer_open, answer_close=answer_close,
        per_tag_pos=approx_format_pos,
        per_tag_neg=approx_format_neg,
    )
    total += approx
    bd["r_approx_format"] = approx
    bd["tag_counts"] = approx_bd

    # 3. Tiered correctness check
    if answer_check and ground_truth_answer is not None:
        # Allow ground truth in gsm8k "#### N" format too.
        gt = extract_gsm8k_answer(ground_truth_answer) or ground_truth_answer
        ans_score, ans_tier = check_answer_score(
            completion, gt,
            think_open=think_open, think_close=think_close,
            answer_open=answer_open, answer_close=answer_close,
            strict_extraction=strict_extraction,
            ambiguous_penalty=ambiguous_penalty,
        )
        total += ans_score
        bd["r_check_answer"] = ans_score
        bd["check_answer_tier"] = ans_tier

        # 4. Strict number double-check (cheap independent signal)
        num_score, num_tier = check_numbers_score(
            completion, gt,
            answer_open=answer_open, answer_close=answer_close,
            reward_match=number_check_reward,
            penalty_wrong=number_check_penalty,
        )
        total += num_score
        bd["r_check_numbers"] = num_score
        bd["check_numbers_tier"] = num_tier

    # 5. Optional strict template (multi-answer / user-marker bleed)
    if strict_template_weight > 0.0:
        strict_s, strict_bd = strict_template_score(
            completion,
            think_open=think_open, think_close=think_close,
            answer_open=answer_open, answer_close=answer_close,
            user_marker=user_marker,
        )
        strict_r = strict_template_weight * strict_s
        total += strict_r
        bd["r_strict_template"] = strict_r
        bd["strict_template_verdict"] = strict_bd["verdict"]

    # 6. Regurgitation penalty (only when a system prompt was provided)
    # Heavy penalty for the common system-prompt-trained failure mode
    # where the model quotes the system prompt back at the user.
    if system_prompt and regurgitation_weight > 0.0:
        regurg, regurg_bd = regurgitation_penalty(
            system_prompt=system_prompt,
            completion=completion,
            max_penalty=regurgitation_max_penalty,
        )
        regurg_scaled = regurgitation_weight * regurg
        total += regurg_scaled
        bd["r_regurgitation"] = regurg_scaled
        bd["regurgitation_overlap"] = regurg_bd.get("overlap_frac", 0.0)
        bd["regurgitation_verdict"] = regurg_bd.get("verdict", "n/a")

    bd["total_reward"] = total
    return total, bd


def compute_group_advantages(rewards: list[float]) -> list[float]:
    """Compute advantages within a group using GRPO normalization.

    GRPO normalizes rewards within each group to zero mean, unit variance.
    This removes the need for a value function baseline.
    """
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5

    if std < 1e-8:
        return [0.0] * n

    return [(r - mean) / std for r in rewards]
