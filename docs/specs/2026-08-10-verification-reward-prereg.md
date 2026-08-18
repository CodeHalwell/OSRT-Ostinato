# Preregistration — verification reward for GRPO v6b

**Status:** declared BEFORE implementation. Written 2026-08-10, after the wave-2
post-mortem and before any verification-reward code exists.

**Why this document exists.** Three times during the wave-2 investigation a
measurement that looked decisive was overturned by the next one: the scalar
loss-term ratio, then the raw gradient norms, then global clipping. Each error
had the same shape — reading one quantity as if it settled a question that
needed a different quantity. The defence is to name the falsifying measurement
in advance. Every threshold below is fixed now; results are scored against
these numbers, not the reverse.

---

## 0. Boundaries on the evidence this rests on

Two limits keep the existing record honest and must not be quietly widened.

**The +6.5pp wave-2 loss** (soup − step 400 acc_on, paired bootstrap
p=0.012, CI [+1.50, +11.50], discordant 19/6) is established **on the
200-question development panel**. Repeated checkpoint inspection, post-hoc soup
construction and a single training seed all prevent treating p=0.012 as a
confirmatory population claim.

**The 77–83° cross-configuration angle** describes *raw gradients on the cached
step-390 batch*:

```
cos(policy_hist,   policy_corrected)    +0.1263
cos(combined_hist, combined_corrected)  +0.2371
cos(combined_hist, policy_corrected)    +0.2346
```

Clipping is scalar and preserves these directions, but AdamW's per-parameter
preconditioning can transform them, and the angle may differ at earlier
checkpoints. So it establishes that the score-function bug materially changed
the **local** gradient. It does **not** establish that every historical
optimizer update had that angle, and it is not required to carry the stronger
claim that it fully explains the 6.5pp loss. It is sufficient on its own to
reject continuation from wave 2 and restart from the soup.

**Not established, and not to be cited as measured trends:** the OFF-up slope
(p=0.052, halves when step 400 is dropped), the delta collapse (p=0.186), and
`acc_off` overtaking the soup (p=0.602, 25 discordant items).

**The reward-misalignment hypothesis** rests on code inspection (90.9% of
maximum positive reward is the final number; `reasoning_bonus` is gated on
`correct and n_steps >= 2` but counts steps without validating them) and direct
rollout observation (whole groups at steps 250 and 350 scored the full +5.50 on
traces that computed 40 and answered 10). That is a well-motivated hypothesis,
not a measured trend.

---

## 1. Deterministic verifier correctness

Fixture-based, no model involved. **All three must hold exactly; any failure
blocks implementation.**

| # | criterion | threshold |
|---|---|---|
| 1.1 | Accuracy on constructed `valid` / `poisoned` / `corrected` / `malformed` fixtures | **100%** |
| 1.2 | Copying a poisoned step earns positive verification reward | **never** |
| 1.3 | Missing or ambiguous verification output silently passes | **never** — must route to an explicit `unparsed` label |

Fixtures are hand-written and committed alongside the verifier, covering at
minimum: correct verdict on a valid hint; correct rejection of a poisoned hint;
a corrected equation that fixes the poison; a verdict with no supporting step;
two contradictory verdicts; a verdict naming the poison value while rejecting
it (which the earlier `str(wrong_val) in final_answer` substring test would have
penalised at −7.5 — the exact behaviour the reward is meant to teach).

## 2. Shortcut resistance

| # | criterion | threshold |
|---|---|---|
| 2.1 | Valid/poison pairs drawn from the **same** underlying problems | balanced, 50/50 |
| 2.2 | `always-valid` baseline balanced verification accuracy | **≤ 50%** |
| 2.3 | `always-invalid` baseline | **≤ 50%** |
| 2.4 | `ignore-hint` / `copy-hint` baseline | **≤ 50%** |
| 2.5 | Hint validity predictable from style, length, position or formatting metadata | **AUC ≤ 0.55** from a logistic probe on those features alone |

2.5 is the one most easily missed: if poisoned hints are systematically longer,
rounder, or differently punctuated, the model learns a surface cue and 2.2–2.4
still pass. Poison must be localised and style-matched.

## 3. Frozen-policy signal viability

Measured on a cached batch via `grpo_diag_batch` + `grpo_grad_diag`, no training.

| # | criterion | threshold | wave-2 reference |
|---|---|---|---|
| 3.1 | Verification-output parser coverage | **≥ 95%** | n/a |
| 3.2 | Positive advantage on incorrect final answers, after clamping | **exactly 0** | 74/232 wrong rollouts (31.9%) had positive advantage |
| 3.3 | Live rollouts | **≥ 50%** | 87.5% unclamped, 58.6% clamped |
| 3.4 | Zero-variance prompt groups | **≤ 15%** | 7.7% all-wrong |
| 3.5 | Fraction of **final-correct** rollouts whose advantage changes under the verification term | **≥ 20%** | 0% — the current +0.3 bonus cannot discriminate among correct answers |

3.5 is the decisive one. If the verification term does not materially reorder
*correct* rollouts, it is decorative in exactly the way the existing
`reasoning_bonus` is, and the design fails regardless of how well it scores on
groups 1 and 2.

## 4. Gates — sanity wave and confirmation run are DIFFERENT

The power calculation forces this split. Paired-binary SE ≈ `sqrt(d/n)` with
`d` the discordant rate; observed `d = 25/200 = 0.125` for soup vs step 400.

```
non-inferiority power at TRUE gap = 0, one-sided alpha = 0.05, d = 0.125
  n=200 :  M=2pp 0.20   M=3pp 0.33   M=4pp 0.48   M=5pp 0.64
  n=1119:  M=2pp 0.60   M=3pp 0.88   M=4pp 0.98   M=5pp 1.00
```

**The 200-item development panel cannot support a non-inferiority claim at any
sensible margin.** So:

### 4a. Sanity wave (25–50 steps, n=200 development panel) — SCREENING ONLY

| metric | gate | rationale |
|---|---|---|
| `acc_on` | **no catastrophic drop**: upper 95% paired bound of (soup − wave) < **10pp** | detectable at n=200; screens for a wave-1-style collapse, nothing finer |
| Verification accuracy | balanced accuracy > **50%** on ≥100 poison prompts | shortcut floor from group 2 |
| `acc_off` | reported as a **control**, never optimised | |
| Format | **≥ 99%** | wave-2 achieved 100% |
| Truncation | **< 3%** | wave-2 ran 0–6/256 |
| `delta` | **diagnostic only** — must not collapse, must never be the target | it improves when `acc_off` falls, which is exactly what the wave-2 soup did |

Passing 4a licenses a full run. It does **not** license any capability claim.

### 4b. Confirmation run (n=1119, `problem_offset=200`) — CLAIMS

Declared in advance, scored once, on problems never used for selection.

| metric | gate | power |
|---|---|---|
| `acc_on` non-inferiority vs the SFT soup's 20.0% | margin **M = 3pp**, one-sided 95%: upper bound of (soup − v6b) < 3pp | **0.88** at true gap 0 |
| Verification accuracy | ≥ **30%** poison-rejection with paired 95% CI excluding zero | CI lower ≈ 21pp at n=100, so power is not binding — 30% is a **usefulness** threshold, not a detectability one |
| `acc_off` | control | |
| `delta` | diagnostic | |

**Honest limitation of the M=3pp gate:** power is 0.88 only if v6b genuinely
*equals* the soup. If the true gap is 1pp power falls to 0.60; at 2pp it is
0.24. So failing this gate will not distinguish "slightly worse" from "much
worse", and M=2pp is not achievable (0.60 power) on the available problems.
A tighter margin needs a larger evaluation set than GSM8K test provides.

---

## 5. Declared changes that are not hypotheses

Landing regardless of the above, recorded so they are not later mistaken for
findings:

- **Strict extraction is lossless on correctness** at step 390 — 248/256
  agreement, 8 disagreements all `no_answer_block`, **zero correctness flips**
  (loose 24/256, strict 24/256). It is **not** reward-neutral: those 8 (3.1%)
  move from `wrong_far_off` (−2.0) to `no_extraction` (−2.5). Declared as a
  deliberate change, not discovered afterwards.
- **Empty gold is a hard loader error**, not a filtered row.
- **`delta` is demoted** from primary to diagnostic.
- Metric hierarchy: primary `acc_on`; co-primary verification accuracy; control
  `acc_off`; diagnostic `delta`.

---

# Amendment 1 — 2026-08-10, still before implementation

Appended rather than edited: silently rewriting a preregistration defeats its
purpose. Everything above stands unless contradicted here.

## A1.1 Reward form — bounded, correctness-gated, lexicographic in effect

```
R = R_correctness + R_format + 1[final correct] * lambda * V

V = +1   correct verdict WITH deterministically verified support/correction
V =  0   verdict present but incomplete or unsupported
V = -1   wrong, contradictory, copied-poison, ambiguous, or unparsed
```

- `V` contributes **exactly zero** when the final answer is wrong, so
  verification is never an independent route to reward. The correctness clamp
  still prevents positive policy advantage on incorrect answers.
- The existing **syntactic `reasoning_bonus` is removed on verification
  prompts** — it counts steps without validating them, which is the defect this
  term replaces.

**Why a modest weight suffices.** Among final-correct rollouts the +5.0 is
*constant* and vanishes under group centring, so `V` only has to reorder within
that subset. It does not need to compete with +5.0. In mixed groups the
correct/wrong gap stays dominant: even at `lambda=2.0`, correct-with-failed-
verification scores `5.0 + 0.2 - 2.0 = 3.2` against the best wrong tier's
`-0.5 + 0.2 = -0.3`.

## A1.2 Criterion 3.5 superseded — materiality required

The original 3.5 ("advantage changes for >=20% of final-correct rollouts") is
passable by an arbitrarily small floating-point perturbation. **Replaced by:**

> At least **20%** of final-correct rollouts change **standardised advantage by
> >= 0.25**.

## A1.3 lambda grid REVISED, from a pre-computation against A1.2

Simulating `V` uniform on {-1,0,+1} against the observed reward distribution
(G=16, ~15% exact, wrong tiers -0.30/-1.80/-2.45):

```
lambda   median |d adv|   frac >=0.25   frac >=0.10
  0.25       0.045           0.0%         9.1%
  0.50       0.088           0.3%        43.0%
  1.00       0.182          30.7%        68.7%
  2.00       0.361          59.7%        80.1%
```

`lambda` in {0.25, 0.5} **cannot pass A1.2** and are eliminated before any
audit. The declared grid is therefore **{1.0, 1.5, 2.0}**, cap 2.0, select the
smallest passing value; if none pass, **the verifier design fails** rather than
the weight being raised further.

Structural note: the group standard deviation (~2.43) is dominated by the
correct/wrong gap, so differences among correct rollouts are small relative to
it. That is a headwind for the intended reordering and the reason small weights
are non-viable. Standardising advantages *within* the correct subset would
change GRPO's advantage definition materially and is **out of scope** here — if
the {1.0, 1.5, 2.0} grid fails, that is a separate design decision, not a
fallback to be taken silently.

## A1.4 Confirmation must be BALANCED, not poison-only

Poison rejection alone is farmed by "always invalid". Confirmation requires
**both**:

- Balanced verification accuracy, **lower 95% bound > 50%**.
- Poison rejection **plus correct repair >= 30%**, interval reported.

This supersedes the single poison-rejection row in section 4b.

## A1.5 Calibration and audit partitions are DISJOINT

`lambda` must not be tuned and tested on the same frozen batch. Split paired
problems deterministically **by question hash**:

- **Calibration partition** — select the smallest passing `lambda`.
- **Locked audit partition** — evaluate criteria 1-3 **once**, after `lambda` is
  fixed.

## A1.6 The semantic surface stays deliberately tiny

Rewarded: a **structured verdict** plus a **canonical corrected equation or
value**, checked against dataset metadata. NOT rewarded: free-form eloquence,
step count, keyword presence, or any LLM judge. Every one of those is a surface
the policy can farm, and step count is the specific failure already on record.

## A1.7 Hint conditioning is a distribution shift, and the transfer test is
unhinted

Hinted GRPO trains `pi(y | q, h)` while the shipped model receives only `q`. A
verifier can pass every anti-hacking fixture and still teach behaviour that
disappears when the hint is absent.

- Hinted verification prompts are a **declared mixture stratum**, NOT a
  replacement for unhinted math prompts.
- **Unhinted `acc_on` is the transfer test** and remains the primary metric.
- Report hinted and unhinted `acc_on` separately; a hinted-only gain is not a
  capability gain.

---

# Amendment 2 — 2026-08-10, still before verifier implementation

## A2.1 CORRECTION: prior hint design DOES exist, and it is sharper than A1.7

Amendment 1 asserted "no prior hint-design contract in the repo". **That was
wrong**, and the search was at fault: it grepped `poison|obfuscated
untruth|hint injection|suggested answer` and so missed a section titled
**"Hint-augmented GRPO"** entirely.

`docs/specs/2026-07-26-precision-and-sft-objective.md` §6 already contains
graded hints, adaptive difficulty selection, the hinted/unhinted policy
mismatch, and a recommendation toward **filtered context distillation** (§9.6).
What does *not* exist is a specification for **poisoned** hints, structured
verification output, or semantic scoring — so the gated verifier is greenfield
*semantically*, but the older transfer analysis applies in full and supersedes
A1.7's weaker framing.

**§6.5 is sharper than A1.7 in one specific way.** A1.7 proposed hinted prompts
as a "declared mixture stratum". §6 shows that is **not sufficient**:

> hinted prompts are precisely the ones producing nonzero advantage — so a
> disproportionate share of the actual gradient teaches the hint-conditioned
> policy. The x% mixing does **not** save this, because the unhinted remainder
> is mostly the zero-gradient set. You would be optimising the mode you don't
> ship.

So mixture fractions do not control the gradient share; **advantage mass does**.
Any hinted design must therefore report the *fraction of non-zero advantage*
arising from hinted prompts, not merely the prompt mixture ratio. Added as a
frozen-audit requirement:

| # | criterion | threshold |
|---|---|---|
| 3.6 | Share of total non-zero-advantage mass arising from **hinted** prompts | report; **≤ 60%** |

§6 also names **Option B** (sample with hint, score without — STaR-style
rationalisation) and two concrete breakages. Both transfer to
`osrt/grpo_train.py`: `Rollout.prompt_len` is computed from the **hinted**
prompt and slices `shift_logits` in `_seq_logprobs`, so scoring against an
unhinted prefix would misalign the labels; and the loss is a direct policy
gradient with no importance ratio, so an off-policy prefix change is
uncorrected. Option B is **out of scope** here and must not be attempted
without fixing both.

§6's own status is "unranked pending measurement" — its ROI case was retracted
because `correctness_partial_credit` had removed the variance trap that
motivated it. **Note the interaction:** those proximity tiers were removed on
2026-08-10, so the retraction's stated premise no longer holds. It does not
resurrect the variance argument, though — measured live rate is 87.5%
unclamped / 58.6% clamped and all-wrong groups are 7.7%, so variance is not the
binding constraint. Our purpose for hints is **chain-dependent reward signal**,
a different objective from §6's sample efficiency.

## A2.2 lambda: {1.0, 2.0} are candidates; {0.25, 0.5} are NEGATIVE CONTROLS

A1.3 called the grid {1.0, 1.5, 2.0}. Corrected: the candidates are
**{1.0, 2.0}** — two, not three. The eliminated values are still **run in the
frozen audit as negative controls**, to confirm the simulation predicted their
failure correctly rather than assuming it.

**The pre-screen is a PREDICTION, not a result.** It used wave-2's *unhinted*
reward distribution with *simulated* uniform `V`. A real hint-conditioned batch
may have different group variance and a different verification-outcome
distribution, either of which moves the materiality figures. So `lambda = 1.0`
is the **predicted** smallest pass; it is not established until the audit.

Corrected arithmetic: with the +0.3 step bonus removed on verification prompts,
the maximum becomes `5.0 + 0.2 + 1.0 = 6.2`, so `lambda = 1.0` contributes
**16.1%** of maximum reward, not the 18% previously stated.

## A2.3 Criterion 3.5 is measured on the AUDIT partition

A1.2 set the materiality bar; A1.5 split calibration from audit. Joining them
explicitly, because the gap allowed choosing `lambda` and declaring materiality
on the same examples:

> Criterion 3.5 (≥20% of final-correct rollouts changing standardised advantage
> by ≥0.25) is evaluated **on the locked audit partition only**, after `lambda`
> is fixed on the calibration partition.

## A2.4 Write the CONTRACT before the fixtures

Criterion 1.1's fixtures must *instantiate* a specification, not silently
*become* it. The next session starts with a contract, not test cases:

1. Exact output **grammar** for the verification block.
2. **Canonicalisation** rules for numbers and equations.
3. **Precedence** for missing, duplicate, ambiguous and contradictory blocks.
4. A **truth table** over (final correctness) x (hint validity) x (verdict) x
   (supporting step) x (correction present) x (poison copied), with the
   resulting `V` in {-1, 0, +1} for **every row**.
5. Fixtures covering every truth-table row, plus adversarial formatting
   variants.

Two questions the contract must answer explicitly, currently undefined:

- What does **"verified support"** mean for a *valid* hint — is agreeing with a
  correct hint enough, or must the model independently derive it?
- Does a **correct `invalid` verdict without a correct repair** earn `0` or
  `-1`?

## A2.5 Landed now (declared, not hypotheses)

Implemented in this session, ahead of the verifier:

- **`word_problem_verify_1shot`** added to `REASONING_ON` and set as
  `GRPOv6Config.system_persona`. Every pre-existing 1-shot persona demonstrates
  *format* with trivial sums (`12 + 8x3`, `25-9`, `60x2`); none addresses the
  observed failures. This one demonstrates name-the-unknown -> equation ->
  solve -> **substitute back** -> answer-consistent-with-working, mirroring the
  step-190 (`250/3200 = 0.5`), step-220 (`4025.25/0.45` giving three answers)
  and step-250 (computed 40, answered 10) rollouts.
- **`few_shot_echo_penalty = -3.0`**, fired by a 12-word verbatim run against
  the persona's exemplar (special tags stripped from both sides, since both
  necessarily contain them). Ordering achieved: copy-and-correct +2.20 vs +5.20
  for real work; copy-and-wrong -3.3, below the worst non-copying tier (-2.3).
  So echoing is strictly worse than both solving and failing honestly.
  **Measured 0/256 false positives** on real step-390 rollouts at n=8..16, and
  a test asserts the same *pattern* with different numbers is NOT penalised.
- **Eval personas pinned by name** (`DEFAULT_EVAL_ON/OFF`). The historical
  default `Random(0).choice(pool)` depends on pool LENGTH, so adding a persona
  can silently rebase every recorded number. A test now asserts the pinned
  names and that the accident still holds.

**This is itself a distribution shift**, and A2.1 applies to it: the exemplar
makes training `pi(y | q, exemplar)` while all eval personas carry no exemplar.
Unexemplified `acc_on` is therefore the transfer test — and per A2.1, the
quantity to watch is the share of non-zero advantage arising from exemplified
prompts, not the prompt mixture.

---

# Amendment 3 — 2026-08-10. Mixed stratum DECIDED; launch config reverted

## A3.1 The all-exemplar config violated 3.6 by construction — reverted

`GRPOv6Config.system_persona = "word_problem_verify_1shot"` made **every**
prompt exemplified, so the exemplified share of advantage mass is necessarily
**100%** and criterion 3.6 (≤60%) fails by construction. That is a
ready-to-launch configuration silently breaking its own preregistration.

**Reverted to `minimal_format`.** The persona and the echo penalty stay
committed and tested; only the launch default returns to unexemplified until
mixed-stratum support passes the frozen audit.

## A3.2 Matched personas, fixed 50/50 PROMPT-LEVEL mixture

- **`word_problem_verify_0shot`** — identical instructions, no demonstration.
- **`word_problem_verify_1shot`** — the committed exemplar.

Both are composed from a shared `_VERIFY_INSTRUCTIONS` constant so they cannot
drift apart; a test asserts the 1-shot text starts with the 0-shot text. If the
instructions differed, a 0-shot vs 1-shot comparison would confound "has an
exemplar" with "has different instructions".

**Training:** fixed **50/50 prompt-level** mixture, every rollout **group
persona-consistent**. With 16 prompts that is **eight unique 0-shot questions
and eight unique 1-shot questions** — NOT eight questions duplicated across
both modes, which would halve problem diversity. The mixture is **static**;
it is never adjusted during training.

**Frozen audit:** the *same* underlying questions under *both* personas, so the
comparison is paired.

## A3.3 Measurements, all AFTER clamping

| quantity | definition |
|---|---|
| Exemplified advantage share | `sum(abs(adv_exemplified)) / sum(abs(adv_all))` — **≤ 60%** (criterion 3.6) |
| Policy-gradient norm | reported **per stratum** |
| Live rate | reported **per stratum** |
| Zero-variance group rate | reported **per stratum** |
| Transfer result | **unexemplified `acc_on`** |

If 50/50 exceeds the 60% cap, the mixture is re-chosen on the **calibration**
partition; the **locked audit** partition must then satisfy the cap. Never
tuned on the audit partition.

## A3.4 Implementation blocker: per-prompt persona metadata

`generate_rollouts` (`osrt/grpo_train.py`) looks the exemplar up **globally**
from `cfg.system_persona`, so it assumes one persona for the whole batch. A
mixed batch cannot be scored correctly: unexemplified rollouts would be charged
against an exemplar they never saw, and the penalty's measured 0/256
false-positive rate does not cover that case. The persona/exemplar identifier
must travel **with each prompt or rollout**. A loud comment now marks the
constraint at the lookup site; the code change belongs with the contract work.

## A3.5 Two safeguards added to the fresh-session contract

**Numeric anchoring.** The n-gram penalty catches copied *prose*, not a copied
*answer*. `VERIFY_EXEMPLAR_ANCHORS = ("500", "625", "1.25", "25%")` are values
appearing only in the 1-shot prompt; compare 0-shot vs 1-shot emission of them
on problems whose gold does **not** contain them. A test asserts the asymmetry
that makes the comparison possible.

**Explicit exemplar storage.** The registry was derived by
`text.split("Example", 1)[1]`, which had two defects, both now fixed by named
constants: it swallowed the trailing *"Do not repeat the example"* INSTRUCTION
into the penalisable span, so a model quoting its own instructions would have
been penalised; and it silently registered any future persona containing the
word "Example". `FEW_SHOT_EXEMPLARS` is now explicit and registers **only** the
personas GRPO trains under — the six legacy 1-shot personas are deliberately
absent, since nothing trains under them with the penalty active and registering
them would extend the penalty's surface with no measured false-positive rate.

## A3.6 A guard fired for real, and forced a fix

Adding `word_problem_verify_0shot` took `REASONING_ON` from 13 to 14 entries and
moved `Random(0).choice` from **`instruction_strict` to `general_default`** —
which would have silently rebased every recorded `acc_on`/`acc_off` number.

`run_reasoning_eval` now resolves personas **by name** from
`DEFAULT_EVAL_ON`/`DEFAULT_EVAL_OFF` rather than sampling, making the historical
panel reproducible however the pools grow. A test asserts the sampled value has
in fact diverged from the pin, so anyone reintroducing sampling fails loudly.

## A3.7 Next-session order

1. Verification **grammar** and **truth table** (A2.4).
2. Matched 0-shot/1-shot personas *(done)* and **per-prompt persona metadata**
   (A3.4).
3. Only then fixtures, and only then reward code.

---

# Amendment 4 — 2026-08-10. Verification semantics SETTLED

A2.4 listed two undefined questions and deferred them to the contract session.
Both are now decided, so the next session starts from a specification rather
than an open design.

## A4.1 "Verified support" = a non-identical, machine-checkable witness

Agreement is not verification, and repetition is not verification. For a
**valid** hint the model must supply a witness that is *independent of the hint
itself*, in one of three forms:

1. **Substitution** — put the value back into the stated relation and show both
   sides equal.
2. **Inverse computation** — derive the hinted quantity by the reverse operation.
3. **Equality of independently evaluated sides** — evaluate each side separately
   and show they match.

**Restating the hint earns V=0, not V=+1.** That is the load-bearing rule: it is
what stops "the hint says 1.25C = 625, I agree" from collecting the same reward
as actually checking it, and it is the same defect as the current
`reasoning_bonus` counting steps without validating them.

## A4.2 The truth table

| result | V |
|---|---|
| Correct verdict **+ verified witness/repair** | **+1** |
| Correct verdict, **missing or unverifiable** support/repair | **0** |
| **Wrong, contradictory, or ambiguous** verdict | **−1** |

Symmetric by construction, and note what is NOT −1: a **correct `invalid`
verdict without a valid repair earns 0.** It detected the problem but supplied
no usable reasoning, which is worth nothing rather than worth punishing.
`V=−1` is reserved for wrong verdicts, self-contradiction, malformed or
ambiguous output, and **adopting** the poisoned claim.

Final-answer correctness still **gates** the term (A1.1), so `V` moves reward
only when the submitted answer is correct, and the clamp still prevents positive
policy advantage on wrong answers.

## A4.3 "Copying" is structural, not lexical

For a poisoned hint, **quoting the bad equation while explicitly rejecting it
must remain allowed** — that is precisely the target behaviour. "Copying" means
**adopting the claim as support**, not mentioning it diagnostically.

This must be represented **structurally** (which claim occupies the support
role), never by substring matching. The concrete failure this avoids is already
on record: the reference DOUBT implementation's `str(wrong_val) in final_answer`
test assigns **−7.5** to `<SOLUTION>12 (not 20)</SOLUTION>` — maximum penalty
for solving correctly *and* naming what it rejected. Substring matching cannot
distinguish adoption from citation, so it is excluded by contract.

## A4.4 Initial scope boundary

**Support verification is limited to normalised arithmetic and equation claims
with canonical metadata.** Free-form strategy verification is **out of scope**
until the deterministic contract proves itself. This keeps the rewarded semantic
surface as small as A1.6 requires, and keeps every V assignment decidable
without a judge model.
