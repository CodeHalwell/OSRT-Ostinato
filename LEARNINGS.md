# LEARNINGS.md — Everything we learned from nano-osrt v5 (363M)

**Scope:** the full lineage of v5 nano-osrt training, from pretrain →
SFT → GRPO → MOPD distillation → grpo_multi → system-prompt
discovery. Every failure, every fix, every metric that surprised us.

**Audience:** future-you when designing OSRT-600M (or v7+), so we don't
repeat any of the v5 mistakes.

**Companion docs:**
- [`README.md`](README.md) — the v6 design that bakes in these lessons
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — v6 technical spec
- [`RESEARCH.md`](RESEARCH.md) — external research that shaped v6

---

## 0. The meta-lesson

> **Most of our pain came from discovering problems late.**

Loop collapse at 4 stages in. Reward hacking after a full GRPO run.
Missing system-prompt support after a year of training. Every one of
these was avoidable with measurement-first design.

**OSRT-600M's defining principle:** measurement before architecture.
Run benchmarks at every checkpoint, log per-loop CE and per-env
hit-rates from day 1, build the OOD probe BEFORE the training run
that needs it.

---

## 1. Architecture lessons

### 1.1 Loop collapse — the biggest single discovery

**The problem:**
- v5 pretraining used 3 physical blocks × 6 recursive loops = 18
  effective layers
- After pretrain + SFT + GRPO + extend → probed loop CE contributions
  via Test 3 ablation
- Found loop 5 was doing **~90% of the cross-entropy reduction**
- Earlier loops were contributing almost nothing — effectively wasted
  compute
- This had been happening for the entire training history; we just
  hadn't measured it

**The fix:** added two architectural knobs

1. **`aux_loop_loss_weight`** (0.05) — apply LM-head cross-entropy at
   every intermediate loop output, not just the final loop. Forces
   every loop to produce a meaningful prediction. Made loops 1-5
   start contributing.
2. **`loop_dropout_prob`** (0.10) — randomly truncate the loop chain
   to a random K ∈ [3, 6] during training. The model can't depend on
   "always 6 loops"; learns to make each loop useful in isolation.

**Result:** by `loopfix_v2`, all 6 loops contributing roughly equally
(Test 3 ablation showed monotonic improvement).

**Why we missed it initially:** the model trained "successfully"
(loss decreased, benchmarks moved) without us ever asking *which loops
were doing the work*. The recursive trick LOOKED like it was working
because outputs were coherent — but only one loop was actually
running.

**Lesson for v6:** print per-loop CE losses during training from step
1. Test 3 ablation auto-pipelined per checkpoint. NEVER trust that
recursion is being used.

### 1.2 The architecture-fix knobs were preserved by accident

When we tightened `aux_loop_loss_weight` during GRPO (0.03 → 0.10) to
preserve depth use, the loop utilisation actually held. But this was
a side effect — we didn't design for it. v6 should make per-loop aux
losses a permanent fixture across all stages, not a fix that's added
when probe shows a problem.

### 1.3 HRA injection just works — but we trained the base too long

HRA (Hyperspherical Reparameterised Adapters) at rank 256 added ~86M
trainable params to the 363M base. Worked great for SFT. But during
GRPO we trained BOTH base and HRA — that was the cause of the
4/6 → 2/6 regression at GRPO step 150. v6 should use HRA-only for all
RL stages.

---

## 2. Tokenizer / format lessons

### 2.1 System prompts were NEVER trained — the biggest miss

**Discovery (after 12+ months of training):** the tokenizer had
`<|system|>` as token id 13, but the model never saw it in any
training context. SFT/MOPD/GRPO all used bare
`<|user|>{q}<|assistant|>{a}`. Teacher rollouts from Gemini API came
without system roles, and we never thought to inject them.

**Symptom:** at inference time, trying to use a system prompt put the
model fully out-of-distribution. Generated malformed output: no
`<|/think|>` close, then degenerate `<|answer|>...<|/answer|>` loops.
Scored 0/12 on OOD probe.

**Fix (mid-v5):** collected 10K system-prompt rollouts from
OpenHermes-2.5 (filtered to system-bearing rows), built `system_sft`
Modal stage to retrofit system-prompt support on top of `grpo_v2_50`
ckpt.

**v6 plan:** bake `<|system|>` into pretraining text. ~10% of
pretrain mix should be chat-formatted with full
`<|system|>...<|user|>...<|assistant|>...` chains. Don't retrofit.

### 2.2 The chat-template format used at INFERENCE wasn't trained in PRETRAINING

`<|think|>/<|/think|>/<|answer|>/<|/answer|>` were reserved tokens
the model saw for the first time at SFT. MOPD then had to teach
format from scratch — wasteful. If pretraining had seen these tags
10K+ times in natural-looking text, MOPD would converge in 100 steps
instead of 1000.

**v6 plan:** include reasoning-formatted text in pretrain mix (~5%)
so the chat template tokens are natural by SFT time.

### 2.3 Open-only role tags caused multi-turn ambiguity

v5 used `<|user|>...<|assistant|>...<|user|>...` (no close tags). At
inference, the model couldn't cleanly signal "end of this turn" — it
just had to emit the next role's open tag. For multi-turn or tool-use,
this is fragile.

**v6 plan:** add `<|end_turn|>` as a single token (ChatML style),
explicitly close each turn.

### 2.4 Vocab choice — 32K was tight at this scale

v5's 32K vocab handled English fine but felt cramped for code
tokenization (long identifiers, function names). v6 moved to 65,536
matching LFM2-700M. Same embedding tax % (~14-17%), better coverage.

---

## 3. Training pipeline lessons

### 3.1 The lineage that actually worked

```
v3/v4 pretrain (17K steps, FineWeb-Edu / CodeParrot / Wikipedia, 1024 ctx)
  ↓ (architectural overhaul for v5)
v5 pretrain (3000 steps from cold start)
  ↓
v5 SFT base → SFT-long → SFT-ultralong (8192 ctx, balanced mix)
  ↓
GRPO (700 steps gsm8k verifiable rewards) — 7-9% gsm8k accuracy
  ↓
pretrain_extend (continued mid-training on Nemotron-CC-Math)
  ↓
sft_refresh + sft_math (format anchor + math polish)
  ↓
pretrain_extend2 (8100 steps, 9-stream mid-training)
  ↓ — DISCOVERED LOOP COLLAPSE HERE via probe
loop_fix (1500 steps with aux LM-head loss)
  ↓
loop_fix_v2 (1500 steps with stacked fixes — depth + curriculum + per-loop)
  ↓
pretrain_extend3 (3000 steps, first mid-training with WORKING depth)
  ↓
MOPD (1000 steps, distillation from 13K Gemini + DeepSeek rollouts)
  ↓
grpo_multi v1 (multi-env GRPO — REGRESSED 4/6 → 2/6)
  ↓
grpo_v2 (HRA-only + anti-hacking — 30 steps tested, math.exact=5%)
  ↓
system_sft (in-progress — system-prompt retrofit)
```

15+ training stages. ~$2-5K total Modal cost. ~12 months calendar.

### 3.2 Multi-stage data curation worked, single-stage didn't

`pretrain_extend2` was an 8100-step mid-training run with **9 streams**
(OpenR1-Math, OpenMath-Reasoning, Open-Web-Math, Open-Thoughts,
Dolmino-FLAN, Dolmino-PES2O, UltraChat, Cosmopedia v2,
FineWeb-Edu). Mid-trained `loopfixv2` showed substantial gains over
single-source extend (extend1 was Nemotron-CC-Math only).

**Lesson:** for small models trained on tight budget, multi-stage
curriculum > single high-quality source.

### 3.3 Multi-teacher MOPD beat single-teacher

Initial MOPD used Gemini 3.5 Flash only. Expensive ($1.50 input /
$9.00 output per 1M tokens) and slow. After 13K Gemini rollouts:

Switched to DeepSeek v4-flash direct API: $0.14/$0.28 per 1M tokens,
2500 concurrent allowed, ~13-15s/rollout. Got ~7700 more rollouts in
15 minutes for ~$4.

**Lesson:** mixed-teacher distillation works. DeepSeek + Gemini gave
diverse rollouts; quality didn't suffer. Never go back to single
teacher.

### 3.4 Format checkpoint chain matters

Going from format-loose to format-strict in stages:
- pretrain: arbitrary format, no chat structure
- sft_refresh: short format-anchor SFT on extend_final (~500 steps,
  peak LR 5e-6, no tool_calling)
- sft_math: math-only polish (~1000 steps, peak LR 3e-6,
  GSM8K + Orca + MathInstruct + NuminaMath)

Each stage gradually tightened format. By the time we hit GRPO,
the model knew the `<|think|>...<|answer|>` structure cold.

**Lesson:** don't try to teach format and capability simultaneously.
Stage them.

### 3.5 Cross-workspace checkpoint migration is painful

We trained across `danielhalwell`, `gradio-winter-hack`,
`codhe-hugging-mcp`, and `build-small` Modal workspaces as we
exhausted credits on each. Each migration required:

1. Download ckpt locally (~3GB)
2. Switch workspace
3. Upload to new workspace's volume
4. Verify integrity (race conditions with concurrent uploads burned
   us once — three `modal volume put` processes raced and corrupted
   the file)

**Lesson:** budget per workspace; don't split a run across workspaces
if avoidable. If you must migrate, do it ONCE per training stage,
not multiple times.

### 3.6 Sanity tests caught more bugs than expected

Every stage we added had a `*_sanity` version (30-50 steps, no compile,
no wandb). These caught:
- OOM bugs (loop_fix sanity hit OOM at batch=8 because aux losses
  materialized 5 extra logit tensors)
- LR schedule bugs (lr_anchor_step mismatch starved gradient)
- Resume bugs (resume_step counter starting at 0 for fresh prefix)
- Per-loop logging bugs (model was in eval mode, aux losses not
  populated)

**Lesson:** ALWAYS run sanity before full launch. The 30 steps cost
~$1-3 and save $20-100 in failed runs.

---

## 4. GRPO lessons (the biggest source of pain)

### 4.1 Loose answer extraction is the dominant reward hack

v5 GRPO used `extract_numeric_answer(text)` which returns the LAST
number found in the answer block. Naive. The model learned to dump
multiple candidate numbers ("I tried 50, then 32, but actually 18")
and the loose extractor scored each rollout as +3 reward whenever
the last number happened to match GT.

**Detection:** built `extract_numeric_answer_strict` in post-mortem.
Returns answer only at high confidence:
- `single_number`: answer block contains exactly ONE number → answer
- `boxed`: `**N**` / `` `N` `` / `\boxed{N}` → answer  
- `concluding`: "= N" / "answer is N" at end → answer
- `ambiguous`: multiple unmarked numbers → return None, score `-0.5`

Tested against actual hack patterns — clean +9.50 vs hack +6.00 in
strict mode (∆ = -3.50). Loose mode was identical.

**v6 lesson:** strict extraction default ON. Use loose only for
inference scoring where partial credit is OK.

### 4.2 Reward EMA climbing ≠ capability improving

The two GRPO step-75→150 runs we did showed REWARD EMA CLIMBING
across all envs while inference accuracy REGRESSED from 4/6 to 2/6.
Per-env breakdown after the fact:
- math.exact_rate: declining (despite math reward EMA up)
- ifeval.constraint_rate: real gains
- mbpp.all_pass_rate: **0/180 — pure format hacking**

mbpp was the worst offender: model wrote format-perfect code that
NEVER passed tests, but format rewards alone gave +4.5 EMA.

**v6 lesson:** every env must log a CAPABILITY metric (exact rate,
test pass rate) IN ADDITION to reward EMA. Without this, you're
blind to hacking.

### 4.3 Training base weights during RL caused the regression

Hypothesis testing in v5: ran both "tighter knobs" and "original knobs"
GRPO from step_75 to step_150. BOTH regressed to 2/6 inference.
Conclusion: it wasn't the knobs. It was that we were training the
BASE weights during RL.

Switched to HRA-only GRPO in `grpo_v2`. After 30 steps:
- KL stayed at 0.001-0.02 (was 0.05-0.20 in non-HRA-only)
- Base weights frozen → MOPD capability anchor structurally preserved
- math.exact climbed 0% → 5% genuinely (real signal, not hack)
- Diagnosed: model gets math format right but can't recall arithmetic
  facts (frozen base = bounded knowledge)

**v6 lesson:** HRA-only RL is the LoRA-of-RL standard. Always use it.
Base-weight RL is the wrong default.

### 4.4 OOD probe should be IN the training loop, not post-hoc

In v5 we built the OOD probe only after seeing the regression. By
then, multiple expensive runs were already wasted. v6 bakes a
12-prompt OOD probe at T=0.3 into the training loop every 25 steps,
with auto-stop if OOD score drops 2× in a row.

The 12 prompts span diverse styles NOT in training distribution
(direct arithmetic, comparison, conversion, etc). Validated in v5
post-mortem; works.

### 4.5 The `system_sft` discovery + few-shot inference fail

Story arc:
1. After grpo_v2 step 50, ran inference at T=0.7 → 2/6
2. Re-ran at T=0.3 → 6/12 OOD (matched baseline)
3. Tried adding `<|system|>...<|user|>...` few-shot prefix → 0/12!
   Model generated malformed output (no `<|/think|>`, then degenerate
   answer loop)
4. Discovered: model has NEVER seen `<|system|>` in training
5. Built `system_sft` Modal stage + OpenHermes data collection

**Key insight:** the inference test at T=0.7 was misleading us. The
real OOD probe at T=0.3 gave a much better picture. v6 inference
benchmarks should ALL use T=0.3.

### 4.6 Reward stack interactions are subtle

Even with all the fixes (strict extraction, OOD probe, hit-rate
logging, HRA-only), v6 sanity showed:
- math.exact hit-rate climbing genuinely
- ifeval.constraints climbing (real instruction-following gains)
- mbpp.all_pass STUCK at 0% — pure format hacking persists

mbpp env doesn't work at our model scale — model can produce
format-perfect code but can't write working code from scratch.
v6 should DROP mbpp env entirely and lean on tool-use for code.

### 4.7 GRPO converges fast at small scale — don't over-train

v5 GRPO sanity (30 steps) showed math reward climbing +1.4 → +4.5.
Full 150 steps usually plateaued by step 75. The math-only GRPO was
500-800 steps total, with most learning in the first 200.

**v6 lesson:** GRPO is a short stage at small model scale. 100-300
steps per env is enough. Don't budget $100 here when $20 will do.

### 4.8 Sandboxed code exec security matters even on Modal

The mbpp_test_reward function originally exec'd model-generated code
with `subprocess.run(["python3", "-c", code])`. Security review
flagged: model code could read HF_TOKEN, WANDB_API_KEY, etc. from
environment.

Hardened: stripped env (`{PATH, LC_ALL, LANG}` only), tempdir cwd
(cleaned up in finally), `start_new_session=True`, `os.killpg` on
timeout, output capped at 64KB, absolute python path, default OFF
behind `allow_unsafe_exec=True` flag.

**v6 lesson:** model code = untrusted code. Always sandbox properly,
even on supposedly-isolated Modal containers.

---

## 5. Distillation (MOPD) lessons

### 5.1 OpenAI-format rollout JSONL was the right schema

Schema:
```
{
  "prompt": "...",        # user question
  "thinking": "...",      # optional teacher reasoning
  "response": "...",      # teacher final answer
  "source": "math|reasoning|chat|code|science",
  "teacher": "deepseek-v4-flash|gemini-3.5-flash",
  "id": "<source>:<row_id>"
}
```

Resume-safe via id-set, deduplicated. Worked across 13K rollouts
without issues.

### 5.2 Concurrency 2000 worked; lower didn't

For DeepSeek v4-flash collection: tried concurrency=8, 50, 100 — slow
(hours for 5K rollouts). Pushed to 2000 with bumped RLIMIT_NOFILE +
ThreadPoolExecutor sized to concurrency+50. Got 7700 rollouts in
15 minutes at $4.29.

**v6 lesson:** for API-based rollout collection, max out concurrency.
Most APIs handle 2000+ requests/sec; the bottleneck is usually local
file descriptors.

### 5.3 Math + non-math mix mattered for chat capability

Initial MOPD was math-heavy (60% math, 30% reasoning, 10% chat).
Result: model could do format-perfect math but bled the math style
into chat ("Step 1: ...\nStep 2: ...\nAnswer: ...") for casual
prompts.

Re-collected with diverse mix: math 25%, reasoning 25%, chat 25%,
code 15%, science 10%. Final MOPD held format better across domains.

**v6 lesson:** balance distillation data even at small scale. Single-
domain MOPD trains a one-trick pony.

### 5.4 Distillation cost was the dominant rollout expense

13K rollouts at mixed teachers:
- Gemini 3.5 Flash (5,440 rollouts): ~$8 input + ~$48 output (rough)
- DeepSeek v4-flash (7,924 rollouts): ~$1 input + ~$2 output
- Net: ~$60 for the 13K corpus

For v6 on $280 budget: stick with DeepSeek v4-flash as primary, top
up with Gemini only for the most complex prompts. Should fit in $30
for ~5K diverse rollouts.

---

## 6. Tooling / infrastructure lessons

### 6.1 Modal `.spawn()` vs `.remote()` for long runs

When using `modal run --detach`, use `.spawn()` not `.remote()` to
get a function call you can wait on independently. This was a
documented pattern in memory but took us a while to figure out.

### 6.2 W&B integration retrofit was painful — bake in early

We started without W&B and added it later. Migration was annoying
(per-stage config setup, multiple runs in same project, etc).

**v6 lesson:** W&B from day 1, run-id stamped into every checkpoint
filename.

### 6.3 Modal volume `put` race conditions

Three concurrent `modal volume put` processes raced on the same file
during MOPD collection. Result: corrupted upload, had to redo.

**Lesson:** serialize volume uploads. Add `-f` flag explicitly only
when overwriting on purpose.

### 6.4 23h rescue checkpoint convention

Every long stage had:
- Regular ckpt at every `ckpt_interval` (e.g. 100 steps)
- 23h-rescue ckpt at `osrt_v5_{stage}_rescue_step_{N}.pt`
- Resume scan ranks both, prefers rescue on ties

Saved us from Modal's 24h container timeout multiple times.

### 6.5 Modal volume download corruption

Killed `modal volume get` mid-stream (the watcher script's `wait`
got interrupted). File looked complete by size but was actually
truncated mid-write. Re-download fixed.

**Lesson:** always check file integrity (re-load via torch.load) after
download. Don't trust file size alone.

---

## 7. Budget / cost lessons

### 7.1 Actual costs per stage (v5 lineage)

Rough Modal H100 spend at ~$4/hr:

| stage | cost |
|---|---|
| Initial v3/v4 pretrain (deprecated) | ~$2K total |
| v5 pretrain | ~$500 |
| SFT base / long / ultralong | ~$150 |
| GRPO run 1-5 (math) | ~$30 |
| pretrain_extend, sft_refresh, sft_math | ~$100 |
| pretrain_extend2 + extensions | ~$60 |
| loop_fix v1 + v2 | ~$25 |
| pretrain_extend3 | ~$60 |
| MOPD (with Gemini API costs) | ~$30 (+ $60 API) |
| grpo_multi v1 + v2 | ~$50 |
| System_sft (in progress) | ~$10 |
| **Cross-workspace total** | **~$3,000 Modal + ~$110 API** |

### 7.2 Pretraining is the dominant cost

The single-biggest line item is pretraining. Everything else combined
(SFT + GRPO + MOPD + RL) was less than pretraining alone. For v6, the
budget question is fundamentally: "how much pretraining can we
afford?"

### 7.3 Sanity tests pay for themselves 5-10×

A failed full run is $20-100 wasted. A sanity test is $1-3. Always
run sanity. We learned this lesson 4-5 times.

### 7.4 GRPO inference is cheaper than expected

H100 compiled GRPO: ~50-100 sec/step at our config. 200 steps =
~3-6 H100-hours = $12-24. Tiny vs pretraining. Don't over-budget
here.

---

## 8. The 363M-specific findings (for OSRT-600M)

These specifically informed v6 architecture decisions:

### 8.1 Multi-digit arithmetic was a hard ceiling

17 × 23 = 391 was the canonical failing prompt. Over multiple ckpts:
- Step 10: "17 * 23 = 39"
- Step 20: "17 * 23 = **1310**"
- Step 25 OOD probe: "17 * 23 = 459"
- Step 40: "17 * 23 = 4374"
- Step 50: "17 * 23 = **459**"
- Step 60: "17 × 23 = **404**"

Format perfect every time, math always wrong. Model genuinely doesn't
know multiplication tables. **No amount of training fixes this at
363M scale.** Either need bigger model OR tool use.

→ v6 600M will be slightly better but still bounded; tool-use
remains essential.

### 8.2 Order-of-operations works; raw arithmetic doesn't

The model could do `12 + 8 × 3 = 36` consistently but failed
`23 + 14 = 37`. Hypothesis: order-of-operations problems have
structural hints (operator precedence) the model can pattern-match
on, while raw arithmetic requires actually computing.

### 8.3 Loop count knob = inference compute knob

Discovery via Test 3 probe: model trained at 6 loops works
reasonably at 3-5 loops too (with quality degradation). Could expose
as an inference-time API knob: `generate(loops=K)` for fast vs slow
modes.

→ v6 productizes this as "controllable inference compute" (Qwen3 /
MiniCPM5-1B style think/no-think).

### 8.4 Per-loop expert routing varied across loops

Probe of MoE routing showed routing distribution AT LOOP 1 differed
from AT LOOP 6 even for the same input — confirming the loop
embeddings were doing their job. The routing per loop was sensible
(different expert subsets activated at different depths).

→ v6 should monitor per-loop expert load to catch any cross-loop
divergence early.

---

## 9. What ALMOST worked but didn't ship

### 9.1 Speculative decoding via aux-loop heads

The `aux_loop_loss_weight` training makes intermediate-loop LM heads
predictive. At inference, loop-3 output could be used as a draft
prediction, verified by loop-6. Expected 2-3× decode speedup with
60-75% accept rate.

We didn't implement it — engineering cost was real and we focused on
training quality. v6 should ship this as a deliverable.

### 9.2 Per-loop adapters / per-loop weights

`per_loop_aux_weights` was wired in v5 (config field) but never
substantively used. Could enable distinct per-loop transformations
beyond just the loop bias.

### 9.3 Vision retrofit (MULTIMODAL.md)

Designed in `MULTIMODAL.md` but never implemented. v6 plans to do it
properly via LLaVA-style projector.

### 9.4 Tool use

Planned as "Stage 6 (optional)" but never built. v6 makes it first-
class (Stage 1 of the post-training pipeline).

---

## 10. Mistakes worth not repeating

| mistake | what should have happened |
|---|---|
| Loop collapse discovered at month 4 | Per-loop CE logged from day 1 |
| Reward hacking discovered after $50 of GRPO | OOD probe + hit-rate logging from day 1 |
| System prompts never trained | Pretrain data includes `<\|system\|>` from day 1 |
| Chat tokens (`<\|think\|>`) trained only at SFT | Pretrain data includes these tokens too |
| Calibration-based eval (6-prompt inference test) | gsm8k full + IFEval + MMLU every checkpoint |
| Cross-workspace migration mid-run | Budget per workspace; complete each run on one |
| GRPO trained base weights | HRA-only from day 1 |
| Last-number-wins reward exploit | Strict extraction from day 1 |
| Multi-env GRPO without per-env hit-rate | Hit-rate logging from day 1 |
| mbpp env at 0% pass rate consuming budget | Drop envs that can't produce real capability |
| Test inference at T=0.7 (noisy) | All benchmarks at T=0.3 (deterministic) |
| Sanity test "looks fine" but didn't include OOD | Sanity tests include critical diagnostics |

---

## 11. What v5 did right (don't lose these in v6)

For balance — the lessons aren't all "what we did wrong":

- **Recursive MoE architecture** — genuinely novel + works
- **HRA injection** — clean adapter surface, plays well with everything
- **Modal-based infrastructure** — volumes, spawn, rescue ckpts all
  paid off
- **Multi-stage training pipeline** — pretrain → SFT → GRPO → distill
  → SFT was the right shape
- **Architecture-fix knobs** — aux_loop_loss_weight + loop_dropout
  preserved capability after discovery
- **Sandbox hardening of code exec** — security review found real
  issues; the hardened version is good
- **Anti-hacking reward stack** — strict_extraction + regurg_penalty
  + OOD probe + hit-rate logging is now built and proven
- **Multi-teacher rollout collection** — DeepSeek + Gemini mix worked
- **Distillation-then-RL** sequencing — MOPD then GRPO was the right
  order

---

## 11a. v6 implementation lessons (CPU pre-flight → GPU bring-up, 2026-06-08/09)

These are NOT v5 training lessons — they're things the v6 (OSRT-600M)
implementation taught us as the `src/osrt/` stack moved from CPU
pre-flight toward the first GPU pretraining run. Recorded here so we
don't relitigate them.

### 11a.1 The attention sink doesn't scale to long context — OOMs at seq 8192

We kept v5's learnable attention sink (per-head sink logit added to the
softmax denominator) into the v6 build. It works only because it
HAPPENED to fit at seq 2048. The sink term cannot be expressed by
`F.scaled_dot_product_attention` (SDPA only normalises over the real
keys), so it forces a **manual attention path that materialises a full
(B, H, S, S) score matrix** (`_attention_with_sink` in model.py). At
the seq-8192 instruction phase that score matrix is ~12 GB and gets
recomputed inside the gradient-checkpointed backward → measured OOM
(>85 GB) at batch 2.

**The fix:** `attention_sink=False` (presets.py). With the sink off,
attention routes through flash SDPA (`F.scaled_dot_product_attention`,
standard GQA, `enable_gqa=True`), which never builds the score matrix —
the SAME workload fits at **35.9 GB**. The sink had no demonstrated
quality benefit; it was kept only because it fit at short context.
The V-from-K latent KV cache is unchanged. Dropping the sink param
moved the model from the old count to **601,444,393 params**.

**Lesson:** don't carry a feature into a longer-context phase just
because it fit at short context. Anything that defeats flash attention
(materialises the score matrix) is a long-context OOM waiting to
happen. Default to SDPA; only diverge from it for a measured win.

### 11a.2 Flash attention alone does NOT let you drop gradient checkpointing

Switching sink→flash freed a lot of activation memory, but the 80 GB
H100 budget still REQUIRES gradient checkpointing (and fused
linear-CE) to fit the recursive 18-effective-layer forward at the
target batch/seq. Flash buys headroom; it does not buy you out of
checkpointing. Both are mandatory together for v6 on a single H100.

**Lesson:** these are independent levers. "We added flash" is not a
licence to turn off checkpointing — re-measure before assuming the
saving compounds.

### 11a.3 B4 grouped-GEMM removes the only torch.compile graph break

v6's MoE dispatch had a per-expert `.nonzero()` gather (the loop path,
`_dispatch_loop`). That data-dependent op is **the only torch.compile
graph break** in the model — it was responsible for all 12 graph
breaks and blocked fullgraph compilation. Replacing it with a
sort-by-expert +
`torch._grouped_mm` grouped-GEMM dispatch (`moe_grouped_gemm=True`,
`_dispatch_grouped`) is dropless and compiles **fullgraph (0 graph
breaks)**, ~9-12% faster steady-state. Weights are identical to the
loop path, so checkpoints load under either dispatch; the loop path is
retained as a CPU-safe fallback.

**Lesson:** one data-dependent op (`.nonzero()`, `.item()`, dynamic
shapes) can cost you fullgraph compile across the WHOLE model. Hunt
the graph breaks; a single fix can be worth all the others combined.

### 11a.4 Non-persistent rope buffers get materialised as garbage by HF from_pretrained

Making the v6 model HF-compliant (`save_pretrained`/`from_pretrained`,
AutoConfig/AutoModelForCausalLM registration) surfaced a subtle data
corruption: HF's `from_pretrained` builds the model skeleton on the
**meta device**, then loads only the saved tensors. RoPE cos/sin
tables are derived from config, so the instinct is to register them
`persistent=False` (don't bloat the checkpoint). But a non-persistent
buffer is NOT in the checkpoint → `from_pretrained` leaves it as the
**uninitialised meta-device garbage** the skeleton allocated → every
reloaded model had corrupted RoPE.

**The fix:** register `rope_cos`/`rope_sin` with `persistent=True`
(model.py) so they ride along in the checkpoint and get restored. ~2 MB
at the real config — negligible. The on-the-fly recompute path still
handles seq_len beyond the cached range.

**Lesson:** under meta-device init, "derived from config" is NOT the
same as "safe to omit from the checkpoint". Any buffer you want
correct after `from_pretrained` must be persistent (or explicitly
re-materialised post-load). Also note v6 deliberately sets
`supports_gradient_checkpointing = False` and gates checkpointing
through a PRIVATE `_osrt_grad_ckpt` flag — HF's own gradient-checkpoint
mechanism collided with the recursive-loop forward, so we keep HF out
of it entirely.

### 11a.5 Collapse telemetry is now wired from step 1 (the v5 lesson, applied)

v5's biggest discovery (loop collapse, §1.1) and the per-loop routing
finding (§8.4) are now permanent instrumentation in v6, not
post-hoc probes: per-loop residual-update norm (`loop/update_norm_l*`,
plus mean/min/last) and a dead-expert count (`moe/dead_experts_total`)
are computed every log step and emitted to BOTH W&B and stdout. A loop
whose update-norm → 0 has collapsed to a no-op; a rising dead-expert
count is router collapse. This is the "measure first" principle from
§0 finally built into the loop instead of bolted on after a regression.

---

## 12. The brutal honest summary

v5 was a year of "look how clever this architecture is" followed by
"oh no we never measured if it works" followed by "oh no the format
isn't trained" followed by "oh no GRPO is hacking the rewards" 
followed by "oh wait some of this actually works."

v6 (OSRT-600M) is "measure first, then build". Everything in 
[`README.md`](README.md) is a direct response to a specific v5
failure mode documented above.

The biggest single takeaway: **for a small model, training-pipeline
discipline matters more than architectural novelty.** v5 had the
architecture and got the pipeline wrong; v6 keeps the architecture and
fixes the pipeline.

---

## 13. Open questions remaining from v5

These are genuinely unresolved and should inform v6 ablations:

1. **Per-loop routing accounting** — is one global router per block
   sufficient, or do we need per-loop routing? Novel territory.
2. **Loop count vs physical depth trade** — is 3 blocks × 6 loops
   actually optimal, or would 2 × 8 / 4 × 4 be better at fixed
   compute?
3. **MTP head speculative decoding accept rate** — measurement
   needed; expected 60-75% but never verified
4. **Tool-use ceiling** — at what scale does calculator tool use
   "click"? Couldn't test at 363M without tool training.
5. **Multi-turn / agentic capability** — never tested at 363M; not
   sure what's possible at this scale.

---

## Document changelog

- **2026-06-07** — initial creation, captures v5 lineage through
  `system_sft` in-progress
- **2026-06-09** — added §11a (v6 CPU-pre-flight → GPU bring-up
  lessons): attention sink OOMs at seq 8192 (flash SDPA required),
  flash alone doesn't drop checkpointing, B4 grouped-GEMM removes the
  only compile graph break, non-persistent rope buffers materialise as
  meta-init garbage under HF `from_pretrained`, collapse telemetry now
  wired from step 1
