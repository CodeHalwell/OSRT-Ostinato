# OSRT v7 (OSRT-Ostinato) — progress record

Everything done on v7 so far, in order, with the measured results. One page;
the detail lives in `docs/specs/2026-08-11-v7-roadmap.md` (section numbers
below), `docs/specs/2026-09-02-data-plan.md` and the git history.
Last updated 2026-10-03.

## 1. Where v7 came from (August 2026)

- v3–v6 (`nano-osrt-100m`, now an archive) established the recursive sparse-MoE
  idea and its failure modes: loop collapse, router collapse, reward hacking,
  and an under-trained base that learns SFT format but not substance
  (`docs/LEARNINGS.md`, roadmap §2).
- The v7 roadmap (§3–§11, 2026-08-11) set the strategic question — highest
  quality at the fastest inference on Blackwell — scanned the DeepSeek V4,
  GLM 5, Kimi K3 and Nemotron 3 lineages, and ranked the candidate techniques.
- **2026-08-18**: the repo was split out as OSRT-Ostinato and made pure v7:
  mHC removed, the v6 post-training stack dropped, `OSRT_V7` preset landed.
  CPU-side prerequisites implemented: Quantile Balancing, WSD schedule, vocab
  padding fix (§12–§14).
- **Tokenizer (G2, §16)**: SmolLM2's 49,152 vocab extended with the 32 OSRT
  special tokens → 49,184 real, padded to 49,280. Chosen on measured fertility
  over the real data mix; v6's 65K BPE was the wrong tool for its arithmetic.
- **Speculative decoding evidence (§15)**: literature review (DSpark) plus a
  plan to use the MTP heads as the drafter.

## 2. The committed architecture (§14, `src/osrt/presets.py`)

| | |
|---|---|
| Physical / active parameters | 968,468,355 / 263,035,779 (`scripts/compute_budget.py`) |
| Depth | 3 physical blocks × 6 loops = 18 effective layers, dim 1536 |
| MoE | 28 routed experts × h2112, top-4, plus 1 shared expert (h3840) |
| Attention | GQA 24q/8kv, head_dim 64, QK-norm; KDV latent KV cache (512-wide) |
| Heads | 2 MTP heads (also the speculative drafter) |
| Routing | Quantile Balancing, aux-loss-free bias, per-loop accounting |
| Activation | SiTU-GLU; sandwich RMSNorm; SwiGLU clamp |
| Optimiser | Muon (hidden 2-D) + AdamW (embeddings/norms/biases) |
| HRA adapters | **off for pretraining** (see §4 below); reserved for SFT/GRPO |

## 3. Making it runnable (2026-09-01)

- Launch-safe training recipe with a fail-closed contract and readiness tests
  (`tests/test_v7_readiness.py`); multi-workspace Modal launcher; `RUNBOOK`.
- Modal lessons that cost runs before they were fixed: `modal run --detach` is
  required for spawned functions; the image had to be torch 2.10.0+cu128
  (2.8.0 could not compile the unbacked symbols); `dataloader_num_workers=0`
  (workers abort at the phase switch); 8 h arm timeout.
- The loop-health early stops existed but were never compared — wired.

## 4. The ladder — E1 and the HRA decision (2026-09-01 → 09-02, §18)

Six arms at the trunk shape, 600 steps each (~80M tokens/arm, ~$30 total),
on H100: `a` (HRA on), `nohra`, `b`, `c`, `dense`, `g4`.

| step | a (HRA) | **nohra** | b | c | dense | g4 |
|---|---|---|---|---|---|---|
| 400 | 6.29 | **4.91** | 6.14 | 6.15 | 7.11 | 6.34 |
| 600 | 5.33 | **3.81** | 5.40 | 5.36 | 5.39 | 5.66 |

**Result**: removing HRA won by 1.5 nats at step 600 and the gap was widening.
Cause: the adapter acted on the raw residual, an un-normalised multiplicative
feedback inside the recursion; block outputs reached 50–300× their inputs.
Fix: the adapter now acts on the normalised input, and **the trunk trains
without HRA**, reinvesting its budget into the shared expert (2816 → 3840,
iso-parameter). HRA is kept as a post-training adapter on the frozen base.
The hidden-norm telemetry was redefined to a within-recursion ratio and wired
to an early stop.

## 5. Data plan (2026-09-02, `docs/specs/2026-09-02-data-plan.md`)

Researched and applied a three-phase pretraining mix plus SFT/RL pools for an
all-round reasoning model with a code lean. Pretraining phases: **foundation**
(5%, seq 2048, 7 sources), **knowledge** (80%, seq 4096, 15 sources incl.
Stack v3 multi-language code, Nemotron code/math, FineMath, arXiv),
**anneal** (15%, seq 8192, 21 sources: instruction 35 / code 26 / math 23 /
STEM 11 / long docs 5). Per-dataset filters and subsampling with a realised-mix
log; a repo-level Stack v3 formatter; 43/43 sources pre-flighted.

## 6. MTP-head speculative decoding (2026-09-02, §15.7)

Built `generate(..., speculative=True, spec_drafter="mtp")`: one forward per
round over the pending token plus the heads' drafts. Measured on a 500-step
ladder checkpoint (a weak drafter):

| precision | greedy tok/s | spec tok/s | speedup | acceptance | identical to greedy |
|---|---|---|---|---|---|
| bf16 | 22.8 | 33.9 | 1.49× | 0.375 | 1/4 |
| fp32 | 18.2 | 27.9 | 1.54× | 0.351 | 4/4 |

## 7. B200 sizing (2026-09-02, §13b)

32K tokens per micro-batch peaks at ~145 GB of 192 at every sequence length;
65K OOMs; gradient checkpointing did not rescue it (~4 MB/token lives outside
the checkpointed blocks — still open). The trunk ran 16×2048×4 / 6×4096×11 /
2×8192×32 after launch 1 OOMed at the phase switch; measured 2.7 / 5.4 / ~13
s per step (≈48K / 47K / 40K tok/s). Steady state used ~60% of the card.

## 8. The trunk pretrain (2026-09-02 → 2026-10-03, §20)

18,000 steps, 5.43B tokens, eleven launches on five Modal workspaces, relaying
the checkpoint whenever a workspace's credit ran out. Incidents, each fixed
in code: OOM at the knowledge switch (cache release + smaller micro-batches);
a held-out eval that iterated 100M streamed rows (eval moved to the newest
dump, after the save); a fraction-of-terms logging glitch; the Modal client
silently zero-filling chunks of an 8 GB download (verify with `testzip`;
move weights Modal → HF from a container instead).

Held-out loss (fineweb-edu CC-MAIN-2025-26):

| step | 1k | 2k | 4k | 6k | 8k | 10k | 12k | 13k | 14k | 15k | 16k | 17k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loss | 5.39 | 5.26 | 4.83 | 4.40 | 4.39 | 4.04 | 3.87 | 3.66 | 3.55 | 3.46 | 3.38 | **3.34** |

Nine consecutive improvements from step 9,000; balance loss ~1.02–1.1
throughout; no early stops, no dead experts beyond last-micro-batch sampling
noise. Two corrections to the record: the LR schedule **was cosine, not WSD**
(the trunk code ignored `lr_schedule`; 6e-4 → 6e-5 over steps 400 → 18,000),
and packing let a single long low-entropy document fill a whole step two or
three times per thousand (add a per-row token cap before any further
pretraining).

**Branch note.** PR #2 (merged 2026-10-01) changed the router gate default,
the Muon LR, moved the router to fp32, rewrote chat formatting and added
drift checks that reject the trunk's checkpoints. The trunk's exact code is
branch **`trunk-pinned`** (= `1db3d89`); everything that resumes, scores or
exports trunk checkpoints runs from there until `main` gains a loader that
reproduces the pinned held-out loss.

## 9. Base evaluation (2026-10-03, §20.5)

**Soup.** Averaging steps 17,000 + 17,500 + 18,000 gives held-out **3.18**
(ppl 24) against 3.30 for the final checkpoint; a fourth member adds nothing.
**The base model is `osrt_soup_17000_17500_18000`**, on the private HF repo
`HallD/OSRT-Ostinato-trunk` (full checkpoint and bf16 safetensors), alongside
steps 16,500–18,000.

**lm-eval (base-model mode, full sets unless noted; GPT-2 small as scale):**

| task | OSRT soup | chance | GPT-2 small |
|---|---|---|---|
| HellaSwag | 27.8 | 25 | ~29 |
| ARC-Easy | 45.1 | 25 | ~44 |
| ARC-Challenge | 20.8 | 25 | ~23 |
| PIQA | 58.7 | 50 | ~63 |
| Winogrande | 51.8 | 50 | ~52 |
| MMLU | 26.7 | 25 | ~26 |
| LAMBADA acc / ppl | 19.7 / 262 | — | ~46 / 35 |
| GSM8K 5-shot (200), strict / flexible | **6.0 / 8.0** | 0 | ~1 |
| HumanEval pass@1 | 0 / 164 | 0 | 0 |

Reading: GPT-2-small class on knowledge and commonsense (what 5.4B tokens
buys), with the maths mix already visible at GSM8K. LAMBADA is a domain gap
(no fiction in the mix). HumanEval is zero partly by format — **the base is
chat-annealed in plain text** (`assistant:` lines, fenced code) and mixes tab
and space indentation — so coding is to be measured after SFT.

## 10. Tooling that exists now

- `scripts/eval_trunk.py` — held-out scoring of checkpoints and soups on
  identical cached batches (branch `trunk-pinned`).
- `src/osrt/lm_eval_wrapper.py` + `scripts/lm_eval_trunk.py` — lm-eval
  harness over OSRT, base-model and chat modes, pulls checkpoints from HF.
- `scripts/measure_spec_decode.py`, `scripts/probe_b200_batch.py`,
  `scripts/preflight_data.py`, `scripts/compute_budget.py`.
- Checkpoint export to HF (full + bf16 safetensors) from a Modal container.

## 11. Spend

Ladder ~$30; trunk ~$280 across five workspaces over September and the first
days of October; evaluation ~$12.

## 12. Next

1. `main` loader gate: load the soup with the trunk's recipe
   (`router_bias_in_gates=True`, Muon 0.02) and reproduce held-out 3.18.
2. SFT on the frozen soup with HRA adapters, per the data plan: short
   chain-of-thought, length-matched, indentation normalised, the chat schema
   the base already half-knows.
3. GRPO with strict verifiable rewards (no partial credit), rollout
   temperature 0.4, dead-prompt filtering, paired-bootstrap checkpoint
   selection, ship a soup.
4. Open engineering: the ~4 MB/token memory overhead; batched generation in
   the eval wrapper; a per-row token cap in the data pipeline; DSpark drafter.
