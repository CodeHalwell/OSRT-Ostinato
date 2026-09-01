# CLAUDE.md

Guidance for AI assistants (Claude Code et al.) working in this repository.

## What this project is

**OSRT** = **Optimized Sparse Recursive Transformer** — a small language model
combining **sparse MoE** + **depth recurrence** + **Muon optimization**.
The first two together are prior art — MoEUT (NeurIPS 2024, roadmap §17.2) is
the closest relative and should be cited as such. The three-way conjunction is
OSRT's, but it is an engineering combination, not an architectural novelty.

This repo is **v7**. Lineage v3–v6 lives in `CodeHalwell/OSRT-605M-A269M`,
which is now an archive; v6's trained checkpoints remain on HF at
`HallD/osrt-v6-ckpt`. v7 is not a re-architecture — it is a re-grain,
re-tokenize, cost-fix and recipe upgrade on the v6 core.

**North star: highest quality at the fastest achievable inference**, targeting
Blackwell — primarily the RTX PRO 6000 (GB202, 96GB GDDR7, single-GPU, no
NVLink), occasionally B200/GB200. **Decode latency is a first-class design
constraint, not a deployment afterthought.**

## The one naming rule

**Never put parameter counts in a name.** Not in the repo name, the package
metadata, a preset, or a directory. The v6 lineage carried four mutually
inconsistent stale counts simultaneously — `OSRT-605M-A269M` (repo),
`nano-osrt-100m` (checkout), `OSRT_605M_A288M` (preset), and "~608M/~279M"
(pyproject description) — against an actual 601M/278M.

`scripts/compute_budget.py` is the **only** trusted source for parameter
counts. It instantiates the real model on a `meta` device. Do not hand-write
param tables in docs; regenerate them.

## v7 committed shape

See `docs/specs/2026-08-11-v7-roadmap.md` §14. Summary:

- 3 physical blocks × 6 loops (18 effective layers), dim 1536
- **28 × h2112 routed experts per block, top-4** (14.3% density)
- 1 × h2816 shared expert per block
- GQA 24q/8kv, KDV compressed-latent cache, QK-norm
- HRA rank 256, native, per effective layer
- Tokenizer: **SmolLM2-based, 49,280 padded / 49,184 real** — G2 resolved (§16)
- **mHC: OFF** — decision stands (§12.3; its 2026-08-18 amendment keeps one
  G3 ladder slot as cheap insurance, so "off" is settled, not unrevisitable)
- Quantile Balancing router bias — **required**, not optional (§14.6)
- SiTU-GLU experts, per-head Muon, seq-balance 1e-4 — set in code per §14.1;
  the **V4 Muon recipe (item 1.3) is the one §14.1 line not yet implemented**

Open: MTP head count (§15 — do **not** slim to 1), loops × blocks (G4).

## Gate board — measure before you build

| gate | question | status |
|---|---|---|
| G8 | drafter accepted length on frozen v6 | **blocked**: `HallD/osrt-v6-ckpt` was absent from an authenticated HF listing on 2026-08-31 — locate the frozen v6 ckpt (Mac?) or G8 is dead |
| G7 | do routed experts get FP8/NVFP4 kernels via grouped-GEMM? | before G3/G4; `probe_gpu.py` ready, needs the GPU box |
| G3a | does the token requirement track active or total params? | blocks the trunk run; no harness yet |
| G2 | tokenizer bake-off | **resolved** — SmolLM2 + 32 OSRT specials (§16), shipped in `tokenizer/` |
| G3/G4 | expert re-grain; loops × blocks | open; no ladder harness yet |

The roadmap's §12 is an independent citation audit of §§4–6 — **read it before
citing any external claim from this repo.** Three material errors were found.

## What is and is not in this repo

**Present:** the architecture (`model.py`, `muon.py`, `hra.py`, `fused_ce.py`,
`quant.py`, `monitoring.py`), the pretraining spine (`train.py`, `data.py`,
`train_config.py`, `train_main.py`), the chat contract (`system_prompts.py`),
the tokenizer, and the tooling under `scripts/`.

**Deliberately absent — v6, archived, not ported:**

| removed | why |
|---|---|
| `mhc.py` and every call site | decided off permanently, roadmap §12.3 |
| `sft_train/sft_eval/sft_data.py` | v6 SFT pipeline, redesigned for v7 |
| `grpo_train.py`, `rewards.py` | v6 RL. The reward is **on record as having made the model measurably worse** (soup − step100 θ = +8.00pp, p=0.002). v7 post-training is redesigned around a verifier that does not exist yet |
| `lm_eval_wrapper.py` | v6 eval lane |
| 31 v6 stage configs | Pretrain-extend ×3, LoopFix ×2, MOPD, SystemSFT, Midtrain ×6, SFT v1–v4/Long/Refresh/Math, GRPO ×4 |
| `run_pretrain_extend`, `run_rollout_eval` | dead once their configs went |

All of it remains in this repo's git history and in
`CodeHalwell/OSRT-605M-A269M`. **Do not resurrect any of it without reading
the roadmap first** — most was removed because it was superseded, and the
GRPO reward because it was measurably harmful.

`docs/v6/` holds the v6 handoff and Colab recipe: kept only because the
roadmap cites them, labelled so nobody mistakes them for current.

## To run: read `RUNBOOK.md`

The design is committed and the trunk is one command — `modal run --detach app.py
--trunk-run`, or the Colab notebook. Roadmap **§19** records every design bet
and its falsifier; results are read against that. The collapse detectors run
during the run. The ladder (`scripts/launch_ladder.sh`) is for explaining
results afterwards, not gating them.

## Environment & commands

Python 3.11, managed with `uv` (`uv.lock` is the source of truth; CI runs
`uv sync --frozen`). CI = ruff + pytest + `compute_budget.py`; keep all three
green locally before pushing:

```bash
uv sync                                                   # install (frozen in CI)
uv run ruff check .                                       # lint (format not enforced yet — see ci.yml)
uv run pytest -q                                          # CPU suite, ~3 min
PYTHONPATH=src uv run python scripts/compute_budget.py    # canonical param counts
PYTHONPATH=src uv run python scripts/sanity_overfit.py    # overfit-one-batch sanity
python scripts/build_tokenizer_v7.py --out tokenizer      # rebuild the v7 tokenizer
PYTHONPATH=src uv run python -m osrt.train_main --help    # pretrain entry point (GPU)
```

GPU work runs on Colab (RTX PRO 6000 Blackwell) via
`notebooks/v7_pretrain_colab.ipynb` — probe first (`scripts/probe_gpu.py`),
cross-session checkpoints via `--hf-repo`. Needs `HF_TOKEN` and
`WANDB_API_KEY` in Colab Secrets.

## Conventions & gotchas

- **`src/osrt/` is ground truth.** When docs and code disagree, the code wins;
  update the doc.
- **`model.py` rounds `expert_hidden` up to a multiple of 64** for tensor-core
  alignment. A non-aligned value is silently rewritten — check the `cfg:` echo
  line from `compute_budget.py`, not the value you passed.
- **Never commit weights, data, or tokens.** This repo is public. Training data
  lives on HF. A 103MB jsonl is over GitHub's 100MB hard limit and the push is
  rejected outright.
- **Third-party paper PDFs are gitignored** — arXiv's default licence grants no
  redistribution right. Cite the arXiv ID.
- **Stability features are load-bearing:** QK-norm, sandwich RMSNorm, per-loop
  routing accounting, aux-loss-free balancing, SwiGLU clamp. Don't remove them
  to "simplify" without understanding the failure they prevent.
- **Run a smoke/sanity variant before any real GPU spend.**

## Where to read more

- **`docs/specs/2026-08-11-v7-roadmap.md`** — the plan, its audit (§12), the
  north star (§13), the committed shape (§14), and DSpark (§15). Start here.
- `docs/00-overview.md` → chapters 01–09 — architecture deep-dive.
- `docs/LEARNINGS.md` — what went wrong in v5, and why the design is shaped this way.
- `docs/RESEARCH.md` — the papers behind each technique.
- `docs/ARCHITECTURE.md` — config-value spec.
