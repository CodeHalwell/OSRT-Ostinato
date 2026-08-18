# CLAUDE.md

Guidance for AI assistants (Claude Code et al.) working in this repository.

## What this project is

**OSRT** = **Optimized Sparse Recursive Transformer** — a small language model
combining three ideas no released frontier model puts together at once:
**sparse MoE** + **depth recurrence** + **Muon optimization**.

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
- **mHC: OFF** — decided permanently (roadmap §12.3)
- Quantile Balancing router bias — **required**, not optional (§14.6)

Open: tokenizer (G2), MTP head count (§15 — do **not** slim to 1),
loops × blocks (G4).

## Gate board — measure before you build

| gate | question | status |
|---|---|---|
| G8 | drafter accepted length on frozen v6 | **runs now, blocks nothing** |
| G7 | do routed experts get FP8/NVFP4 kernels via grouped-GEMM? | before G3/G4 |
| G3a | does the token requirement track active or total params? | blocks the trunk run |
| G2 | tokenizer bake-off | open |
| G3/G4 | expert re-grain; loops × blocks | open |

The roadmap's §12 is an independent citation audit of §§4–6 — **read it before
citing any external claim from this repo.** Three material errors were found.

## Environment & commands

Python **3.11**, dependencies via **uv** (`uv.lock` is committed).

```bash
uv sync                       # install deps (incl. dev: pytest, ruff)
uv run pytest                 # full CPU suite
uv run ruff check .           # lint — this repo starts at zero errors, keep it there
uv run ruff format .

PYTHONPATH=src uv run python scripts/compute_budget.py   # canonical param counts
PYTHONPATH=src uv run python scripts/sanity_overfit.py   # overfit one batch → loss ~0
PYTHONPATH=src uv run python scripts/dummy_train.py      # synthetic copy task
```

CI runs lint, format check, tests and the budget script on every push and PR.

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
- `LEARNINGS.md` — what went wrong in v5, and why the design is shaped this way.
- `RESEARCH.md` — the papers behind each technique.
- `ARCHITECTURE.md` — config-value spec.
