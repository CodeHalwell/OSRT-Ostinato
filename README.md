# OSRT

**Optimized Sparse Recursive Transformer** — a small language model that
combines **sparse MoE**, **depth recurrence**, and **Muon optimization**.
Three physical decoder blocks are applied six times through a loop, giving 18
effective layers from one third of the parameters.

> **v7, in development.** The v3–v6 lineage is archived at
> [`CodeHalwell/OSRT-605M-A269M`](https://github.com/CodeHalwell/OSRT-605M-A269M);
> v6's trained checkpoints remain on Hugging Face. This repo starts from the
> v6 core and applies the v7 plan — see
> [`docs/specs/2026-08-11-v7-roadmap.md`](docs/specs/2026-08-11-v7-roadmap.md).

## North star

Highest quality at the fastest achievable inference, on Blackwell — primarily
the RTX PRO 6000 (96GB, single-GPU), occasionally B200/GB200. Decode latency
is a design constraint, not a deployment afterthought.

## Shape

| | |
|---|---|
| blocks × loops | 3 × 6 (18 effective layers), dim 1536 |
| routed experts | 28 × h2112 per block, top-4 (14.3% density) |
| shared expert | 1 × h2816 per block |
| attention | GQA 24q/8kv, KDV compressed-latent cache, QK-norm |
| adapters | HRA rank 256, native, per effective layer |
| optimizer | Muon (2D hidden matrices) + AdamW (embeddings, norms, biases) |

**Parameter counts are not stated in this README on purpose.** Generate them:

```bash
PYTHONPATH=src uv run python scripts/compute_budget.py
```

That script instantiates the real model on a `meta` device and is the only
trusted source. The previous repo carried four mutually inconsistent stale
counts at once; this one states none.

## Quick start

To actually launch runs, follow [`RUNBOOK.md`](RUNBOOK.md) — it sequences the
probe, the launch gate and the ladder, and says what each result means.

### Local checks

```bash
uv sync
uv run pytest                                            # CPU suite
PYTHONPATH=src uv run python scripts/compute_budget.py   # param budget
PYTHONPATH=src uv run python scripts/sanity_overfit.py   # overfit one batch
```

## Status

**Pre-training-run.** The architecture, tokenizer and pretraining spine are in
place; nothing has been trained under v7. The v6 post-training stack (SFT, GRPO,
reward functions, eval lanes) was **not** ported — v7 redesigns it around a
verifier that does not exist yet, and the v6 GRPO reward is on record as having
made the model measurably worse. History lives in
[`CodeHalwell/OSRT-605M-A269M`](https://github.com/CodeHalwell/OSRT-605M-A269M). The open decision gates —
and the evidence behind each — are tracked in the roadmap; the measure-first
rule from `docs/LEARNINGS.md` still governs.

## Reading order

1. [`docs/specs/2026-08-11-v7-roadmap.md`](docs/specs/2026-08-11-v7-roadmap.md) — the plan, its independent citation audit (§12), the north star (§13), the committed shape (§14)
2. [`docs/00-overview.md`](docs/00-overview.md) — architecture, chapters 01–09
3. [`docs/LEARNINGS.md`](docs/LEARNINGS.md) — v5's failure modes, and why the design looks like this
4. [`docs/RESEARCH.md`](docs/RESEARCH.md) — the papers behind each technique
