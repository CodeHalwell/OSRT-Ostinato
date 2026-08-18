# OSRT Ostinato — Architecture Overview

*The entry point to the `docs/` architecture series. Read this first, then
follow the numbered chapters. Each chapter is grounded in `src/osrt/` and
cites `file:line`; where the code and the older spec (`ARCHITECTURE.md`)
disagree, the chapters trust the code and flag the drift.*

---

## What OSRT is

**OSRT** = **Optimized Sparse Recursive Transformer**.

- **O**ptimized — Muon optimizer + a quantized deployment stack
- **S**parse — mixture-of-experts (top-4 of 28 routed + 1 shared per block)
- **R**ecursive — 3 physical blocks applied 6 times via depth recurrence
- **T**ransformer — a standard pre-norm decoder backbone

It is a **recursive Mixtral-style sparse MoE decoder**: only 3 physical
decoder blocks exist, but they are run **6 times in a loop** (weights
reused each pass), giving **18 effective layers** of depth from one-third
the parameters. The design folds in a stack of 2025-2026 small-LM
techniques (MLA-style KV compression, Muon, aux-loss-free MoE balancing)
on top of the lessons from the v5 lineage.

The single defining bet: **depth recurrence is parameter-efficient
depth** — the model does the forward compute of a much deeper dense model
because the same blocks are traversed six times.

## The numbers (generated, not hand-derived)

All counts come from `PYTHONPATH=src python scripts/compute_budget.py`,
which instantiates the canonical `OSRT_V7` preset (`src/osrt/presets.py`)
and counts real parameters. Re-run it after any config change — this file
deliberately quotes no number that the script does not generate.

| quantity | value |
|---|---|
| **Physical parameters** | **968,468,355** |
| **Active per token (inference)** | **263,035,779 (27.2%)** |
| Hidden dim `d_model` | 1,536 |
| Vocab | 49,280 padded / 49,184 real (SmolLM2 base + 32 OSRT specials) |
| Physical blocks × loops | 3 × 6 = **18 effective layers** |
| Attention | GQA 24 query / 8 KV heads, head_dim 64 |
| MoE | 1 shared (h=2,816) + 28 routed (h=2,112), top-4 |
| HRA adapters | rank 256, 18 (one per effective layer) |
| MTP heads | 2 (training-only, dropped at deploy) |

> **Naming rule.** No parameter count appears in any name — repo, preset,
> package or directory. The v6 lineage carried four mutually inconsistent
> stale counts at once. `scripts/compute_budget.py` is the only source.

## Status

The **architecture is implemented** in `src/osrt/`. 144 unit tests pass,
and the `dummy_train` / `sanity_overfit` CPU smoke tests confirm the full
stack trains and learns without MoE/loop collapse. GPU bring-up has begun:
the attention sink was **dropped** in favour of flash
`F.scaled_dot_product_attention` (the manual sink path OOMed at seq 8192;
flash fits — `presets.py:47-54`), and **grouped-GEMM MoE** is on
(`moe_grouped_gemm=True`, validated on H100 to track the loop path,
dropless, ~9-12% faster — `presets.py:55-60`). Fused cross-entropy and
gradient checkpointing are wired and mandatory for the fit.

## How a token flows through the model

```
input_ids
   │
   ▼
[ Embedding ]  (tied with the LM head)                      → ch.01
   │
   ├─────────────── for loop r in 0..5 (6 loops) ───────────┐  → ch.05
   │                                                          │
   │   for block b in 0..2 (3 physical blocks):              │
   │     ┌──────────────────────────────────────────────┐   │
   │     │ Attention sub-block                            │   │
   │     │   GQA + KDV (Key-Derived Value) latent cache   │   │ → ch.02
   │     │   QK-Norm, RoPE, flash SDPA (no sink)          │   │
   │     │   + HRA rank-256 adapter (this effective layer)│   │ → ch.04
   │     ├──────────────────────────────────────────────┤   │
   │     │ MoE sub-block                                  │   │
   │     │   loop_emb biases the router                   │   │ → ch.05
   │     │   sqrt(softplus) router, top-4 of 28           │   │ → ch.03
   │     │   1 shared expert (always on) + 4 routed       │   │
   │     │   aux-loss-free balance bias                   │   │
   │     │   grouped-GEMM dispatch (dropless, fullgraph)  │   │
   │     └──────────────────────────────────────────────┘   │
   │                                                          │
   │   (end of loop r → per-loop aux loss + collapse         │ → ch.05/06
   │      telemetry: loop/update_norm_l* residual norm)      │
   └──────────────────────────────────────────────────────────┘
   │
   ▼
[ Tied LM head ] → logits (B, S, 49280)                      → ch.05
   │
   ├─ training: + per-loop aux LM losses + MTP losses
   │            + router balance / z losses                   → ch.07
   └─ inference: sample → KV-cached decode → (speculative)    → ch.09
```

Optimizer (**Muon + AdamW**, ch.08) trains it; quantization (ch.10)
shrinks it for deployment.

## The chapters

| # | chapter | covers |
|---|---|---|
| 01 | [Tokenizer & Embedding](01-tokenizer-embedding.md) | byte-level BPE, the 21-token contract (14 built / 7 missing), tied embedding ↔ LM head |
| 02 | [Attention](02-attention.md) | GQA via flash `F.scaled_dot_product_attention`, the KDV (Key-Derived Value) latent KV cache, RoPE, QK-Norm (attention sink dropped) |
| 03 | [MoE & Routing](03-moe-and-routing.md) | shared + 28 routed experts, sqrt(softplus) router, aux-loss-free balancing, hash routing, SwiGLU clamp |
| 04 | [HRA Adapters](04-hra-adapters.md) | the 18 rank-256 attention-path adapters (+ the separate injected retrofit path) |
| 05 | [Recursion](05-recursion.md) | depth recurrence, loop embeddings, loop dropout, per-loop aux heads |
| 06 | [Heads & Losses](06-heads-and-losses.md) | tied LM head, per-loop aux losses, MTP heads, router losses, the total loss |
| 07 | [Optimizer (Muon)](07-optimizer.md) | Muon + Newton-Schulz, the Muon/AdamW split, decoupled weight decay |
| 08 | [Inference & KV Cache](08-inference-kv-cache.md) | prefill/decode, the latent-only cache, preallocated decode, speculative decoding |
| 09 | [Quantization & Deployment](09-quantization-deployment.md) | int4 KV (implemented), AlphaQ FP4 experts (planned), the memory budget |

## How these docs relate to the other files

- **`ARCHITECTURE.md`** (repo root) — the terse spec / single source of
  truth for config values. These `docs/` chapters are the *explanatory*
  companion: they teach the WHY and walk the code. Where the two
  disagree, the chapters cite the code and flag it (the spec predates
  some implementation choices).
- **`README.md`** — design philosophy and the integrated training plan.
- **`LEARNINGS.md`** — what the v5 (363M) runs taught us (loop collapse,
  router collapse, reward hacking) — the failure modes this architecture
  is built to avoid.
- **`RESEARCH.md`** — the external papers behind each technique.
- **`src/osrt/`** — the implementation. The ground truth for exact
  behaviour. When in doubt, read the code; these docs point you to the
  right `file:line`.

## A note on accuracy

Every chapter was written by reading the actual `src/osrt/` source, not
by paraphrasing the spec. As a result they surface several places where
`ARCHITECTURE.md` had drifted from the code (e.g. where loop embeddings
are applied, which params go to which optimizer, the exact balance-bias
update). Those flags are features, not bugs — they're the difference
between documentation you can trust and documentation you can't. If a
chapter and the spec disagree, **the chapter (and the code) wins.**
