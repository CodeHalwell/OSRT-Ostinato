# Cross-Loop KV Reuse — Design Note (investigation)

**Date:** 2026-06-16
**Status:** Investigation / hypothesis — NOT approved, NOT implemented.
**Owner axis:** decode memory bandwidth (the project's stated attention thesis).
**Companion:** `ARCHITECTURE.md` §6.2–6.3 (KDV), `docs/02-attention.md`.

---

## 1. Why this note exists

OSRT's attention design is justified on **memory bandwidth**, not expressivity
(`ARCHITECTURE.md` §6.2): autoregressive decode is HBM-bandwidth-bound, so
decode throughput scales as `1 / (cache bytes per token per layer)`. KDV banks a
**constant 2×** on that axis (cache 512 scalars/token/layer instead of 1024).

This note catalogues the *larger* bytes-moved levers that KDV does **not**
touch, and works through the one that is unique to OSRT's recursive stack:
**reusing the KV latent across recursive loops.** It is the highest-leverage
bandwidth idea available to this architecture (up to ~6×), and it is also the
one most in tension with the model's central reasoning hypothesis — so it must
be *measured*, specifically against the reasoning-on>off north-star metric, not
just perplexity.

---

## 2. The current state — 18 independent caches

The forward loop builds one cache entry per **effective** layer
(`src/osrt/model.py:1658-1711`):

```python
for loop in range(n_loops_to_run):              # 6 loops
    for block_idx, block in enumerate(self.blocks):  # 3 physical blocks
        idx = loop * self.config.num_blocks + block_idx   # 0..17
        layer_past = past_key_values[idx] if past_key_values is not None else None
        x, present_kv = block(..., loop_idx=loop, past_key_value=layer_past, ...)
        presents.append(present_kv)
```

So the cache holds **18 = 6 × 3** distinct latents per token. The *weights* of
the 3 physical blocks are shared across loops, but each `(loop, block)` pair
caches its **own** latent because the residual stream `x` has been updated by
the previous loop. Per-token cache footprint:

```
512 scalars × 18 layers × 2 bytes (bf16) = 18,432 B/token ≈ 18 KB/token
```

At the 8000-token reasoning target that is **~144 MB** of KV cache, and the
bytes *moved* over a full decode is the quadratic sum `≈ Σ_t (cache up to t)`,
so the per-token-per-layer figure multiplies the entire decode bandwidth bill.

## 3. The lever ranking (why this one)

For the 8000-token decode, rank levers by the factor they remove from
bytes-moved:

| lever | factor | axis | status |
|---|---|---|---|
| **Cross-loop KV reuse** (this note) | up to **6×** | fewer layer-caches | not started |
| Sequence-axis compression (NSA/CSA, sliding-window+sink) | `m×`, grows with length | fewer token-entries | sink code dormant |
| **KDV** (latent vs K+V) | 2× (constant) | smaller entries | shipped |

Cross-loop reuse is architecture-native: because the same physical block fires
6×, collapsing the 6 per-loop caches of a block into **one** is a natural
question that a non-recursive model cannot even ask. It **stacks** with KDV:
6× (reuse) × 2× (KDV) = **12×** vs a GQA K+V baseline. Precedent for cross-
layer KV sharing with modest degradation: CLA (Cross-Layer Attention),
YOCO (decoder-decoder, one global KV), and RRT-style recursive KV sharing.

## 4. The core tension (state it up front)

Recursion in OSRT is sold as **iterative refinement** — each loop is supposed
to *change* the representation (the recursion-for-reasoning premise,
`paper.tex` related work). If loops genuinely refine, then their keys/values
**should** differ across loops, and sharing them is lossy *by construction*.

> Cross-loop KV reuse trades against the exact thing the architecture bets on.
> That is why the evaluation gate below is the reasoning-on>off accuracy delta,
> not perplexity — a change can look ppl-neutral while quietly flattening the
> per-loop specialization that the long-reasoning goal depends on.

Mitigating prior: the per-loop `loop_embeddings` and step-specific HRA adapters
already inject most of the loop-to-loop *query-side* variation; it is an open
empirical question how much unique information lives in the per-loop **K/V**
specifically. That question is cheap to answer (§6, probe P0) before committing
to any training run.

## 5. Design space (variants, least → most aggressive)

Let `B = 3` physical blocks, `L = 6` loops. Cache today = `B·L = 18`.

1. **Grouped reuse (recompute every `g` loops).** Cache KV at loops
   `{0, g, 2g, …}`; intervening loops attend into the most recent cached latent
   for their physical block. Cache `= B·ceil(L/g)`: `g=2` → 9 caches (2×),
   `g=3` → 6 caches (3×). Safest middle ground; tunable.
2. **First-loop share (compute once, reuse `L-1`).** Each physical block
   computes its latent at loop 0 only; loops 1..5 reuse it. Cache = `B = 3`
   (**6×**), and 6× fewer `kv_down` matmuls. Most aggressive, most lossy.
3. **Last-loop / single-pass KV (YOCO-style).** A single global KV is produced
   once (e.g. a dedicated KV pass or the final loop) and *all* loops' queries
   attend into it; only `Q` recurs. Cache = `B` or even 1 global. Largest win,
   biggest architectural change.
4. **Low-rank per-loop delta.** Cache the loop-0 latent plus a small
   per-loop correction `Δ_loop` (rank `r ≪ 512`). Cache = `B·(512 + (L-1)·r)`;
   recovers most of the per-loop signal at a fraction of full re-caching.
   Best expressivity/bytes trade if P0 shows loops *do* carry distinct KV.

Recommended order: **probe P0 → variant 1 with `g=3` (3×) → variant 2 (6×) if
loss holds → variant 4 only if P0 says per-loop KV is information-rich.**

## 6. Measurement plan (cheap, before any GPU spend)

**P0 — does per-loop KV actually differ? (no training)** —
`scripts/probe_cross_loop_kv.py`. Run the mid-trained base on a held-out batch
with telemetry; per physical block, dump cross-loop cosine / linear CKA of the
`c_kv` latents, the adjacent-loop contraction series `1-CKA(k,k+1)`, and the
injected rel-L2 of each reuse scheme. If loops are near-collinear, full share is
nearly free; if they fan out, reuse is off the table. Forward-only — hours, not
dollars.

> **RESULT (2026-06-16, `midtrain_final`):** reuse is OFF — mean injected
> rel-L2 ≈ 1.04 (block 0 = 1.52, loops anti-correlated to loop 0). The loops
> carry *distinct* KV; the adjacent-CKA move size shrinks monotonically
> (block 0: 0.30, 0.17, 0.07, 0.05, 0.05) — a **contracting fixed-point
> iteration** (loop 0 = encoding pass; later loops refine less, converging by
> loops 4–5). Triangulated by the same-forward `last_loop_update_norm`
> (front-loaded |dx|/|x|, ~13→0.5) and per-loop routing telemetry. Non-degenerate
> recursion confirmed; KV reuse rejected. Live corollary: the convergence by
> loops 4–5 empirically supports the variable-loop knob (§12.2) — validate the
> drop-to-4/5 lever at the logit/accuracy level, not just representation CKA.

The probe captures all three triangulation signals (KV-CKA, residual-update
norm, per-loop routing entropy) on one forward; `--texts general|mixed` repeats
it off the math-heavy default for robustness.

**P1 — tiny-config ablation (CPU/1-GPU).**
Using `tests/test_model.py::tiny_config` (dim=128, 2 blocks, 2 loops), train
the baseline vs each variant for a fixed step budget; compare val loss/ppl.
Gate: variant is viable if `Δppl` is within a pre-registered band (e.g. ≤1%).

**P2 — the real gate (north-star).**
Any variant that passes P1 is judged on the **reasoning-on > off accuracy
delta** (GSM8K/MATH-500) at the SFT/GRPO stage, not ppl. A bandwidth win that
shrinks `Δaccuracy(on−off)` is rejected regardless of throughput.

**Throughput model (report alongside quality).**
Per-token-per-layer bytes × effective layers, and the quadratic decode total at
{2k, 4k, 8k} tokens, for baseline / KDV / KDV+variant — so the bytes-moved win
is quantified, not asserted.

| scheme | scalars/tok/layer | layers cached | KB/tok | 8k-tok cache |
|---|---|---|---|---|
| GQA K+V | 1024 | 18 | 36 | ~288 MB |
| KDV (shipped) | 512 | 18 | 18 | ~144 MB |
| KDV + grouped `g=3` | 512 | 6 | 6 | ~48 MB |
| KDV + first-loop share | 512 | 3 | 3 | ~24 MB |

## 7. Implementation sketch (if a variant graduates)

All changes localize to the forward loop and the cache plumbing; default keeps
today's 18-cache behaviour.

- **Config:** add `kv_share_mode: Literal["none","grouped","first"] = "none"`
  and `kv_share_group: int = 1` to `OSRTConfig` (`config.py`), validated.
- **Forward (`model.py:1658-1711`):** when sharing, compute `present_kv` only on
  loops where `loop % g == 0` (grouped) or `loop == 0` (first); on other loops
  pass the retained latent for that physical `block_idx` as `past_key_value`
  and skip the `kv_down` recompute. The `idx → cache slot` map changes from
  `loop*B + block_idx` to `(loop//g)*B + block_idx` (grouped) or `block_idx`
  (first), shrinking `presents` accordingly.
- **Decode (`model.py:2087, 2255`):** `past_key_values` length becomes
  `B·ceil(L/g)` (grouped) or `B` (first); the length-validation at
  `model.py:1561-1566` keys off the same formula (single source of truth).
- **Block API:** `_attention` is unchanged — it already takes `past_key_value`
  and returns `present_kv`; reuse just feeds the same retained tensor to
  multiple loops and suppresses its recompute.
- **Tests:** extend `tests/test_batched_generate.py` to assert cached-vs-uncached
  parity *per mode*, and `tests/test_model.py` for the new cache length.

## 8. Recommendation

1. Run **P0** now (forward-only, on `midtrain_final`) — it is the single cheapest
   way to learn whether the recursion actually puts distinct information in
   per-loop K/V. The answer dictates everything downstream.
2. If P0 shows collinear per-loop latents → pursue **first-loop share (6×)**.
   If it shows rich per-loop structure → pursue **low-rank delta (variant 4)**.
   If ambiguous → ship **grouped `g=3` (3×)** as the safe middle and re-probe.
3. Gate the survivor on **P2 (reasoning-on>off)**, never on ppl alone.

This is a hypothesis, not a plan of record. The prize is up to a **6× decode
bandwidth reduction on top of KDV's 2×** — directly serving the 8000-token
reasoning north star — at the risk of eroding the per-loop specialization that
same north star depends on. That risk is measurable for ~free (P0) before any
training dollars are committed, which is why this is worth doing next on the
attention axis rather than further feature-width cleverness.
