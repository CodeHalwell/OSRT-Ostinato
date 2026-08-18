# Optimizer: Muon + AdamW Hybrid

> **v7 status.** The architecture this chapter describes is current, but its
> **`file:line` citations, parameter tables and config values were written
> against v6** and have not been regenerated. mHC references have been removed
> (roadmap §12.3); expert counts, vocab and param figures may still be stale.
> Regenerate counts with `scripts/compute_budget.py`; `src/osrt/` is ground
> truth where they disagree.


> Part of the OSRT-605M `docs/` architecture series. This chapter explains the
> optimizer that trains the model: a **hybrid** that runs *Muon* (momentum
> orthogonalised by Newton-Schulz) on 2D hidden weight matrices and *AdamW* on
> everything else. Ground truth is `src/osrt/muon.py` (the whole file) and the
> optimizer wiring in `src/osrt/train.py:740-828`; default hyper-parameters live
> in `src/osrt/train_config.py`. Cross-refs: `docs/02-attention.md` (QK-Norm),
> `docs/05-recursion.md`.

---

## 1. Purpose — why a specialised optimizer

Most transformers are trained with AdamW and nothing else. OSRT instead uses a
**two-optimizer hybrid**: Muon for the big 2D weight matrices, AdamW for the
embeddings, norms, biases, and routing logits. The motivation is throughput and
stability per FLOP.

A weight matrix `W` of shape `(out, in)` is not a bag of independent scalars —
it has *matrix structure*. Its action on activations is described by its
singular value decomposition: a few dominant singular directions do most of the
work, and many rare directions contribute little. Per-parameter adaptive
optimizers (Adam, Lion) scale each scalar by its own running gradient statistics.
That scaling implicitly *shrinks* the rare singular directions — exactly the
under-used feature directions you might want to grow. Muon attacks this directly:
it takes the momentum update, **orthogonalises it** so every singular direction
gets an equal-magnitude step, and applies that. The docstring puts it plainly
(`src/osrt/muon.py:11-14`):

> That update equalises the singular spectrum, so under-represented feature
> directions get the same step size as dominant ones — Adam/Lion shrink rare
> directions by their per-parameter variance scaling.

For a sparse MoE this matters even more: cold experts recover faster because each
expert's projection sees an *orthogonal* update rather than one biased toward the
directions that already had large magnitude (`src/osrt/muon.py:17-20`).

**Lineage.** Muon ("MomentUm Orthogonalized by Newton-Schulz") comes from Keller
Jordan's *modded-nanoGPT* speedrun and the associated 2024 recipe; the
Moonshot/Kimi line of models scaled it to production training (`src/osrt/muon.py:1-4`,
`:42`). The headline claim there is roughly **~2× compute efficiency** versus
AdamW for the matrices Muon covers, at **< 1 % FLOP overhead** for the
orthogonalisation itself (`src/osrt/muon.py:15-16`). OSRT adopts it for the
attention and expert projection matrices, where that 2× lands.

---

## 2. Muon's core idea — momentum then orthogonalisation

Muon is two steps stacked:

1. **SGD momentum** (Nesterov by default) on the gradient — the *direction*.
2. **Newton-Schulz orthogonalisation** of that momentum update — reshape it onto
   the *Stiefel manifold* before applying.

Step 1 is ordinary. The interesting part is step 2. From `step()`
(`src/osrt/muon.py:169-174`):

```python
buf.mul_(momentum).add_(grad_fp32)
update = grad_fp32.add(buf, alpha=momentum) if nesterov else buf
# Newton-Schulz orthogonalisation of the update.
ortho = newton_schulz5(update, steps=ns_steps).to(dtype=p.dtype)
```

### What "all singular values → 1" means geometrically

Any matrix `M` factorises as `M = U Σ Vᵀ` (SVD): `U` and `V` are rotations, `Σ`
is a diagonal of singular values that stretch/shrink along each axis.
Orthogonalising `M` means replacing it with `U Vᵀ` — the same rotations, but with
**every singular value set to 1**. This is the *polar factor* of `M` (its nearest
semi-orthogonal matrix in Frobenius norm).

The set of matrices with all singular values equal to 1 is the **Stiefel
manifold** — the manifold of semi-orthogonal matrices. Muon projects the raw
momentum update onto it. Geometrically:

- The raw update `update` has a lopsided spectrum: some directions huge, some
  tiny. Applying it as-is means a giant step along a few axes and almost nothing
  along the rest.
- The orthogonalised update `U Vᵀ` keeps the *directions* (which way to move) but
  flattens the *magnitudes* (how far) to be equal across all axes.

So Muon's update is "move every feature direction by the same amount, in the
direction momentum says." That equal-step property is the entire point — it is
what equalises the singular spectrum of the *weights over time* and gives rare
directions a fighting chance.

Computing `U Vᵀ` exactly needs an SVD — expensive and fp32-only. Muon avoids the
SVD entirely with an iterative approximation: the Newton-Schulz iteration.

---

## 3. The Newton-Schulz iteration — the quintic

`newton_schulz5` (`src/osrt/muon.py:55-84`) approximates the polar factor with a
fixed number of matmul-only iterations. No SVD, no eigendecomposition, no fp32.

### The quintic update rule

The iteration is a degree-5 (quintic) polynomial in the matrix, applied
repeatedly. With Gram matrix `A = X Xᵀ` (`src/osrt/muon.py:80-81`):

```python
gram = x @ x.T
x = a * x + (b * gram + c * (gram @ gram)) @ x
```

i.e. `X ← a·X + (b·A + c·A²)·X`. Iterated, this pushes all singular values of
`X` toward 1 while preserving the singular *vectors* — exactly the polar factor,
without ever forming the SVD. (`A` and `A²` are symmetric, so the matmuls are
well-behaved in low precision.)

### The real coefficients and iteration count

These are the load-bearing constants, tuned by Keller Jordan to maximise the
convergence rate of `X → orthogonal(X)` (`src/osrt/muon.py:41-52`):

```python
_NS_COEFFS = (3.4445, -4.7750, 2.0315)   # (a, b, c)
_DEFAULT_NS_STEPS = 5
```

So `a = 3.4445`, `b = -4.7750`, `c = 2.0315`, and **five** iterations by default.
Note `a > 1` and `b < 0`: this is *not* a contraction toward zero — it is a
carefully shaped map whose fixed point is "all singular values = 1". The file
notes five steps is enough for bf16 gradients in practice (matches the
modded-nanoGPT speedrun); ten steps gives marginally cleaner orthogonality at
twice the cost (`src/osrt/muon.py:48-50`).

### Two preconditions before iterating

The iteration only converges if `X` starts well-conditioned, so two things
happen first (`src/osrt/muon.py:70-73`):

```python
x = g.to(dtype=torch.bfloat16)         # run in bf16
x = x / (x.norm() + 1e-7)              # normalise spectral norm ≈ 1
```

- **bf16.** The whole iteration runs in bf16. bf16 matmuls are ~2× the
  throughput of fp32 on H100 tensor cores, and NS is well-conditioned enough that
  the low precision doesn't hurt once `X` is normalised. This is why the
  orthogonalisation stays under 1 % of forward+backward FLOPs.
- **Normalisation.** Dividing by the Frobenius norm pulls the largest singular
  value to roughly 1 before iterating; otherwise the first step can diverge.

### The smaller-Gram-side transpose trick

The cost of the iteration is dominated by `X Xᵀ`. For an `m × n` matrix that
Gram product is `m × m`. If `m > n`, you'd rather compute the *other* Gram
`Xᵀ X`, which is `n × n` — smaller, hence cheaper. The code transposes to the
smaller side before iterating and transposes back after
(`src/osrt/muon.py:74-83`):

```python
transposed = x.size(0) > x.size(1)
if transposed:
    x = x.T
for _ in range(steps):
    gram = x @ x.T
    x = a * x + (b * gram + c * (gram @ gram)) @ x
if transposed:
    x = x.T
return x.to(dtype=g.dtype)
```

The polar factor of `Xᵀ` is the transpose of the polar factor of `X`, so this is
exact, not an approximation — it only changes which dimension the FLOPs scale
with. Combined with bf16, this is what keeps the overhead negligible.

### The shape-aware scale (with a caveat)

After orthogonalisation Muon multiplies the step by a shape factor
(`src/osrt/muon.py:179-180`):

```python
rows, cols = p.shape
shape_scale = max(1.0, rows / cols) ** 0.5
```

This makes the per-element step variance match what an Adam-scale optimizer would
produce, so the same LR schedule is sane across matrices of different aspect
ratio (Keller Jordan's heuristic, `src/osrt/muon.py:28-31`). **Caveat / code
note:** the inline comment (`src/osrt/muon.py:176-178`) says fat matrices
(`rows < cols`) are *shrunk*, but `max(1.0, rows/cols)` returns exactly `1.0` for
fat matrices — they are left unchanged; only *tall* matrices (`rows > cols`) get
scaled up. The comment is backwards relative to the arithmetic; the behaviour is
"tall matrices step bigger, everything else unchanged."

---

## 4. The hybrid split — what goes to Muon vs AdamW

Muon is only valid for 2D matrix parameters of *hidden* layers. Embeddings,
norms, biases, and scalars must use AdamW — orthogonalisation is the wrong
operator on a lookup table or a per-channel gain. The split is decided in one
reviewable place, `build_param_groups` (`src/osrt/muon.py:241-309`), rather than
scattered through the training loop. The `Muon.__init__` even validates that
every param it receives is 2D and raises loudly otherwise
(`src/osrt/muon.py:131-138`).

The **rule of thumb** is: *2D hidden matrices → Muon; embeddings / norms / biases
/ scalars → AdamW.* But the exact routing has carve-outs that matter, and two of
them contradict the "obvious" description — read these carefully.

### The routing logic (in order)

`build_param_groups` iterates `named_parameters()` and applies, in order
(`src/osrt/muon.py:270-302`):

1. **Router-like names → AdamW, wd=0** (`src/osrt/muon.py:274,277-282`). This is
   checked *first*, before the 2D test:
   ```python
   is_router_like = ("router" in name) or ("loop_embeddings" in name)
   if is_router_like:
       adamw_no_decay.append(param)
       continue
   ```
2. **Norms / scalars / embeddings → AdamW, wd=0** (`src/osrt/muon.py:284-291`).
3. **Remaining 2D params → Muon** (`src/osrt/muon.py:293-294`).
4. **ndim > 2 → hard error** (`src/osrt/muon.py:296-302`) — OSRT has none, but a
   future conv would force an explicit decision rather than silently slipping
   through.

### What actually lands where

| Parameter class | Example | Optimizer |
|---|---|---|
| Attention projections (q / kv / v / out) | `Linear.weight` in the attn block | **Muon** |
| MoE expert SwiGLU (gate / up / down) | per-expert projections | **Muon** |
| HRA adapter weights | `src/osrt/hra.py` | **Muon** |
| Token embedding (and tied LM head) | `nn.Embedding` | **AdamW**, wd=0 |
| RMSNorm gains, incl. QK-Norm | 1D scales | **AdamW**, wd=0 |
| **Router weight** `self.router` | `nn.Linear(dim, num_routed, bias=False)`, `model.py:215` | **AdamW**, wd=0 |
| `loop_embeddings` | `nn.Embedding`, `model.py:213` | **AdamW**, wd=0 |
| `router_balance_bias` | registered buffer, `model.py:237` | **no optimizer** (heuristic) |

This matches `ARCHITECTURE.md` §16.7 (Muon = attention / experts / HRA;
AdamW = embedding, LM head, norms, biases; router bias = heuristic only).

### Two surprises worth flagging (code beats the docstring)

**(a) The router weight is a 2D matrix, but it goes to AdamW, not Muon.** The
router is `nn.Linear(config.dim, self.num_routed, bias=False)`
(`src/osrt/model.py:215`) — a genuine 2D weight named `…router.weight`. Because
`"router" in name` is tested *before* the 2D check
(`src/osrt/muon.py:274,277-282`), it is routed to AdamW with `wd=0`. This is
**deliberate** — the routing logits and the bias controller shouldn't be fighting
weight decay, and orthogonalising the router can destabilise expert balance — but
it **contradicts** `muon.py`'s own module docstring, which lists "MoE (router,
expert SwiGLU projections)" among the matrices Muon orthogonalises
(`src/osrt/muon.py:6,11`). Trust the code: **router weight → AdamW**. (Distinct
point: `router_balance_bias` is a *buffer* updated by a heuristic EMA rule at
`model.py:379-382`, and is in **no** optimizer at all — don't conflate the two.)

**(b) `loop_embeddings` is grouped with the router, not with the embedding rule.**
It is caught by the `is_router_like` branch (`src/osrt/muon.py:274`) and gets
AdamW wd=0, matching the Lion config carve-out so the depth-conditioning table
isn't decayed (cross-ref `docs/05-recursion.md`).

---

## 5. Decoupled weight decay — why it's mandatory with Muon

Weight decay in Muon is **decoupled** (AdamW-style): it multiplies the parameter
directly, before the update lands, rather than being folded into the gradient
(`src/osrt/muon.py:182-186`):

```python
if wd != 0.0:
    p.mul_(1.0 - lr * wd)
p.add_(ortho, alpha=-lr * shape_scale)
```

So each step does `p ← p·(1 − lr·wd)` then `p ← p − lr·shape_scale·ortho`.

### Why this is *critical* with Muon specifically

With Adam, the update magnitude is tied to the gradient magnitude, so a weight
that drifts large tends to get larger updates and self-corrects somewhat. **Muon
removes that feedback.** The orthogonalised update has all singular values ≈ 1
*regardless of how big the weight already is* — the step size does not shrink as
the weight grows. Nothing in Muon's own update naturally scales weight magnitude
down. Left unchecked, the spectral norms of Muon-trained matrices **drift
upward** over training, which destabilises attention logits and the MoE router
downstream.

Decoupled weight decay is the counterweight: the `(1 − lr·wd)` factor pulls every
weight geometrically toward zero each step, independent of the gradient, capping
the spectral drift. This is why the Muon recipe treats decoupled WD as part of
the algorithm, not an optional regulariser (`ARCHITECTURE.md` §16.7: "Weight
decay applied via decoupled scheme (Muon paper)").

### Surprise: in OSRT, *all* the weight decay is Muon's

There is a bug-shaped quirk in `build_param_groups`. It intends to give AdamW two
groups — a decayed group for 2D non-embedding params and a non-decayed group for
norms/embeddings (`src/osrt/muon.py:245-263` docstring). But the routing makes
the decay group **unreachable**: the outer guard
`if is_norm_or_scalar or is_embedding:` (`src/osrt/muon.py:284`) and the inner
`if is_embedding or is_norm_or_scalar:` (`src/osrt/muon.py:287`) are the *same*
condition, so the `else: adamw_decay.append(...)` branch (`src/osrt/muon.py:290`)
is dead code. `adamw_decay` is never populated, so only the wd=0 AdamW group is
built (`src/osrt/muon.py:304-308`).

Net effect, and the thing to internalise: **AdamW runs with `wd=0` on
everything; the only place weight decay actually bites is Muon's decoupled term**
(`src/osrt/muon.py:185`, scaled by `muon_lr`, not `peak_lr`). The
`train.py` startup print "wd=… on non-norm/non-embed only"
(`src/osrt/train.py:789-790`) describes the *intended* behaviour, not the
realised one. In practice this is fine — the params that benefit from decay (the
big matrices) are exactly the ones on Muon — but it means there is no separate
AdamW-decayed group despite the docstring.

---

## 6. The fp32 momentum buffer fix — precision matters

The momentum buffer is forced to **fp32**, and the gradient is cast to fp32
before it accumulates into the buffer (`src/osrt/muon.py:159-170`):

```python
buf = state.get("momentum_buffer")
if buf is None:
    buf = torch.zeros_like(grad, dtype=torch.float32)   # force fp32
    state["momentum_buffer"] = buf
...
grad_fp32 = grad.to(dtype=torch.float32)
buf.mul_(momentum).add_(grad_fp32)
update = grad_fp32.add(buf, alpha=momentum) if nesterov else buf
```

The comment (`src/osrt/muon.py:164-167`) records the bug this fixes: a plain
`torch.zeros_like(grad)` inherits the gradient's dtype — and OSRT gradients are
**bf16**. A bf16 momentum buffer accumulates roundoff error over millions of
steps, contradicting the intent that the buffer be fp32. With `momentum=0.95`,
the buffer is a long EMA over the gradient history; bf16's ~3 decimal digits of
mantissa quietly lose the small-gradient tail of that EMA. The fix:

1. Allocate the buffer fp32 (`:161`).
2. Cast the gradient to fp32 before accumulating, so the EMA math is fp32
   throughout (`:168-169`).
3. Run Newton-Schulz (which casts to bf16 internally) on the fp32 update, then
   cast the orthogonalised result back to the parameter dtype before the in-place
   apply (`:174`).

So precision is high where it accumulates (the buffer) and low where it's cheap
and safe (the matmul iteration) — the orthogonalisation tolerates bf16, the
momentum EMA does not.

---

## 7. Interaction with the rest of the architecture

Muon is fast and FLOP-efficient, but it is *aggressive*: its equal-singular-value
update can rotate a weight matrix's representation substantially in a single
step. In a plain transformer that is tolerable. In OSRT it is amplified by
**recursion** — the same 3 physical blocks run 6 times
(`docs/05-recursion.md`) — so a representation shift propagates through six passes
per token. Muon + recursion can change representations *too fast per loop* if
nothing bounds it. Two architectural features are what make Muon viable here:

- **QK-Norm** (`docs/02-attention.md` §5.1). Muon-trained models are known to be
  prone to attention-logit growth (the rising spectral norms of §5 feed straight
  into `q·k`). QK-Norm normalises q and k to unit RMS *before* the dot product,
  so `|q·k|` cannot run away no matter how the projection weights drift. The
  attention doc states this dependency outright: "this matters more here than
  usual because the model is Muon-trained, and Muon-trained models are known to
  be prone to logit growth" (`docs/02-attention.md:120`; Kimi K2 added "QK-Clip"
  on top — OSRT relies on QK-Norm plus Muon's own stability). QK-Norm gains are
  1D, so they themselves are trained by **AdamW**, not Muon.

- **Sandwich RMSNorm, per-loop aux losses and loop dropout** bound how much a
  fast Muon update to one block can perturb the shared residual that all six
  loops read and write.

> **Corrected in v7.** Earlier revisions of this chapter claimed Muon's
> viability was *contingent* on mHC bounding the residual mixing. That was an
> assertion, never a measurement. mHC's benefit against a plain residual stream
> was never established at any scale (roadmap §12.1-C2 — the paper's headline
> +2.1 BBH is mHC-vs-HC, not mHC-vs-plain), and v5 ran Muon over 18 effective
> layers on a plain residual stream without loop collapse. mHC was removed in
> v7 (§12.3). Muon's real preconditions here are QK-Norm on the logits and the
> sandwich-norm / aux-loss / loop-dropout stack on the recursion.

---

## 8. Practical — wiring and learning rates

The hybrid is selected by `optimizer_name == "muon"` (the default) in
`train.py:747`; `"lion"` and AdamW-everything are the fallbacks
(`src/osrt/train.py:792-828`). Construction (`src/osrt/train.py:754-783`):

```python
muon_params, adamw_groups = build_param_groups(
    inner_model.named_parameters(), weight_decay=train_cfg.weight_decay,
)
muon = Muon(muon_params, lr=muon_lr, momentum=0.95,
            nesterov=True, weight_decay=train_cfg.weight_decay)
adamw = torch.optim.AdamW(adamw_groups, lr=train_cfg.peak_lr,
                          betas=(0.9, 0.95), eps=1e-8)
optimizer = HybridMuonAdamW(muon, adamw)
```

### Two learning rates, one schedule

Muon's effective step is much smaller-magnitude than AdamW's per-parameter scale
(the NS update is normalised), so **Muon uses a much larger LR** — roughly 30–50×
the AdamW LR (`src/osrt/muon.py:94-97`). The two are kept as separate config
knobs so you can A/B Muon without touching the AdamW/Lion peak. Pretrain defaults
(`src/osrt/train_config.py`):

| Knob | Value | Applies to |
|---|---|---|
| `peak_lr` / `min_lr` | `6e-4` / `6e-5` | AdamW groups |
| `muon_lr` / `muon_min_lr` | `0.02` / `2e-3` | Muon group |
| `weight_decay` | `0.3` | Muon decoupled term (AdamW effectively wd=0, §5) |
| AdamW `betas` / `eps` | `(0.9, 0.95)` / `1e-8` | `train.py:772-773` |
| Muon `momentum` / `nesterov` | `0.95` / `True` | `train.py:765-766` |

Both LRs ride the **same cosine-with-warmup shape** but to their own
peak/floor. Each group is tagged with `_peak_lr` / `_min_lr` at construction
(`src/osrt/train.py:776-782`), and `_set_param_group_lrs`
(`src/osrt/train_config.py:57-93`) writes the per-group LR each step, honouring
those tags. So one schedule call drives Muon to `0.02 → 2e-3` and AdamW to
`6e-4 → 6e-5` simultaneously. Fine-tuning stages scale both down together: e.g.
the first extend stage uses `peak_lr 1.5e-5` with `muon_lr 5e-3`
(`src/osrt/train_config.py:349,364`), and GRPO/MOPD stages step them down further
in lockstep.

### The HybridMuonAdamW wrapper

`HybridMuonAdamW` (`src/osrt/muon.py:191-238`) makes the pair look like a single
`torch.optim.Optimizer` to the training loop:

- `.param_groups` concatenates both optimizers' groups
  (`src/osrt/muon.py:214-216`), so one LR-schedule pass touches both.
- `.zero_grad()` and `.step()` fan out to both; `.step()` runs **Muon first, then
  AdamW** (`src/osrt/muon.py:222-228`). Order is irrelevant for correctness — the
  two touch disjoint params — but Muon's NS iteration is the longer step, so it's
  kicked off first for a tidier wall-clock profile.
- `.state_dict()` / `.load_state_dict()` use one dict with `"muon"` and
  `"adamw"` sub-keys (`src/osrt/muon.py:230-238`), so checkpoints stay one file.
  `train.py`'s resume path wraps the load in try/except and starts the optimizer
  fresh on a type mismatch, so swapping optimizer mid-run (e.g. Lion → Muon)
  doesn't break resume.

---

## Summary

- Muon orthogonalises the SGD-momentum update via a 5-step Newton-Schulz quintic
  in bf16 (`coeffs (3.4445, -4.7750, 2.0315)`), projecting onto the Stiefel
  manifold so every singular direction gets an equal step. ~2× compute
  efficiency, < 1 % FLOP overhead.
- The hybrid sends 2D hidden matrices (attention, experts, HRA) to Muon and
  everything else to AdamW. **Two code-vs-docstring surprises:** the router
  *weight* is 2D but routed to AdamW (wd=0), and the AdamW "decay" group is dead
  code so all real weight decay comes from Muon's decoupled term.
- Decoupled WD `(1 − lr·wd)` is mandatory because Muon's normalised update never
  scales weights down on its own — without it spectral norms drift.
- The momentum buffer is forced fp32 (a fix for it inheriting bf16 from grads);
  NS runs bf16, the EMA stays fp32.
- Muon's preconditions here are QK-Norm bounding the logits and the
  sandwich-norm / aux-loss / loop-dropout stack bounding the recursion
  (`docs/02-attention.md`, `docs/05-recursion.md`). The older claim that mHC was
  required is corrected above.
