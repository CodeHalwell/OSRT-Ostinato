# HRA — High-Rank Adapters

> **v7 status.** The architecture this chapter describes is current, but its
> **`file:line` citations, parameter tables and config values were written
> against v6** and have not been regenerated. mHC references have been removed
> (roadmap §12.3); expert counts, vocab and param figures may still be stale.
> Regenerate counts with `scripts/compute_budget.py`; `src/osrt/` is ground
> truth where they disagree.


*Part of the `docs/` OSRT-605M architecture series.*

This document explains the **HRA (High-Rank Adapter)** block of OSRT-605M:
what it is, the (honest) story behind the name, the math, where the 18
adapters actually live, how they are sized, and how they are trained — at
pretraining time and during RL. It is grounded in the source: every
non-obvious claim cites `file:line`.

---

## 1. Purpose — capacity, not parameter efficiency

LoRA exists to make fine-tuning *cheap*. You freeze a big pretrained model
and bolt on a tiny rank-4-to-64 `A@B` delta so you can adapt with almost no
new parameters and almost no new compute. The whole point is **efficiency**.

OSRT's HRA borrows LoRA's *form* — a parallel `A@B` path added to a linear
layer — but inverts its *intent*. The adapter rank here is **256**, not 16.
At rank 256 the adapter is no longer a cheap correction term; it is a
substantial, full-time learning subspace that the model can grow into. The
argument is **capacity**:

- OSRT is a *recursive* transformer: 3 physical blocks are unrolled 6 times
  to give 18 effective layers (`ARCHITECTURE.md:131`). Weight sharing across
  loops keeps the parameter count down but forces every loop to reuse the
  same block weights.
- A high-rank adapter that is **distinct per (block, loop)** gives each of
  the 18 effective layers its own learnable delta on top of the shared
  block. It buys back per-layer expressivity that pure weight-tying would
  otherwise cost, at a fraction of the cost of un-sharing the blocks.

So HRA is best read as "per-loop high-rank residual capacity," not as a
parameter-efficient fine-tuning trick.

---

## 2. The honest naming note

Two things about the "HRA" name are worth stating plainly, because the code
does not quite match the label.

**(a) It is not a Householder / hyperspherical reparameterization.** In the
literature "HRA" sometimes denotes *Householder Reflection Adaptation*, a
method that builds the adapter from a chain of Householder reflections so the
update is orthogonal/norm-preserving. **OSRT does none of that.** The forward
is a plain matrix product `x @ A @ B` — see `hra.py:71` and `model.py:996`.
There is no reflection, no orthogonality constraint, no hyperspherical
parameterization. It is a **high-rank LoRA-form adapter**, full stop. Calling
it "HRA" is a name, not a description of the parameterization.

**(b) The name is overloaded across two real mechanisms.** OSRT actually
contains *two distinct* `A@B` adapter systems, and they are easy to confuse
(the code itself documents a bug born of confusing them — `train.py:1263`):

1. **The 18 inline per-loop adapters** — created in the model constructor,
   one pair per effective layer, applied on the attention path. The code
   calls them "Per-pass low-rank adapters" (`model.py:1254`). **These are the
   canonical "HRA adapters" the architecture doc counts**, and the main
   subject of this document.
2. **The injected `HRALinear` adapters** — defined in `hra.py`, monkey-patched
   onto a *pretrained checkpoint* at SFT/RL time across seven projection
   modules per layer. This is a retrofit path (section 7).

Both use the identical math. They differ in *where* they are attached, *how
many* there are, and *when* they come into existence. Keeping them separate is
the key to reading the rest of this doc — and to reading `ARCHITECTURE.md`,
where §2.4 describes mechanism (1) and §5.4 describes mechanism (2).

---

## 3. The adapter math and the zero-init trick

The reference implementation is `HRALinear` in `hra.py`. Its forward is the
whole idea in three lines (`hra.py:69-72`):

```python
def forward(self, x: Tensor) -> Tensor:
    base_out = self.original(x)
    hra_out = (x @ self.adapter_a) @ self.adapter_b
    return base_out + self.scale * hra_out
```

So the layer computes

```
y = base(x) + scale · (x @ A @ B)
```

where `A` has shape `(in_features, rank)` and `B` has shape
`(rank, out_features)`. The `x @ A @ B` product is rank-bounded by `rank`,
which is exactly why a high rank (256) buys real capacity.

The 18 inline adapters implement the *same* expression directly inside the
attention method (`model.py:996`):

```python
adapter_out = adapter_scale * (x_in @ adapter_a @ adapter_b)
```

and fold it into the attention output residual (`model.py:1055`):

```python
return self.out_proj(attn_out) + adapter_out, present_kv
```

### Why B starts at zero

Both mechanisms initialize `A` to small random values and `B` to **zeros**:

- Inline (the 18): `A = randn(dim, rank) * 0.01`, `B = zeros(rank, dim)`
  (`model.py:1257-1262`).
- Injected (`HRALinear`): `A` is Kaiming-scaled, `randn(in, rank) *
  (2/in_f)**0.5`; `B = zeros(rank, out)` (`hra.py:58-63`).

The init constant differs between the two implementations — that is simply
two authors making two reasonable choices — but the load-bearing detail is
shared: **`B = 0`**. With `B = 0`, the adapter output `x @ A @ B` is exactly
zero at step 0, so `y = base(x)` and the adapter is a **no-op identity
residual** on the first forward pass.

That matters because HRA is injected from day one (section 6). Starting as a
clean no-op means adding the adapter does not perturb the model's initial
function at all — the adapter only ever *adds* signal as `B` learns away from
zero. (The inline `A = randn * 0.01` is also deliberately tiny, so even once
`B` moves, the early adapter contribution grows gently rather than shocking
the residual stream.)

---

## 4. Where the 18 adapters live (the corrected §2.4 enumeration)

There is **one rank-256 adapter pair per effective layer**: 3 blocks × 6
loops = **18 pairs** (`ARCHITECTURE.md:166-168`). They are stored as two
parallel `ParameterList`s on the model (`model.py:1254-1264`):

```python
# Per-pass low-rank adapters
total_pairs = config.num_blocks * config.recursive_loops   # 3 × 6 = 18
self.adapters_a = nn.ParameterList(
    [nn.Parameter(torch.randn(config.dim, config.adapter_rank) * 0.01)
     for _ in range(total_pairs)]
)
self.adapters_b = nn.ParameterList(
    [nn.Parameter(torch.zeros(config.adapter_rank, config.dim))
     for _ in range(total_pairs)]
)
self.adapter_scale = config.adapter_alpha / config.adapter_rank
```

The decode loop picks the right pair by flattening `(loop, block)` into a
single index and threading it into the block (`model.py:1463-1465`):

```python
idx = loop * self.config.num_blocks + block_idx
adapter_a = self.adapters_a[idx]
adapter_b = self.adapters_b[idx]
```

So loop 0 / block 0 gets adapter 0, loop 0 / block 1 gets adapter 1, …, loop
5 / block 2 gets adapter 17. Each effective layer has a genuinely distinct
adapter even though it shares block weights with the other five loops.

**It is one parallel path per block forward — NOT per-projection.** An early
draft of `ARCHITECTURE.md` (and §5.4, which still describes the *other*
mechanism) envisioned 87 or 132 injection points spread across Q/K/V/O, every
expert, and the router. The canonical inline mechanism is far simpler: **18
attention-path adapters** (`ARCHITECTURE.md:164-184`, corrected in the
errata at `ARCHITECTURE.md:1464`). The 87/132 numbers are not nonsense — they
correctly describe the injected `HRALinear` path of section 7 — but they are
not what the 18 are.

**Attention path, not the routed experts.** The adapter is added to the
attention sub-block output (`model.py:996`, `model.py:1055`). That means it
sits on the **always-run** part of the layer, *not* on the sparse top-2 MoE
experts. There is no top-k masking of HRA: every token sees every adapter.

---

## 5. `adapter_scale = alpha / rank`

The adapter output is multiplied by a fixed scalar before being added to the
residual (`model.py:1264`):

```python
self.adapter_scale = config.adapter_alpha / config.adapter_rank
```

This is the same `alpha/rank` convention LoRA uses. It decouples *how big the
adapter's contribution is* from *how high its rank is*: if you change `rank`
to give the adapter more capacity, the scale automatically compensates so the
typical magnitude of `scale · x@A@B` does not blow up.

For the canonical 605M preset (`presets.py:35-36`):

```python
adapter_rank=256,      # real HRA capacity (NOT LoRA-style 16)
adapter_alpha=256.0,   # match rank so scale = 1.0
```

`alpha = rank = 256`, so `adapter_scale = 256 / 256 = 1.0`. The adapter
contribution passes through un-attenuated. (Note: the *defaults* in
`config.py:52-53` are `rank=16, alpha=16` — also scale 1.0 but a LoRA-sized
adapter; the 605M preset deliberately overrides both to 256. See section 8 for
why the rank choice changes the parameter count by 16×.)

---

## 6. Day-1 injection vs retrofit

A central design decision: **HRA is part of the architecture from the first
pretraining step**, not bolted on afterward.

The 18 inline adapters are created in `RecursiveOSRT.__init__`
(`model.py:1254`), so they exist before any data is seen and are trained
end-to-end alongside the base weights through pretraining, mid-training, and
SFT. The model learns its block weights *in the presence of* the per-loop
adapters, so the two co-adapt; the adapters are not a foreign correction term
stapled on at the end.

Contrast this with the retrofit path. `hra.py`'s `inject_hra` exists precisely
to add adapters to a model that was *not* born with them: it walks the module
tree and replaces target `nn.Linear`s with `HRALinear` wrappers after the fact
(`hra.py:91-145`). OSRT's SFT/RL stages do exactly this — they inject HRA into
a *pretrained checkpoint* before loading weights (`train.py:1338-1350`):

```python
# ── HRA injection (BEFORE state_dict load) ──
if extend_cfg.hra_enabled:
    inject_hra(model, rank=extend_cfg.hra_rank, scale=..., freeze_pretrained=False)
```

The zero-init `B` (section 3) is what makes both paths safe: whether an
adapter is born with the model or grafted onto a trained checkpoint, it begins
as an exact identity residual and disturbs nothing until it learns.

Why prefer day-1 for the core 18? Because adapters trained from scratch
alongside the blocks can shape *what the blocks learn*; a retrofit can only
correct a frozen function after the fact. The retrofit path is reserved for
adding **fresh, freezable capacity** at post-training time (next section).

---

## 7. HRA-only RL (GRPO): freeze the base, train the adapters

During RL, OSRT uses the classic "adapters-only" recipe — the same pattern
DPO/PPO/GRPO commonly use with LoRA. The base weights are frozen and only the
adapter delta receives gradient. In OSRT-605M's GRPO-v2 this is the **central
architectural fix** (`train_config.py:1842-1866`):

```python
hra_only_training: bool = True
```

The rationale, quoted from the config (`train_config.py:1845-1855`):

- The MOPD/SFT capability anchor is **structurally preserved** — the frozen
  base contribution to the logits cannot drift.
- KL drift is **bounded by construction**: since `y = base(x) + scale·x@A@B`
  and `base` is fixed, only the additive adapter term can move.
- ~4× fewer parameters carry Adam state → faster, less memory.
- Lower risk of catastrophic forgetting.

The trade-off is honestly noted in the same comment: a rank-256 low-rank delta
may not be able to reach capabilities that genuinely need base-weight surgery
(`train_config.py:1857-1862`). This recipe was adopted after a step-75→150
capability regression in GRPO-v1, where unconstrained base drift under
policy-gradient pressure was eroding distilled capabilities.

### The differential-LR alternative

The `hra.py` docstring shows a lighter-touch option: instead of freezing the
base, train **both** groups but give the adapters a much higher learning rate
(`hra.py:16-20`):

```python
# Differential LR: lower for pretrained, higher for HRA
optimizer = AdamW([
    {"params": base_params, "lr": 2e-5},
    {"params": hra_params,  "lr": 1e-4},
])
```

`get_param_groups` (`hra.py:148-191`) builds exactly these two groups,
splitting parameters by identity so the adapters get `hra_lr` (typically 5–10×
base) and everything else gets `base_lr`. OSRT's SFT stages set `hra_lr`
explicitly (e.g. `train_config.py:1028-1030`). So there are three distinct
knobs, and they must not be conflated:

| Knob | Effect | Source |
| --- | --- | --- |
| `hra_only_training=True` | freeze base, train adapters (GRPO-v2 default) | `train_config.py:1866` |
| differential LR | train both, higher LR on adapters | `hra.py:148-191` |
| `hra_frozen=True` / `_freeze_hra_params` | the *opposite*: freeze the **injected** adapters | `train.py:1256-1287` |

The third is a footgun: `_freeze_hra_params` deliberately matches only the
**singular** `.adapter_a` / `.adapter_b` (the injected `HRALinear` tensors) and
**not** the plural `model.adapters_a` (the 18 inline ones). A previous version
used a substring match, froze the wrong 884k params, and silently left the
intended tensors trainable (`train.py:1263-1271`). The lesson: the two
mechanisms share a name but not a namespace.

---

## 8. Parameter and active cost

For the canonical 605M preset (rank 256), each adapter pair is:

```
adapter_a : dim × rank = 1,536 × 256 = 393,216
adapter_b : rank × dim = 256 × 1,536 = 393,216
            -------------------------------------
per pair  : 786,432
```

Across all 18 pairs (`ARCHITECTURE.md:105-110`, `ARCHITECTURE.md:181`):

```
18 × 786,432 = 14,155,776 params  (~14.16M)
```

`compute_budget.py` confirms this independently. It buckets any parameter
whose name contains `"adapter"` into the `adapters` category
(`compute_budget.py:36`) and reports it as both physical and active:

```
HRA adapters (rank 256, 18 injection points)   14,155,776   14,155,776
```

**All 14.16M are active per token.** In `active_per_token`, only the routed
experts are scaled down by the sparse fraction and only the MTP heads are
dropped; everything else — adapters included — counts in full
(`compute_budget.py:61-75`). This follows directly from section 4: the
adapters live on the always-run attention path, so there is no sparsity to
discount.

### Don't mix up the three numbers

Three adapter parameter counts float around the codebase; each belongs to a
different mechanism or config, and only the first is "the architecture":

- **14.16M** — the 18 inline adapters at **rank 256** (canonical preset). Use
  this for the architecture budget.
- **884k** — the 18 inline adapters at the **rank-16 config default**
  (`config.py:52`); the number the `train.py:1268` comment references.
- **~86M** — the **injected** `HRALinear` path at rank 256, spread across
  seven projections per layer at SFT/RL time
  (`train_config.py:1846`, `ARCHITECTURE.md:314`). This is the GRPO-stage
  delta, not the day-1 architecture.

---

## 9. Caveats and future directions

**It is not "true" HRA today.** As section 2 says, the parameterization is
plain `A@B`. A genuine Householder-reflection HRA (orthogonal, norm-preserving
updates) could be dropped in behind the same `y = base + scale·x@A@B`
interface with no change to where the 18 adapters attach. Whether that buys
anything at rank 256 is an open question — the orthogonality benefits of HRA
in the literature are most pronounced at low rank.

**Attention-path only.** The 18 adapters currently sit only on the attention
sub-block. The injected path already shows the alternative — per-projection
adapters across Q/K/V/O, expert gate/up/down, and the router
(`ARCHITECTURE.md:380`). Moving (some of) the inline 18 to a per-projection
layout would trade a larger parameter count for finer-grained capacity; it is
a deliberate simplicity-vs-capacity choice, not an oversight.

**Two mechanisms, one name.** The cleanest future cleanup is terminological:
the "Per-pass low-rank adapters" (`model.py:1254`) and the injected
`HRALinear` retrofit are different enough that calling both "HRA" invites
exactly the freeze-the-wrong-tensors bug the code already had to guard against
(`train.py:1263-1271`). Distinct names would make the architecture easier to
reason about.

**Deployment folding.** Because the adapter is a linear residual, post-RL it
can be folded into the base weights (`W ← W_base`, then the attention-path
delta absorbed) to eliminate its separate memory and compute at inference, or
quantized alongside the base (`ARCHITECTURE.md:1168`). The day-1, always-active
design does not force a permanent inference cost.

---

### One-line summary

OSRT's "HRA" is an honest high-rank (256) LoRA-*form* adapter — plain `x@A@B`,
zero-init `B`, 18 of them (one per effective layer) on the attention path,
14.16M fully-active params (`compute_budget.py`), born day-1 and trained
adapters-only during GRPO — not a Householder reparameterization, and not to
be confused with the separate per-projection `inject_hra` retrofit.
