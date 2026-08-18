# Manifold-Constrained Hyper-Connections (mHC)

*Part of the OSRT-605M `docs/` architecture series. This chapter explains the
single most mathematically subtle block in the model. Ground truth is
`src/osrt/mhc.py` and the mHC integration in `src/osrt/model.py`; design intent
is `ARCHITECTURE.md` §8. Where my prose and the code disagree, the code wins —
discrepancies are flagged inline.*

---

## 1. Purpose — why deep recursive residual stacks need this

OSRT is a *recursive* transformer: 3 physical blocks are run 6 times in a loop,
giving **18 effective layers** (`ARCHITECTURE.md:131`). Recursion is how the
model gets depth for cheap — the same weights are reused, so 18 layers cost the
parameters of 3. But weight reuse turns a benign property of a single layer into
a compounding one across the loop.

A plain residual (`x ← x + F(x)`) adds a contribution per sub-block, so the
per-layer gain on the residual stream is exactly 1 — composing 18 of them
neither amplifies nor vanishes. But the moment you want a *learned* mix of the
residual stream (multiply by some matrix instead of just adding), you reintroduce
a multiplicative per-layer gain, and across 18 reused layers it compounds. A gain
of 1.5 per layer amplifies by `1.5 ** 18 ≈ 1,478×` (a straight road to
`inf`/`NaN` in bf16); a gain of 0.9 *vanishes* (`0.9 ** 18 ≈ 0.15`). The window
between explode and vanish is narrow and narrows with depth.

mHC's whole reason for existing is to give the model a **learned, per-token,
dynamic residual mix** while *mathematically guaranteeing* the per-layer
residual transform has spectral norm ≤ 1, i.e. a gain of at most 1
(`1 ** 18 = 1`: no amplification, no vanishing). The model learns rich routing of
information between residual channels; the optimizer never walks a tightrope to
keep training finite. That guarantee — turning a potential 1,478× amplifier into
a non-expansive 1.0× map — comes from a single algebraic fact
(doubly-stochastic ⇒ spectral norm ≤ 1) that §4 makes precise.

> The module docstring states the goal directly: B is projected onto the
> Birkhoff polytope "which guarantees ‖B_l‖₂ ≤ 1 — the residual transform is
> non-expansive, so the 18 effective layers stay numerically stable"
> (`src/osrt/mhc.py:11-12`).

---

## 2. The n_hc-channel residual stream

A standard transformer carries one residual vector per token:
`x ∈ (B, S, D)` with `D = 1536`. mHC **widens the residual stream** to `n_hc = 4`
parallel channels (`config.n_hc`, `src/osrt/config.py:60`):

```
X ∈ (B, S, n_hc, D)   # (batch, seq, 4, 1536)
```

The expansion happens at the embedding, by replicating the embedded token across
the 4 channels (`src/osrt/model.py:1351-1355`):

```python
x = self.embedding(input_ids)
if self.use_mhc:
    # .repeat (not .expand) so the channels are independent storage — §10 Bug 1.
    x = x.unsqueeze(2).repeat(1, 1, self.config.n_hc, 1)
```

### What the 4× expansion *is*

It is four **independent storage slots** for residual information. The block can
keep distinct running summaries in different channels — e.g. one channel a
"main" residual, another a slower-moving context, another a scratch lane — and
mix them per token as it sees fit. This is the "hyper-connection" idea
(Zhu et al.): replace the single residual highway with several, connected by a
learned mixing matrix.

### What the 4× expansion is **not**

It does **not** make attention or the MoE 4× wider. The inner layers are
completely unchanged: they consume a single `D`-dim vector and produce a single
`D`-dim vector. mHC sits *around* them. Concretely, before a sub-block runs, mHC
collapses the 4 channels into one `D`-dim **input view** (the `A` matrix, §3);
after the sub-block produces its single `D`-dim output, mHC writes that output
back into the 4 channels (the `C` matrix). The attention/MoE code never sees
`n_hc`.

> Why `.repeat` and not `.expand`? `.expand` returns a *view* — all four
> channels would alias the same memory, and an in-place write to one would
> corrupt the others. This was "Bug 1" in an early design draft
> (`src/osrt/model.py:1353-1354`, `ARCHITECTURE.md:793-795`); the real code uses
> `.repeat` to get four genuinely separate buffers.

The cost of widening is paid *only* on the residual tensor and the small mixing
matrices — not on the expensive attention/expert matmuls. That is what keeps the
4× expansion affordable (§9).

---

## 3. The three mixing matrices A / B / C and the update rule

Each time a sub-block runs, mHC generates **three** small per-token matrices
from the current residual stream (`generate`, `src/osrt/mhc.py:77-90`):

| matrix | shape (per token) | range | role |
|--------|-------------------|-------|------|
| `A` | `(n_hc,)` i.e. `1×4` | `[0, 1]` (σ) | **input map**: collapse the 4 channels into the one `D`-vector fed to the sub-block |
| `B` | `(n_hc, n_hc)` i.e. `4×4` | doubly-stochastic | **residual mix**: how the 4 channels recombine into themselves (the constrained one) |
| `C` | `(n_hc,)` i.e. `4×1` | `[0, 2]` (2σ) | **output map**: distribute the sub-block's single `D`-output back across the 4 channels |

The update rule the module implements is

```
X_next  =  B · X   +   C ⊗ F(A · X)
            └─────┘     └──────────┘
         residual mix   layer contribution
```

where `F` is the sub-block (attention or MoE). Read it as: *mix the existing
residual channels with B, then add the sub-block's output, scattered into the
channels by C.* This mirrors `X_{l+1} = B_l @ X_l + C_l ⊗ F_l(A_l · X_l)` from
the docstring (`src/osrt/mhc.py:6`).

### The three einsums

The contractions are spelled out explicitly so the index bookkeeping is correct
by construction (this was "Bug 2" in the old draft — `ARCHITECTURE.md:797-800`).

**Input view** — weight each channel by `A` and sum over channels `c`
(`src/osrt/mhc.py:92-95`):

```python
@staticmethod
def input_view(X, a):
    """Collapse the channel stream to one layer input: Σ_c a_c · X_c."""
    return torch.einsum("bsc,bscd->bsd", a, X)
```

`a` indexes channels `c`; `X` is `(b,s,c,d)`; the output drops `c`, giving the
single `(b,s,d)` vector the sub-block consumes.

**Residual update** — mix channels with `B`, then add the scattered output
(`src/osrt/mhc.py:97-102`):

```python
@staticmethod
def update(X, b_mat, c_out, f_out):
    """Residual update: B @ X + C ⊗ f_out."""
    mixed = torch.einsum("bsij,bsjd->bsid", b_mat, X)   # B mixes channels j→i
    contrib = c_out.unsqueeze(-1) * f_out.unsqueeze(-2)  # (B,S,n_hc,dim)
    return mixed + contrib
```

- `mixed`: `B` is `(b,s,i,j)`, `X` is `(b,s,j,d)`; summing over the *source*
  channel `j` produces the *destination* channel `i`. So `mixed[...,i,:]`
  `= Σ_j B[i,j] · X[j,:]` — exactly matrix-times-stream, one matmul per token.
- `contrib`: the outer product `C ⊗ f_out`. `c_out` is `(b,s,c)` →
  `unsqueeze(-1)` → `(b,s,c,1)`; `f_out` is the sub-block's `(b,s,d)` output →
  `unsqueeze(-2)` → `(b,s,1,d)`. Broadcasting multiplies them to
  `(b,s,c,d)`: channel `c` receives `C[c]` copies of the output. This is the
  "`C ⊗ F(...)`" term.

### How the block wires it together

`Block.forward` calls `generate → input_view → (run sub-block) → update`, twice
— once around attention, once around the MoE (`src/osrt/model.py:1153-1167`):

```python
if self.use_mhc:
    a, b_mat, c_out = self.mhc_attn.generate(x)
    x_in = self.mhc_attn.input_view(x, a)
    f_attn, present_kv = self._attention(x_in, ...)
    x = self.mhc_attn.update(x, b_mat, c_out, f_attn)

    a2, b2, c2 = self.mhc_ffn.generate(x)
    x_in2 = self.mhc_ffn.input_view(x, a2)
    f_moe = self._moe(x_in2, loop_idx, token_ids=token_ids)
    x = self.mhc_ffn.update(x, b2, c2, f_moe)
    return x, present_kv
```

Compare the standard path on the same method (`src/osrt/model.py:1169-1176`):
`x = x + f_attn; x = x + self._moe(x)`. The mHC path replaces each plain `+`
with a `generate/input_view/update` triple. There are **two** mHC instances per
physical block (`mhc_attn`, `mhc_ffn`, `src/osrt/model.py:969-974`), and each is
**shared across all 6 loop iterations** — the generators are owned per
sub-block, reused every loop (`src/osrt/mhc.py:14-16`). That sharing is what
keeps the parameter cost down (§9).

---

## 4. The Birkhoff / doubly-stochastic constraint on B

`A` and `C` are just bounded scalars per channel — easy. `B` is the hard one,
because it is the matrix that *multiplies the residual stream every layer*, so
its spectral norm is precisely the per-layer gain from §1. Left unconstrained it
would amplify or vanish.

The constraint: **B is doubly stochastic.** That means

1. every entry is non-negative: `B[i,j] ≥ 0`, and
2. every **row** sums to 1, and
3. every **column** sums to 1.

The set of all such `n×n` matrices is the **Birkhoff polytope** (`ARCHITECTURE.md`
§8.3). Permutation matrices are its corners; convex combinations of permutations
fill its interior. "Manifold-constrained" in the block's name refers to
constraining `B` to lie on this set.

### Why doubly-stochastic ⇒ non-expansive

The key fact:

> A doubly-stochastic matrix has spectral norm exactly **1**, hence
> `‖B‖₂ ≤ 1`, so `‖B·X‖ ≤ ‖X‖` — the map is **non-expansive**.

A short argument. `B` has non-negative entries with rows summing to 1, so it is
*row-stochastic*; the largest absolute row sum (the matrix ∞-norm) is 1, which
caps the largest eigenvalue magnitude at 1. Columns also sum to 1, so `Bᵀ` is
likewise row-stochastic and `‖B‖₁ = 1` too. The spectral norm satisfies
`‖B‖₂ ≤ sqrt(‖B‖₁ · ‖B‖∞) = sqrt(1·1) = 1`. (The constant vector `1` is a
fixed point — `B·1 = 1` — so the bound is tight at 1.) Non-negativity plus
both unit sums is therefore exactly what pins the gain to ≤ 1.

Two consequences make this the right tool for a recursive stack:

- **Per-layer non-expansion.** `B·X` never grows the residual norm, so the
  forward pass cannot blow up through the channel-mixing path. Backprop through a
  non-expansive linear map is equally well-behaved.
- **Closed under composition.** The product of two doubly-stochastic matrices is
  doubly-stochastic. So stacking 18 effective layers keeps the *composed*
  residual transform doubly-stochastic and still ‖·‖₂ ≤ 1 — the property does
  not erode with depth (`ARCHITECTURE.md:688-689`).

This is the 1,478× → 1.0× conversion promised in §1, now earned.
`ARCHITECTURE.md` §16.2 lists it as a hard invariant: *"`B_l` MUST satisfy
‖B_l‖₂ ≤ 1 at every step (doubly stochastic)"* (`ARCHITECTURE.md:1239`). The
test `tests/test_mhc.py::test_sinkhorn_is_doubly_stochastic_and_nonexpansive`
checks `matrix_norm(b, ord=2) ≤ 1.01` empirically.

But the raw generated `B` is an arbitrary `4×4` of real numbers. How do we
*project* it onto the Birkhoff polytope, differentiably, inside the forward
pass? That is Sinkhorn-Knopp.

---

## 5. Sinkhorn-Knopp in the log domain

### The algorithm

Sinkhorn-Knopp turns any non-negative matrix into a doubly-stochastic one by
**alternately normalizing rows and columns** until both sum to 1. The naive
recipe:

```
M ← exp(raw)                 # make entries positive
repeat ~20 times:
    M ← M / row_sums(M)      # rows now sum to 1
    M ← M / col_sums(M)      # cols now sum to 1 (rows drift slightly)
```

Each step fixes one constraint and slightly disturbs the other, but the two
constraints are compatible, so the alternation converges geometrically to a
matrix that satisfies both. ~20 iterations is plenty for a 4×4 near the model's
operating regime (`mhc_sinkhorn_iters = 20`, `src/osrt/config.py:61`).

### Why the log domain — the load-bearing detail

The naive `exp`-then-divide form is a **numerical trap** inside a trained
network, for two reasons:

1. **`exp` overflows.** The raw logits feeding `B` include the identity bias
   `4·I` (§6) plus a dynamic term. `exp(4) ≈ 54.6` is fine, but transient large
   logits during training (or a wide spread) push `exp` toward `inf`, and then
   every subsequent divide is `inf/inf = NaN`.
2. **Gradients explode through the iterations.** Backprop runs through all ~20
   row/col divides. Division is numerically hostile — its gradient is
   `∂(a/b)/∂b = -a/b²`, which blows up as denominators get small. Chaining 20 of
   these compounds the instability, and the model trains to `NaN`.

The fix is to do the *entire* iteration in **log space**, where multiplication
becomes addition and normalization becomes subtraction of a `logsumexp`. This is
the implementation (`src/osrt/mhc.py:25-39`):

```python
def sinkhorn_doubly_stochastic(logits: Tensor, iters: int) -> Tensor:
    """Project per-token n×n logit matrices onto the Birkhoff polytope.
    Runs in the LOG domain (alternating logsumexp normalization) — the naive
    exp-then-divide form produces exploding gradients through the 20 iterations
    and drives training to NaN. Log-domain Sinkhorn is the stable standard.
    """
    log_m = logits
    for _ in range(iters):
        log_m = log_m - torch.logsumexp(log_m, dim=-1, keepdim=True)  # rows
        log_m = log_m - torch.logsumexp(log_m, dim=-2, keepdim=True)  # cols
    return log_m.exp()
```

Why this is the same algorithm, just stabilized: normalizing a row to sum 1
means dividing by the row sum, and the log of that row sum is exactly
`logsumexp(log_m)` — so dividing becomes **subtracting**
(`log_m -= logsumexp(log_m, dim=-1)`, then `dim=-2` for columns). `logsumexp` is
itself numerically stable (it subtracts the max before exponentiating), so no
intermediate overflows. The only `exp` is the single final one (`log_m.exp()`),
applied to values already ≤ 0 (each entry was divided by a sum it belongs to), so
the result lands in `(0, 1]` — and the gradient path is a chain of stable
`subtract`/`logsumexp` ops instead of 20 divides.

> **Note vs. `ARCHITECTURE.md` §8.3.** The design sketch shows `M_0 = exp(~B_l)`
> followed by row/col normalization on `M` (`ARCHITECTURE.md:679-682`) — i.e. the
> *naive* form. The shipped code does **not** do that; it treats the raw logits
> directly as the log-domain matrix and never forms `exp(~B)` at all (the
> docstring even says the naive form "drives training to NaN"). Trust the code:
> the log-domain version in `mhc.py` is the real one, and the §8.3 pseudocode is
> an illustrative sketch (`ARCHITECTURE.md:788` flags the whole §10 walkthrough
> as illustrative for the same reason).

A practical caveat worth knowing: 20 iterations gives *approximate* double
stochasticity. The stress test (wide logits) converges only to ~1.5e-2 row/col
error, while the model's actual near-identity regime converges to <1e-3
(`tests/test_mhc.py:27-39`). The spectral-norm bound stays ≈ 1 either way, which
is what matters for stability.

---

## 6. Identity init — why start as a plain residual

A brand-new mHC block should *not* immediately impose some random channel mix —
that would throw away the well-understood behaviour of a standard residual
before the model has learned anything useful. So the static biases are
initialized to make mHC behave, at step 0, almost exactly like a plain residual.

The bias for `B` is a **scaled identity** (`src/osrt/mhc.py:68-70`):

```python
self.s_pre  = nn.Parameter(torch.zeros(n_hc))
self.s_res  = nn.Parameter(torch.eye(n_hc).reshape(-1) * 4.0)   # ← scaled identity
self.s_post = nn.Parameter(torch.zeros(n_hc))
```

And the dynamic component is gated by a *small* learnable scalar
`alpha = 0.1` (`src/osrt/mhc.py:73-75`, `alpha_init=0.1`). At init, `B`'s raw
logits are therefore dominated by `s_res = 4·I`:

```python
b_raw = (self.alpha_res * self.w_res(flat) + self.s_res).reshape(b, s, c, c)
b_mat = sinkhorn_doubly_stochastic(b_raw, self.sinkhorn_iters)
```

Why `4.0` and not `1.0`? Because Sinkhorn operates on logits. Feeding a *sharp*
diagonal (large on the diagonal, ~0 off it) into Sinkhorn projects to a matrix
very close to the **identity permutation** — channel `i` maps to channel `i`,
the residual passes straight through. A diagonal of `1.0` would be too soft:
after row/col normalization it would smear toward the uniform `1/n_hc` matrix
(every channel averaged together), which is *not* a pass-through. The `×4`
sharpens the logit gap so the projected `B ≈ I` at init (`src/osrt/mhc.py:66-69`:
"so that at step 0 ... B ≈ identity — the stream starts as a near-standard
residual, then learns to mix").

For `A` and `C` the static biases are zero (`s_pre = s_post = 0`); the dynamic
terms are tiny (`alpha = 0.1` times a small random `W·x`), so at init
`A ≈ σ(0) = 0.5` per channel and `C ≈ 2σ(0) = 1.0` per channel — every channel
contributes roughly equally to the input view and receives the output roughly
equally. Combined with `B ≈ I`, the step-0 behaviour is a symmetric,
near-standard residual across identical channels; the small `alpha` gates then
let the dynamic generators learn to *differentiate* the channels and mix them as
training proceeds. The design principle — start as close to a proven standard
residual as possible, then learn the interesting structure under gradient —
recurs in the collapse-head init (§7).

---

## 7. The final collapse head

The residual stream lives as `(B, S, n_hc, D)` all the way through the loop, but
the LM head (and every per-loop auxiliary head) needs a single `(B, S, D)`
vector. mHC must therefore **collapse** the 4 channels back to one.

The naive choice would be to reuse a dynamic `A` matrix for this. The code
deliberately does **not**. Instead it owns a dedicated learnable parameter
`mhc_collapse` of length `n_hc`, initialized to the uniform average `1/n_hc`
(`src/osrt/model.py:1273-1277`):

```python
self.use_mhc = config.use_mhc
if config.use_mhc:
    self.mhc_collapse = nn.Parameter(
        torch.full((config.n_hc,), 1.0 / config.n_hc)   # uniform channel average
    )
```

The collapse is a single einsum (`src/osrt/model.py:1288-1291`):

```python
def _collapse(self, X):
    """mix the n_hc residual channels into one d_model vector via the
    dedicated learnable collapse head. X: (B, S, n_hc, D) -> (B, S, D)."""
    return torch.einsum("c,bscd->bsd", self.mhc_collapse, X)
```

It is applied at the very end of the loop stack before the final norm
(`src/osrt/model.py:1533-1535`), and *also* for every intermediate per-loop aux
head (`src/osrt/model.py:1524-1527`).

### Why a dedicated head, not a reused A

This was "Bug 3" in the early design (`ARCHITECTURE.md:802-805`). The dynamic
`A` matrices are generated *inside* each sub-block to pick that sub-block's
input view, conditioned on the residual at that moment. Reusing the *last*
sub-block's `A` to collapse for the LM head would be using a stale, purpose-built
input selector as if it were an output reader — the two jobs are different, and
the `A` you happen to have lying around at the end was computed to feed the MoE,
not to summarize the whole stream for prediction. A dedicated `mhc_collapse`:

- is a clean, single-purpose readout, learned end-to-end for *its* job;
- is shared identically across the main LM head and all aux per-loop heads, so
  every readout sees the channels the same way (consistency across §9.2 aux
  losses);
- costs a trivial `n_hc = 4` parameters.

Uniform `1/n_hc` init means the first readout is a plain channel average (the
same start-from-standard principle as §6), and the model then learns which
channels carry the prediction-relevant signal.

---

## 8. Dynamic generation of A / B / C per token

The three matrices are not fixed weights — they are **functions of the current
residual stream**, computed fresh for every token at every sub-block. This is
the "hyper" in hyper-connection: a small hypernetwork emits the connection
weights.

The generators are three bias-free linear maps plus an RMSNorm
(`src/osrt/mhc.py:56-63`):

```python
flat = n_hc * dim                       # 4 × 1536 = 6144
self.norm   = nn.RMSNorm(flat)
self.w_pre  = nn.Linear(flat, n_hc,        bias=False)   # → A : (..,4)
self.w_res  = nn.Linear(flat, n_hc * n_hc, bias=False)   # → B : (..,16) → 4×4
self.w_post = nn.Linear(flat, n_hc,        bias=False)   # → C : (..,4)
```

And `generate` ties it together (`src/osrt/mhc.py:77-90`):

```python
def generate(self, X):
    b, s, c, d = X.shape
    flat = self.norm(X.reshape(b, s, c * d))             # flatten 4 channels, RMSNorm
    a     = torch.sigmoid(self.alpha_pre  * self.w_pre(flat)  + self.s_pre)
    c_out = 2.0 * torch.sigmoid(self.alpha_post * self.w_post(flat) + self.s_post)
    b_raw = (self.alpha_res * self.w_res(flat) + self.s_res).reshape(b, s, c, c)
    b_mat = sinkhorn_doubly_stochastic(b_raw, self.sinkhorn_iters)
    return a, b_mat, c_out
```

Reading it:

- **Flatten + normalize.** The 4 channels are flattened to one `6144`-vector and
  RMSNorm'd. The generators are conditioned on *all* channels jointly, so the
  mix can depend on the relationship between channels, not just one.
- **`A = σ(α·W_pre·x + s_pre)`** — bounded to `[0, 1]` per channel (see below).
- **`C = 2σ(α·W_post·x + s_post)`** — bounded to `[0, 2]` per channel.
- **`B = Sinkhorn(α·W_res·x + s_res)`** — the raw `16` outputs are reshaped to
  `4×4` and projected to the Birkhoff polytope.

Each piece = `static bias` + `alpha · dynamic`. The static bias supplies the
identity-init behaviour (§6); `alpha` (small at init) gates how much the
per-token dynamic term is allowed to perturb it. The `W_*` are 2D matrices, so
they are optimized by Muon (`ARCHITECTURE.md:1273-1274`).

### The sigmoid bounds on A and C — why

`A` and `C` are *not* Sinkhorn-constrained; they are bounded by sigmoid:

- `A ∈ [0,1]`: the input view `Σ_c A_c · X_c` is a non-negative blend of channels
  — no channel can be subtracted, so the input the sub-block sees can't be a
  cancellation artifact.
- `C ∈ [0,2]`: the output is scattered into channels with non-negative weights,
  but the factor 2 (rather than 1) leaves headroom to *amplify* a sub-block's
  contribution into a channel when useful (`ARCHITECTURE.md:713-721`: "Prevents
  signal cancellation. The factor 2 on C_l preserves the ability to scale layer
  contributions").

Note `A` and `C` are *cheap* gains on small tensors — the stability-critical
matrix is only `B`, because only `B` multiplies the persistent residual every
layer. That is why only `B` gets the expensive doubly-stochastic projection.

---

## 9. Cost

### Parameters — exactly 921,766

From `scripts/compute_budget.py` (run it: `python scripts/compute_budget.py`):

```
mhc   921,766
```

This is **~0.15% of the 601M-param model** and breaks down exactly as the math
predicts. Each `ManifoldHyperConnection` instance holds:

| component | count | params |
|-----------|-------|--------|
| `norm` (RMSNorm gain over `flat`) | `6144` | 6,144 |
| `w_pre` (`6144 × 4`) | | 24,576 |
| `w_res` (`6144 × 16`) | | 98,304 |
| `w_post` (`6144 × 4`) | | 24,576 |
| `s_pre`, `s_res`, `s_post` | `4 + 16 + 4` | 24 |
| `alpha_pre/res/post` | `3` | 3 |
| **per instance** | | **153,627** |

There are 2 instances per physical block (attn + ffn) × 3 blocks = **6
instances** (shared across the 6 loops, *not* re-instantiated per loop):

```
6 × 153,627  =  921,762
+ mhc_collapse (length n_hc)          =        4
─────────────────────────────────────────────────
total                                 =  921,766   ✓ matches compute_budget.py
```

The crucial line is **shared across loop iterations** (`src/osrt/mhc.py:14-16`).
If mHC re-created generators per loop it would be 6× the params; instead one
attn-generator and one ffn-generator per block are reused all 6 loops, which is
why the count is ~922K and not ~5.5M. `ARCHITECTURE.md:723` rounds this to
"~720K, ~6.7% overhead" — the precise figure from `compute_budget.py` is
**921,766**; trust the script.

### Runtime — the real cost is Sinkhorn, not parameters

The parameter overhead is negligible; the *compute* overhead is the Sinkhorn
projections. Per forward pass:

```
18 effective layers × 2 sub-blocks = 36 mHC applications   (ARCHITECTURE.md:727)
each runs Sinkhorn with 20 iterations
each iteration = 2 logsumexp + 2 subtracts over (B, S, 4, 4)
→ 36 × 20 = 720 row/col normalization rounds per forward
```

Each individual op is tiny (a `4×4` per token), but there are a lot of them, and
`logsumexp` on small tensors is bandwidth/launch-bound rather than FLOP-bound.
Unfused, that is ~720 small kernel launches per forward — death by a thousand
cuts on GPU. **This block wants `torch.compile` or a fused Sinkhorn kernel** to
collapse the per-iteration ops into one graph; eager-mode small-tensor Sinkhorn
is the dominant mHC runtime cost. `ARCHITECTURE.md:730` budgets ~6.7%
wall-clock overhead, which assumes the projections are reasonably fused.

### The NaN watch-item

Despite the log-domain Sinkhorn (which exists *specifically* to prevent
divergence, §5), mHC is **flagged NaN-prone under sustained CPU training** and
has not yet cleared a GPU-phase stability gate (§10). The math guarantees
‖B‖₂ ≤ 1, but the gate is about the *whole training trajectory* — the interplay
of the `alpha` gates, the aux-loss gradients flowing through the collapse head,
and bf16 precision across 18 reused layers — not just the per-step spectral
bound. This is the open item, not a settled property.

---

## 10. Status

- **Enabled in the canonical preset.** `use_mhc=True` in `presets.py:43`
  (alongside `n_hc=4`, `mhc_sinkhorn_iters=20`) and in the shipped config
  `configs/osrt-605m-a279m/config.json:78`. Note the *dataclass default* in
  `config.py:59` is `use_mhc=False` — the preset turns it on, so a config built
  bare (without the preset) runs the standard single-stream residual. The block
  is fully wired and tested (`tests/test_mhc.py`: Sinkhorn double-stochasticity
  + non-expansion, shape correctness, finite forward).
- **Pending a GPU-phase stability gate.** `ARCHITECTURE.md:138` and `:1483`
  flag mHC as "NaN-prone on [sustained] training — needs GPU profiling." The
  feature is on by design intent but carries a stability caveat: the proof gives
  per-step boundedness; the empirical question is whether full training stays
  finite end-to-end on GPU. Until that gate is cleared, mHC is the model's most
  promising *and* most watched architectural feature.

### One-paragraph recap

mHC widens the residual stream to 4 channels and replaces each plain `x + F(x)`
with a learned per-token mix `X ← B·X + C ⊗ F(A·X)`. `A` (σ-bounded) picks the
sub-block's input view from the channels; `C` (2σ-bounded) scatters its output
back; `B` recombines the channels and is the one that matters for stability.
Constraining `B` to be doubly-stochastic (via log-domain Sinkhorn-Knopp, ~20
iters) pins its spectral norm at ≤ 1, so the residual transform is non-expansive
and composes safely across 18 effective layers — turning a potential
multi-thousand-× amplifier into a 1.0× map. Identity init (`s_res = 4·I`) makes
it behave like a standard residual at step 0; a dedicated `mhc_collapse` head
reads the channels back down to `D` for the LM head. Cost: 921,766 params
(`compute_budget.py`), 36 Sinkhorn projections per forward (wants
`torch.compile`), on in the preset, pending a GPU stability gate.
