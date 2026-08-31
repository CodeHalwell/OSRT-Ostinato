# MoE & Routing

> **v7 status.** The architecture this chapter describes is current, but its
> **`file:line` citations, parameter tables and config values were written
> against v6** and have not been regenerated. mHC references have been removed
> (roadmap §12.3); expert counts, vocab and param figures may still be stale.
> Regenerate counts with `scripts/compute_budget.py`; `src/osrt/` is ground
> truth where they disagree.


*Part of the OSRT-605M `docs/` architecture series. Companion to `ARCHITECTURE.md §7`.*

This document explains the **mixture-of-experts (MoE)** sub-block of the
OSRT-605M model — the part that replaces the dense feed-forward network of a
vanilla transformer with a small set of *experts* and a *router* that sends
each token to only a few of them. Everything here is grounded in
`src/osrt/model.py` (the `MoELayer` class, roughly lines 163–860) and the
canonical preset `OSRT_605M_A288M` in `src/osrt/presets.py`.

> **A note on ground truth.** Three sources describe this block and they do
> *not* fully agree. The **code** wins. `ARCHITECTURE.md §7` contains stale
> draft numbers (12 experts, `mod 12`, a duplicated §7.6), and the
> `OSRTConfig` *constructor defaults* in `config.py` are deliberately neutral
> (and actually invert the routed/shared widths). The real model is built
> from the preset. Where the code contradicts the design docs, this document
> follows the code and flags the divergence inline.

---

## 1. Purpose: a sparse FFN

In a dense transformer, every token flows through one big FFN — every
parameter does work on every token. MoE breaks that FFN into many smaller
**experts** and adds a **router** that, per token, activates only `top_k` of
them. The parameter *count* grows (you store all experts), but the per-token
*compute* stays small because each token touches only a few.

OSRT takes the **DeepSeekMoE hybrid** shape: one *shared* expert that every
token always runs, plus a pool of *routed* experts of which only the top-2
fire per token.

```python
# model.py:199-210
self.shared_expert = ExpertFFN(config.dim, config.shared_expert_hidden, clamp=clamp)
self.experts = nn.ModuleList([
    ExpertFFN(config.dim, config.expert_hidden, clamp=clamp)
    for _ in range(self.num_routed)
])
```

The layer returns **two tensors** — the shared output and the routed output —
and lets the enclosing `Block` combine them (see §2 and §11). It does *not*
fold them together itself:

```python
# model.py:858-860
# Return (shared, routed) so the Block can apply moe_gate only to
# the routed contribution. Shared expert stays at full weight.
return shared_out, moe_out
```

For the OSRT-605M preset (`presets.py:22-64`) the numbers are:

| Component       | Count | Hidden (`h`) | Active per token        |
|-----------------|-------|--------------|-------------------------|
| Shared expert   | 1     | 2,816        | always                  |
| Routed experts  | 8     | 3,840        | top-2 (25% of the pool) |

---

## 2. Shared vs routed experts — and why a shared one at all

Why keep an always-on expert when the whole point of MoE is sparsity?

A pure top-k router has to learn *both* the per-expert knowledge *and* the
routing decision from scratch. Early in training the router is random, so
experts receive a noisy, intermittent gradient. Worse, the model has no
guaranteed capacity to learn the "common" computation that *every* token
needs — basic syntax, copying, the boring-but-essential stuff. DeepSeekMoE's
insight: carve that common computation out into a **shared expert** that runs
unconditionally. The routed experts are then free to specialise on what's
left, and the model always has a dense backbone to fall back on.

In OSRT the shared expert is *narrower* than each routed expert
(`h=2,816` vs `h=3,840`). That is a budget decision — the 8 wide routed
experts are where the parameters (and the specialisation capacity) live; the
shared expert is a lean always-on floor.

> **Width inversion in `config.py`.** The constructor *defaults* are
> `expert_hidden=2048` and `shared_expert_hidden=4096`
> (`config.py:80-81`) — i.e. the shared expert is *wider*. That is **not**
> the trained model. The preset overrides both
> (`expert_hidden=3840`, `shared_expert_hidden=2816`,
> `presets.py:33-34`), making the routed experts wider, as intended.

The shared/routed split also maps onto a two-level gating scheme (§11): the
routed branch is scaled by a per-block learned scalar `moe_gate`, the shared
branch is always at full weight. That asymmetry is a deliberate guard against
a real OSRT failure mode (§12).

---

## 3. The expert FFN: SwiGLU with a stability clamp

Both kinds of expert are the same module, `ExpertFFN` — a standard SwiGLU
feed-forward block with an optional clamp:

```python
# model.py:96-117
class ExpertFFN(nn.Module):
    """SwiGLU feed-forward. Used for both shared and routed experts."""
    def __init__(self, dim, hidden, clamp=None):
        hidden = 64 * ((hidden + 63) // 64)  # TC-align
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up   = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        self.clamp = clamp

    def forward(self, x):
        gate = self.w_gate(x)
        up = self.w_up(x)
        if self.clamp is not None:
            gate = gate.clamp(max=self.clamp)
            up = up.clamp(min=-self.clamp, max=self.clamp)
        return self.w_down(F.silu(gate) * up)
```

A few things worth teaching here:

- **SwiGLU** = `w_down( SiLU(w_gate·x) ⊙ (w_up·x) )`. The `w_up` path is a
  data-dependent gate on the `SiLU(w_gate·x)` path — it can multiplicatively
  amplify activations, which is exactly why it needs a clamp.
- **The clamp (`swiglu_clamp=10.0`, `presets.py:46`)** bounds the
  pre-activations so one extreme value can't blow up the product. The gate is
  clamped on the **top** only (`max=clamp`) — `SiLU` already saturates
  smoothly toward zero on the negative side, so a huge *negative* gate is
  harmless. The `up` path is clamped on **both** sides because it enters the
  product *linearly* and a large magnitude of either sign is dangerous. This
  is a DeepSeek-V4-style stability measure (`ARCHITECTURE.md §7.8`); for a
  healthy model it is a no-op that only caps the tails. **Since 2026-08-31
  `OSRT_V7` sets `situ_glu=True`** (roadmap §14.1), whose param-free smooth
  cap *replaces* SwiGLU + this hard clamp; the clamp value stays in the
  preset as the fallback for the G3 ladder's SiTU-vs-clamp A/B.
- **`hidden` is rounded up to a multiple of 64** (`model.py:101`) for
  tensor-core alignment. Both preset widths (3,840 and 2,816) are already
  multiples of 64, so nothing changes for OSRT-605M.

Shapes per routed expert (preset): `w_gate, w_up ∈ ℝ^(1536×3840)`,
`w_down ∈ ℝ^(3840×1536)`.

---

## 4. The router: `sqrt(softplus)` affinity, top-2, gate renormalisation

The router is a single bias-free linear layer that turns a hidden state into
one score per routed expert. Before projecting, OSRT adds a **per-loop
embedding** so the same physical experts can be routed differently at each
recursive depth (OSRT runs each block multiple times — see the loops doc):

```python
# model.py:515-517
loop_emb = self.loop_embeddings.weight[loop_idx].view(1, 1, D)
router_input = x + loop_emb
router_logits = self.router(router_input.reshape(N, D))  # (N, E)
```

OSRT supports two affinity transforms (`router_affinity`, `model.py:535`).
The preset uses **`sqrt_softplus`** (`presets.py:62`), the DeepSeek-V4 style:

```python
# model.py:536-542
if affinity_mode == "sqrt_softplus":
    affinity = torch.sqrt(F.softplus(router_logits))  # (N, E), always ≥ 0
    raw_router_probs = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-9)
```

Why `sqrt(softplus(·))` instead of a plain `softmax`?

- **`softplus`** maps any logit to a strictly positive, smooth value. Unlike
  `softmax`, it does *not* force the scores to compete in a single normalised
  simplex — each expert's affinity is independent, so a token can like several
  experts strongly at once (or none).
- **`sqrt`** compresses the tail, keeping the dynamic range tame so a single
  runaway logit doesn't dominate.
- Because the affinity is **non-negative**, an additive load-balancing bias
  (§5) and Gumbel exploration noise (§6) can be added directly to it and the
  result is still a valid, non-negative routing score.

> The other mode, `"softmax"` (`model.py:572-597`), is the historical
> Mixtral/v5 path, kept bit-identical for A/B comparison. It adds the bias to
> the *logits* pre-softmax. The preset does not use it.

**Top-2 selection and gate renormalisation.** The selection scores
(`probs`, the bias+Gumbel affinity normalised to sum to 1) are sorted and the
top-2 are taken. The chosen gates are then **renormalised to sum to 1**:

```python
# model.py:599-616
raw_top_probs, top_idx = probs.topk(self.top_k, dim=-1)  # (N, K)
...
top_probs = raw_top_probs / raw_top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
```

This renormalisation matters. Without it, the MoE branch magnitude would
depend on how confident the router happened to be: a token whose top-2
affinities summed to 0.4 would get a weaker FFN contribution than one summing
to 0.9, for no good reason. Forcing the `k` chosen gates to sum to 1 keeps the
routed branch at a consistent scale regardless of `k` or router sharpness
(`model.py:610-613`).

---

## 5. Load balancing: a non-learned bias + Switch loss + z-loss

Left alone, a top-k router collapses: a few lucky experts win early, get more
gradient, win more, and the rest go cold. OSRT fights this on **three**
fronts.

### 5.1 The aux-loss-free balance bias

A persistent additive **bias buffer**, shaped `[num_loops, num_routed]`,
steers *which* experts get selected without touching the router's gradient:

```python
# model.py:236-240
self.register_buffer(
    "router_balance_bias",
    torch.zeros(self.num_loops, self.num_routed, dtype=torch.float32),
    persistent=True,
)
```

It is a **buffer, not a parameter** — no gradient ever flows to it. It is
applied to the *affinity* used for selection:

```python
# model.py:543-551
if self.bias_enabled:
    loop_bias = self.router_balance_bias[loop_idx].view(1, -1)
    clean_affinity = affinity + loop_bias
...
clean_affinity = clean_affinity.clamp_min(0.0)
```

It is updated once per optimiser step by the trainer calling
`apply_balance_update()`, which nudges each expert's bias *down* if it was
over-used and *up* if under-used, toward a uniform target `1/E`:

```python
# model.py:378-382
target = 1.0 / self.num_routed
delta[active] = current_frac[active] - target
self.router_balance_bias.add_(delta, alpha=-self.bias_update_rate)
self.router_balance_bias.clamp_(-self.bias_max, self.bias_max)
```

This is DeepSeek-V3's "aux-loss-free" load balancing: the router weights learn
*purely* from the task gradient, while a cheap non-gradient controller handles
balance. No fragile trade-off between an auxiliary balance loss and the LM
loss.

> **Two divergences from `ARCHITECTURE.md §7.6` and the project notes — code
> wins:**
>
> 1. **The update is proportional, not a fixed `±γ`.** The design notes
>    describe DeepSeek-V3's recipe literally (`b -= γ` / `b += γ`,
>    `γ=0.001`). The code instead moves by
>    `−update_rate · (current_frac − 1/E)`, with
>    `router_balance_bias_update_rate = 0.10` (`presets.py` inherits the
>    `config.py:190` default) and a clamp at `±1.5`
>    (`router_balance_bias_max`, `config.py:192`). So the step is
>    **load-deviation-proportional**, much larger than 0.001, and self-damping
>    as balance improves.
> 2. **The delta uses the *instantaneous* fraction, not the EMA.** There is an
>    `expert_ema_fraction` buffer tracked at rate 0.05 (`model.py:367-376`),
>    but the bias `delta` is computed from `current_frac` (this step's load),
>    not the EMA (`model.py:380`). The EMA is bookkeeping/telemetry, not the
>    control signal — do not describe this as "EMA-driven."

**Why per-loop?** Capacity (§7) is enforced *per MoE call*, and OSRT calls the
same block once per recursive loop. A single block-level bias could look
balanced in aggregate while an individual loop's call overflows. A separate
bias row per loop corrects each loop's load independently (`model.py:226-231`).

### 5.2 Where the bias does *and does not* act — the real DeepSeek trick

This is the subtle, important part, and it is where the code and the design
notes most clearly diverge. The notes (and `ARCHITECTURE.md §7.4`) say *"bias
only in TOP-K selection, not in gating weights."* **The code does not do
that.** The gating weights are renormalised from `raw_top_probs`, which come
from `probs` — the bias-and-Gumbel affinity (`model.py:600, 614`). So the bias
*does* enter the gate weights.

What is actually kept bias-free is a separate tensor, `raw_router_probs`
(`model.py:540`), used **only** for the auxiliary losses and pre-bias
telemetry. The code says so directly:

```python
# model.py:631-635
# Switch balance loss ... Compute it on the RAW router logits, not the noisy
# dispatch path or the bias-corrected clean path. ... Dispatch below still
# uses bias+Gumbel top_idx/probs.
```

So the precise statement of the "DeepSeek trick" *as implemented* is:

> The balance bias is a **non-learned buffer** (no gradient flows to it), and
> the **auxiliary losses are computed on the bias-free distribution** — *not*
> "the bias is excluded from the gate weights."

The reason this matters: the bias is an *external controller*. If its effect
leaked into the gradient of the balance/z-loss, those losses would start
penalising the router for imbalance that the controller is *already*
correcting — double-counting, and a corrupted learning signal. By computing
the aux gradient on the raw, un-biased, un-noised distribution, the loss keeps
pushing the *learned router itself* away from collapse, independently of the
controller.

### 5.3 The Switch balance loss

A small gradient-based balance loss still runs, as a backstop, on the raw
distribution:

```python
# model.py:640-653
raw_balance_f = raw_balance_one_hot.float().sum(dim=(0,1)) / (N * self.top_k)
raw_balance_p = raw_router_probs.float().mean(dim=0)
self.balance_loss = self.num_routed * (raw_balance_f * raw_balance_p).sum()
```

This is the Switch-Transformer loss `E · Σ_i f_i · p_i`, where `f_i` is the
*hard* fraction of token-expert pairs routed to expert `i` and `p_i` is the
*soft* mean probability. It is minimised (`= 1.0`) at a uniform split. Note
it is computed in **fp32** — under bf16 autocast, `f·p` can underflow late in
training when both terms approach `1/E = 0.125`, killing the gradient
(`model.py:643-646`). The trainer scales it by `router_aux_loss_coeff=0.10`
(`presets.py:55`).

### 5.4 The router z-loss

```python
# model.py:662-663
z = torch.logsumexp(router_logits.float(), dim=-1)  # (N,)
self.z_loss = (z ** 2).mean()
```

The ST-MoE z-loss penalises the *magnitude* of the router logits
(`mean_token (logsumexp logits)²`). It keeps softmax exponentials from
overflowing in bf16/fp8 and keeps early distributions flatter so cold experts
retain gradient through LR warm-up. Coefficient `router_z_loss_coeff=1e-3`
(`presets.py:56`). Like the balance loss, it is computed on the **raw**
logits so it disciplines the learned router, not the controller or the noise.

### 5.5 Sequence-wise balance loss (off by default)

There is also a per-sequence Switch loss (`model.py:665-683`) that penalises
imbalance *inside* a single sequence — useful at long context where one
document can dominate a micro-batch. `router_seq_balance_loss_coeff`
defaults to `0.0` in `OSRTConfig`, but since 2026-08-31 `OSRT_V7` sets it
to `1e-4` per the roadmap's §14.1 committed router line.

---

## 6. Gumbel exploration noise

Even with balancing, an expert that loses the *first few* router updates can
go permanently cold before it ever gets a fair chance. OSRT injects
**Gumbel noise** into the selection scores during training to keep
exploration alive:

```python
# model.py:559-568
selection_affinity = clean_affinity
if self.training:
    u = torch.rand_like(clean_affinity).clamp_(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(u))
    tau = self.gumbel_tau.to(dtype=clean_affinity.dtype)
    selection_affinity = (clean_affinity + tau * gumbel).clamp_min(0.0)
```

Adding Gumbel noise to a score and taking the top-k is the standard way to
*soft-sample* the routing decision rather than always taking the deterministic
argmax — occasionally a slightly-less-preferred expert wins, gets gradient,
and stays warm. In `sqrt_softplus` mode the noise is added to the **affinity**
(not the logits), then clamped non-negative, so it interacts correctly with
the non-negative affinity transform (`model.py:556-558`).

The temperature `tau` is a buffer initialised from
`router_gumbel_tau_init`, which defaults to **0.0** (`config.py:211`) and is
*not* overridden by the preset. So out of the box there is **no** noise; the
**trainer** schedules `gumbel_tau` up early and **anneals it back to ~0**
before the health-gate evaluation, so the final pass is judged on the clean,
deterministic router (`model.py:585-588`). Two consequences:

- Eval/inference (`self.training == False`) never adds Gumbel noise — routing
  is deterministic.
- The balance and z-losses are computed on the *raw* (pre-Gumbel)
  distribution, so exploration noise never pollutes the aux gradient.

---

## 7. Dispatch: grouped-GEMM (default) vs the per-expert loop (fallback)

Once the top-2 experts and their renormalised gates are chosen, the tokens
have to actually be *run through* the chosen experts. OSRT ships **two
numerically-equivalent dispatch implementations**, selected by the
`moe_grouped_gemm` flag (`config.py:88`). The canonical preset turns the
grouped-GEMM path **on** (`moe_grouped_gemm=True`, `presets.py:60`):

```python
# model.py:849-856
if self.grouped_gemm:
    moe_out, total_dropped = self._dispatch_grouped(x_flat, top_idx, top_probs)
else:
    moe_out, total_dropped = self._dispatch_loop(
        x_flat, top_idx, top_probs, capacity)
```

### 7.1 The grouped-GEMM dispatch (B4, the preset default)

`_dispatch_grouped` (`model.py:594-627`) runs **all** experts in a single
grouped matmul instead of looping expert-by-expert. The recipe:

1. **Flatten** the `(token, rank)` pairs into three parallel `(N·K,)` vectors:
   the chosen expert id, the gate, and the source token index
   (`model.py:609-613`).
2. **Argsort by expert** (stable, so it's deterministic) so every expert's
   tokens form one contiguous span (`model.py:616`).
3. **`bincount` → `cumsum`** to get per-expert cumulative *end* offsets
   (`model.py:619-620`).
4. **One grouped SwiGLU** over the sorted tokens via `_grouped_ffn`
   (`model.py:550-592`): the per-expert `w_gate`/`w_up`/`w_down` are stacked
   into `(E, D, H)`/`(E, H, D)` and run as three grouped matmuls. On CUDA this
   is the fused `torch._grouped_mm` kernel; on CPU (tests, and as the parity
   oracle) it falls back to a loop-of-matmuls reference, `_ref_grouped_mm`
   (`model.py:164-185`), because the kernel's CPU *backward* is broken in
   torch 2.10.
5. **Apply the gate** (in fp32, like the loop) and **`index_add` scatter**
   back to per-token outputs (`model.py:622-626`).

The grouped path is **dropless by construction** — it has *no* capacity cap
and keeps every token (`model.py:597-604`). This is the key behavioural
difference from the loop path (§7.3): there is no `capacity` argument, nothing
is ever dropped, and `total_dropped` is always `0`.

The motivation is `torch.compile`. The old per-expert loop's data-dependent
`.nonzero()` was the **only `torch.compile` graph break in the whole model**;
replacing it with the fixed-shape grouped ops (argsort, bincount, cumsum,
index_add) lets the model compile as a single **fullgraph — 0 breaks vs 12**
for the loop path, worth **~9–12% steady-state** throughput (`presets.py:56-59`,
verified on H100). The grouped kernel needs two extra Dynamo flags —
`capture_scalar_outputs` and `capture_dynamic_output_shape_ops` — which the
trainer sets only when *both* compile and grouped are on (`train.py:757-761`).

**Weights are identical to the loop path.** The grouped dispatch is a pure
rearrangement of the same per-expert SwiGLU math, so a checkpoint trained with
one dispatch loads and runs bit-compatibly under the other
(`presets.py:58-59`).

### 7.2 The per-expert loop dispatch (retained fallback)

`_dispatch_loop` (`model.py:508-548`) is the original implementation, kept as a
fallback (`moe_grouped_gemm=False`). For each expert it `.nonzero()`-gathers
every token that picked it at *any* top-k rank, runs that expert, and
scatter-adds the gated output — so a token that chose the same expert at two
ranks contributes twice via `index_add_`. Unlike the grouped path, **this one
enforces a capacity cap and drops tokens** (§7.3). Its `.nonzero()` is the lone
graph break that motivated B4.

### 7.3 Capacity and token dropping (loop path only) vs drop-free eval

The **capacity cap lives only on the loop path.** In training it limits each
expert; at eval it is disabled entirely:

```python
# model.py:782-788 (capacity computed before dispatch)
if self.training:
    capacity = max(1, int(math.ceil(
        self.capacity_factor * self.top_k * N / self.num_routed)))
else:
    capacity = N * self.top_k  # effectively unlimited
```

With `router_capacity_factor=2.0` (`config.py:267`), each expert may take up
to `2×` its uniform share. Tokens beyond that are **dropped** for that
expert — they simply skip its branch this batch. Dropping is not just an
efficiency measure: it creates *balancing pressure*. An overloaded expert
loses tokens (and their gradient), which together with the bias controller
pushes the router toward a flatter distribution.

Two careful details (both specific to the loop path):

- **Drops are shuffled before truncation.** `nonzero()` returns indices in
  token-major order, so a naïve `[:capacity]` would always drop the *tail* of
  every sequence and keep prefix positions — training the model to ignore
  late tokens under overload. OSRT permutes the (token, rank) pairs first so
  every position has equal survival probability (`model.py:526-537`).
- **Eval is drop-free by construction.** Setting `capacity = N·top_k` means
  nothing is ever dropped at inference. This is deliberate: a documented v4
  failure mode was *chunk-instability* — prefill-then-decode producing
  different routing than a single full forward because capacity drops differed
  across chunk boundaries. Drop-free eval makes generation chunk-stable.

The **grouped path sidesteps this entirely** by never dropping at all (§7.1) —
it is dropless in both train and eval — so when it is enabled (the preset
default) capacity dropping never happens. The loop path's drops are the only
source of `drop_rate` telemetry.

---

## 8. Hash routing (off by default)

OSRT can replace the learned router in early physical blocks with a
**deterministic loop-indexed top-1 hash**:

```python
# model.py:413
assign = (token_ids.reshape(N) + loop_idx) % E  # (N,), one expert per token
```

A block hash-routes iff `block_idx < config.hash_routing_blocks`
(`model.py:197`), a hard switch decided at construction. The default is
`hash_routing_blocks = 0` — **off** in the canonical preset
(`config.py:231`, `presets.py:61`). It is a stability A/B knob, not part of
the trained model.

The motivation: a learned router is unreliable before it has warmed up. A
hash gives perfectly deterministic, balanced-in-expectation assignment, so the
*experts* can start learning useful features before the router exists — a
stable scaffold you can later remove. Top-1 (not top-2) keeps it simple and
maximally balanced; **loop-indexed** (`+ loop_idx`) means the same token maps
to *different* experts at different recursive depths, encouraging depth
specialisation rather than every loop hammering the same expert.

Because the assignment is deterministic, there is nothing to balance: the
hash path sets `balance_loss`, `z_loss`, and `seq_balance_loss` to **zero
tensors** (kept non-`None` so the wrapper's accumulation stays well-defined)
and populates telemetry from the hard histogram so the collapse monitor never
reads stale learned-router values (`model.py:426-466`).

> `ARCHITECTURE.md §7.5` still says `hash(token_id) mod 12` on "blocks 0 and
> 1." That is stale draft prose — the authoritative spec is the
> "DECISION MADE" box in §7.5 and the `_hash_route` code: `(token_id +
> loop_idx) % num_routed_experts`, off by default.

---

## 9. Orthogonal expert init (symmetry breaking)

If every routed expert starts from the same distribution, they begin
near-identical and the router has no reason to prefer one over another — they
can drift together and waste capacity. OSRT initialises each expert's
projections to be **orthogonal**, in a *different* random subspace per expert,
so gradients push them apart from step 1:

```python
# model.py:120-132 (orthogonal_expert_init)
# w_gate, w_up: (hidden, dim) — columns span a subspace of R^hidden
# w_down: (dim, hidden) — rows span a subspace of R^dim
# Uses QR decomposition of a random matrix (deterministic given seed).
```

The crucial operational detail is **when** this runs. HuggingFace's
`post_init()` walks the module tree and calls `_init_weights` on every
`nn.Linear`, which would *stomp* the orthogonal weights. So the orthogonal
init is **deferred** — requested in the constructor but applied via
`apply_orthogonal_init()` *after* `post_init()` has finished:

```python
# model.py:272-275
# NOTE: orthogonal expert init is NOT applied here because HF's
# post_init() walks the module tree and calls _init_weights on every
# nn.Linear, which would stomp the orthogonal weights. Apply via
# apply_orthogonal_init() after post_init() has finished.
```

Each expert gets a distinct seed (`self._moe_seed * 1000 + ei`,
`model.py:341-345`) so the symmetry-breaking is reproducible.
`expert_orthogonal_init` defaults to `True` (`config.py:245`).

(The QR routine itself carries a subtle correctness fix: for "fat" matrices
where `hidden > dim` the orthonormal-column variance is `1/sqrt(rows)`, not
`1/sqrt(cols)`, which previously left `w_gate`/`w_up` ~13% under their target
std — `model.py:145-157`.)

---

## 10. Telemetry gating (why `.item()` calls are guarded)

`MoELayer.forward` computes a rich set of routing diagnostics — per-token
entropy, marginal entropy, expert fractions, drop rate, router confidence —
across three "views" of the distribution (raw / clean / dispatch). Each stat
ends in a `.item()` or `.tolist()`. On CUDA, **every `.item()` forces a
CPU-GPU synchronisation**, and with 18 effective MoE applications per forward
(3 blocks × 6 loops) that is ~21 syncs × 18 calls — a real throughput hit.

So the whole telemetry block is gated behind a flag:

```python
# model.py:748-749
if not self.telemetry_enabled:
    return shared_out, moe_out
```

The trainer flips `telemetry_enabled = False` on non-logging steps via
`OSRTForCausalLM.set_moe_telemetry(False)` (`model.py:1798-1800`), which delegates
to `OSRTModel.set_moe_telemetry` (`model.py:1729-1745`); that method loops over the
blocks and sets each `blk.moe.telemetry_enabled` (`model.py:1744-1745`) — and the
*same* method also gates the recursive-loop collapse hook (chapter 06 §4.4). The `last_*`
attributes simply retain their previous-logging-step values; that is safe
because the consumers only read them on logging steps too (`model.py:870-879`).
Default is `True` so downstream tools work without opt-in (`model.py:294-297`).

### Dead-expert collapse signal

On logging steps the trainer's `_collect_moe_metrics` (`train.py:305`) reads the
gated per-loop expert-fraction telemetry and derives a **dead-expert count**: per
`(block, loop)` it counts experts whose load fell below **10% of the uniform
share** `1/E` (`train.py:409-414`), then accumulates a global
`moe/dead_experts_total` (`train.py:465`). A rising total is the single clearest
sign the router is collapsing onto a handful of experts (§13) — it is surfaced in
W&B and in the stdout `collapse:` line alongside `bias_abs_max`
(`train.py:1222-1227`). Because it is built purely from the already-gated
telemetry, it never runs on the compiled fast path.

---

## 11. How the block combines the two outputs (two-level gating)

The `MoELayer` returns `(shared_out, routed_out)`; the enclosing `Block`
combines them with a **second, per-block scalar gate** applied to the routed
branch only:

```python
# model.py:1133
return h_shared + self.effective_moe_gate() * h_routed
```

`moe_gate` is a single learned parameter, but reparameterised through
`softplus` so its *effective* value is always positive:

```python
# model.py:962, 976-977
self.moe_gate = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))  # softplus(raw) ≈ 1.0
def effective_moe_gate(self):
    return F.softplus(self.moe_gate)
```

So there are **two levels of gating**:

1. *Inside* `MoELayer`: per-token, renormalised top-2 gates that mix the
   chosen routed experts (§4).
2. *Outside*, in the `Block`: a per-block scalar that scales the whole routed
   contribution relative to the always-full-weight shared branch.

The softplus reparameterisation (init `log(e−1) ≈ 0.5413`, so
`softplus ≈ 1.0`) is a direct guard against a v4 failure mode where an
unbounded gate could drift negative and **zero out the routed branch
entirely**, leaving the shared expert to do all the work — the "dense crutch"
(`model.py:953-959`). With softplus the gate can shrink but can never cross
zero.

---

## 12. Parameter and compute cost

Do not hand-derive these — `scripts/compute_budget.py` is the source of
truth. Running it on the canonical preset (`OSRT_605M_A288M`) reports:

```
cfg: dim=1536 vocab=65536 blocks=3 loops=6 kv_heads=8
     experts=28 top_k=4 h_routed=2112 h_shared=2816 rank=256 mtp=2
----------------------------------------------------------------
  embedding           100,690,944
  attention            17,308,032
  shared_expert        38,928,384
  routed_experts      424,673,280
  router                   36,867
  ...
  TOTAL PHYSICAL      601,444,393  (~601M)
  ACTIVE / TOKEN      278,217,769  (~278M, 46.3% of physical, excl. MTP)
```

Takeaways:

- **Routed experts (424.67M) are ~71% of the physical model** — the dominant
  term — yet only **top-2 of 8 = 25% of them** are active per token. That is
  the whole MoE bargain: store a lot, compute a little.
- The **shared expert is 38.93M** and the **router is tiny (36,867 params)** —
  routing is nearly free; the cost is in the experts.
- **Active per token is ~278M (46.3% of physical)** at inference, excluding the
  training-only MTP heads.

> **Stale preset docstring.** `presets.py` claims "~607M physical / ~288.3M
> active" and ships an `OSRT_605M_A279M` alias. The *actual* `compute_budget`
> output is **601M / 278M (46.3%)**. The headline name and docstring drifted
> from a stale solve; trust the live `compute_budget.py` numbers above.

---

## 13. Failure modes this design guards against

The MoE block is shaped by hard lessons from earlier OSRT iterations
(`LEARNINGS.md`). Each mechanism above maps to a failure it prevents:

- **Router collapse** (a few experts win everything). Guarded by the
  *combination* of the non-learned balance bias (§5.1–5.2), the Switch
  balance loss (§5.3), the z-loss (§5.4), Gumbel exploration (§6), and — on the
  loop dispatch path only — capacity-driven drops (§7; the grouped-GEMM default
  is dropless, so it leans on the other four). The redundancy is deliberate —
  each handles a different regime (controller vs gradient, magnitude vs
  distribution, warm-up vs steady-state). **Detected** at runtime by the
  dead-expert count (§10, `moe/dead_experts_total`): experts whose load drops
  below 10% of the uniform share, accumulated across all blocks and loops.

- **Expert starvation / under-utilisation** (cold experts that never learn).
  Guarded by Gumbel exploration keeping losers warm (§6), orthogonal init
  giving each expert a distinct starting subspace (§9), and the choice of
  **8 routed experts instead of 12** so top-2 yields denser routing (25% vs
  16.7%) — more capacity per token, less idle capacity at 601M scale
  (`presets.py:31`, `ARCHITECTURE.md §7.1`).

- **The "dense crutch"** (routed branch zeroed out, shared expert does
  everything). Guarded by the softplus-reparameterised `moe_gate` that can
  never cross zero (§11, `model.py:953-959`).

- **Chunk-instability at inference** (prefill vs decode routing mismatch from
  capacity drops). Guarded by **drop-free eval** on the loop path
  (`model.py:782-788`) and, by construction, by the dropless grouped-GEMM
  dispatch the preset uses (§7).

- **Loop-depth collapse** — the project's single biggest discovery: probing
  showed one recursive loop was doing ~90% of the cross-entropy reduction
  while the others were near-idle (`LEARNINGS.md §1.1`). **Important
  scoping:** this is *not* fixed inside `MoELayer`. It is addressed at the
  loop/training level by `aux_loop_loss_weight` (apply the LM head at every
  loop) and `loop_dropout_prob` (randomly truncate the loop chain so no single
  loop is load-bearing). The MoE block's contribution is only *loop-aware
  routing* — the per-loop loop embedding (§4) and the per-loop bias rows
  (§5.1) — which let each recursive pass specialise its routing. The actual
  collapse fix lives in the recursion/training code, not here.
