# Attention

> **v7 status.** The architecture this chapter describes is current, but its
> **`file:line` citations, parameter tables and config values were written
> against v6** and have not been regenerated. mHC references have been removed
> (roadmap §12.3); expert counts, vocab and param figures may still be stale.
> Regenerate counts with `scripts/compute_budget.py`; `src/osrt/` is ground
> truth where they disagree.


*Part of the `docs/` architecture series for OSRT-605M. This document explains the attention sub-block; see `ARCHITECTURE.md` §6 (attention) and §13 (KV cache) for the original design intent, and `src/osrt/model.py` for the implementation that ships.*

---

## 1. Purpose / summary

Each physical transformer block in OSRT runs an attention sub-block followed by a Mixture-of-Experts sub-block (`RecursiveBlock`, `src/osrt/model.py:1019`). The attention sub-block is grouped-query attention (GQA) with an MLA-style **compressed K/V latent**: 24 query heads attend over 8 key/value heads of head-dim 64, but instead of caching K *and* V it caches a single un-rotated 512-dim latent `c_kv`, reads K straight off it, and derives V from it with one learned linear map. The actual attention is **standard flash SDPA** (`F.scaled_dot_product_attention`) — the score matrix is never materialised. On top of GQA + latent KV it layers two numerical/expressivity refinements that have become standard in 2024–2025 small models: **QK-Norm** (bound the logits) and **RoPE** (inject position). A third refinement — a learnable **attention sink** (let a head attend to "nothing") — was implemented but is now **dropped from the shipping preset** (`attention_sink=False`); its code path still exists behind that flag but is dormant (§6). The method returns a *pre-residual contribution*, which the caller folds back into the stream with a plain residual add. This document walks each piece and explains *why* it is built the way it is.

The whole attention stack costs **17,308,032 parameters across the 3 blocks** (`scripts/compute_budget.py` — see §8).

---

## 2. Why GQA + KDV (Key-Derived Value, the memory motivation)

Attention's parameter cost is modest; its *memory* cost at inference is not. During autoregressive decode the model must keep, for every past token and every layer, the keys and values it attended to — the KV cache. For OSRT-605M that cache is multiplied by **18 effective layers** (3 physical blocks × 6 recursive loops; see §7 and `ARCHITECTURE.md` §13.2), so anything that shrinks the per-token, per-layer footprint pays off 18×.

Two design choices attack this:

1. **GQA (grouped-query attention).** Full multi-head attention would give every one of the 24 query heads its own K and V head. Instead 24 query heads share **8** KV heads — groups of `group_size = 3` queries point at the same KV head (`src/osrt/model.py:903-907`). That alone cuts the K/V width from `24×64 = 1536` to `8×64 = 512`, a 3× reduction.

2. **KDV (Key-Derived Value, MLA-inspired latent).** Even with GQA, the textbook approach caches both K (512) and V (512) = 1024 floats/token/layer. OSRT instead caches **one** 512-dim latent and reconstructs both K and V from it (`ARCHITECTURE.md` §6.2-6.3). K is read directly off the latent; V is a learned linear function of it — hence **Key-Derived Value (KDV)**: the value at each token is *derived from* its key (the cached latent) by a single learned `Linear(512→512)+bias`. That halves the cache again, to 512 floats/token/layer.

The combined effect (`ARCHITECTURE.md` §13.2): **512 floats/token/layer**, ~18 KB/token across all 18 layers, ~72 MB raw at 4K context in bf16 — versus 144 MB for a standard GQA K+V cache. Further deployment compression (int4 + sliding window) is described in `ARCHITECTURE.md` §13.3; here we only care about the architectural halving.

The trade-off is expressivity: forcing `V = f(K)` is a strictly smaller function class than independent K and V. This was the explicit decision recorded in `ARCHITECTURE.md` §6.2 ("constraint accepted") — the same expressivity class as DeepSeek MLA's shared `c_KV`, judged acceptable at this scale, revisited only if attention quality stalls.

---

## 3. The projections

All four attention projections are created in `RecursiveBlock.__init__` (`src/osrt/model.py:918-930`):

```python
self.norm_attn = nn.RMSNorm(config.dim)
self.q_proj   = nn.Linear(config.dim, self.heads * self.head_dim, bias=False)
self.kv_down  = nn.Linear(config.dim, self.kv_dim, bias=False)
self.v_from_k = nn.Linear(self.kv_dim, self.kv_dim, bias=True)
...
self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
```

With the locked `OSRT_605M_A288M` preset (`src/osrt/presets.py:22-47`): `dim = 1536`, `heads = 24`, `head_dim = 64`, `num_kv_heads = 8`, so `kv_dim = 8 × 64 = 512`.

| Projection | nn.Linear shape | Output | Role |
|---|---|---|---|
| `q_proj` | `1536 → 1536` (`24×64`), no bias | `(B, S, 24, 64)` | full query projection |
| `kv_down` | `1536 → 512`, no bias | `(B, S, 512)` | compress hidden → the **one cached latent** `c_kv` |
| `v_from_k` | `512 → 512`, **bias=True** | `(B, S, 512)` | **KDV (Key-Derived Value):** derive V from the latent: `V = W·c_kv + b` |
| `out_proj` | `1536 → 1536`, no bias | `(B, S, 1536)` | mix concatenated heads back to model dim |

Note the asymmetry that is the heart of the design: there is **no separate `k_proj`**. K is not projected — it is the latent itself, merely reshaped (§4). Only V gets a learned transform, and it is the *only* attention projection with a bias term, because the affine `W·c + b` form is what gives V a degree of freedom K does not have.

The block normalises its input with `norm_attn` (a pre-norm `RMSNorm(dim)`) before any projection (`src/osrt/model.py:998`), the standard pre-LN placement.

---

## 4. The KV latent cache — what's stored, why un-rotated, how K and V are recovered

This is the subtlest part of the block. The relevant code is `_attention`, `src/osrt/model.py:1000-1015`:

```python
c_kv_new = self.kv_down(h)            # (B, S, kv_dim) — un-rotated latent

# The cache holds ONLY the un-rotated latent. K and V are recomputed
# from the full latent every step: RoPE is positional and KDV
# (Key-Derived Value) must operate on un-rotated K, so neither may
# be cached rotated.
if past_key_value is not None:
    c_kv = torch.cat([past_key_value, c_kv_new], dim=1)  # (B, L+S, kv_dim)
else:
    c_kv = c_kv_new
present_kv = c_kv if use_cache else None
total_len = c_kv.shape[1]
past_len = total_len - S

# Derive K and V from the same latent (K = identity reshape; V = KDV).
k = c_kv.view(B, total_len, self.kv_heads, self.head_dim)
v = self.v_from_k(c_kv).view(B, total_len, self.kv_heads, self.head_dim)
```

**What is stored:** exactly `c_kv` — the un-rotated 512-dim latent, one vector per token per layer. That single tensor *is* `past_key_value`, and the freshly concatenated version becomes `present_kv` returned to the caller. Nothing else is cached.

**How K is recovered:** `k = c_kv.view(B, total_len, self.kv_heads, self.head_dim)` (`src/osrt/model.py:1014`). This is an **identity reshape** — no matmul, no parameters. The 512 latent dims *are* the 8 KV heads × 64 head-dim. K is the latent, viewed as heads. (RoPE is then applied to it, §5.)

**How V is recovered:** `v = self.v_from_k(c_kv)...` (`src/osrt/model.py:1015`) — the one learned transform, `Linear(512→512)+bias`, reshaped the same way. This is the *only* place a projection touches the cached latent to produce attention values, and it implements the **Key-Derived Value (KDV)** contract: V at every token is a fixed learned affine map of the *same* latent that defines K.

**Why un-rotated:** RoPE (§5) is position-dependent — it multiplies each head vector by a rotation matrix that depends on the *absolute* position of the token. If we cached K *after* rotation, two things break:

1. We could not re-derive V correctly, because `v_from_k` is a fixed linear map that must see the *content* latent, not a position-rotated one — the KDV (Key-Derived Value) contract is `V = W·c_kv + b` with `c_kv` un-rotated. Rotating first would feed a position-warped vector into a transform that was never meant to absorb position. The linear K→V relationship would be broken (`ARCHITECTURE.md` §6.2 callout, §13.4).
2. RoPE in this codebase is applied relative to a token's position in the full sequence. Caching the raw latent and rotating at attention time means a token's cached representation is position-agnostic and reusable; the rotation is re-applied fresh against the current `total_len` every step.

So the cache is deliberately the *pre-rotation, pre-V-transform* latent, and both K and V are rebuilt from it on every forward — cheap, because K is a reshape and V is one small matmul.

**Position bookkeeping for incremental decode** (`src/osrt/model.py:1010-1011, 1020-1029`): `total_len` is past + new, `past_len = total_len - S`. Queries are the `S` new tokens, so they rotate at positions `[past_len:total_len]`; keys span the whole sequence and rotate over `[0:total_len]`:

```python
q = apply_rope(q, rope_cos[:, past_len:total_len].to(q.dtype),
                  rope_sin[:, past_len:total_len].to(q.dtype))
k = apply_rope(k, rope_cos[:, :total_len].to(k.dtype),
                  rope_sin[:, :total_len].to(k.dtype))
```

---

## 5. RoPE and QK-Norm

### 5.1 QK-Norm (applied first)

Before RoPE, both q and k pass through per-head RMSNorm (`src/osrt/model.py:1017-1019`):

```python
# QK-Norm before RoPE.
q = self.norm_q(q)
k = self.norm_k(k)
```

`norm_q` and `norm_k` are `nn.RMSNorm(config.head_dim)` (`src/osrt/model.py:928-929`). "Per-head" here means the RMS normalization runs **over the 64-dim head axis** — each head vector is independently normalised to unit RMS. It does **not** mean 24 (or 8) separate norm modules: there is a single learnable weight vector of length `head_dim`, *shared across all heads* (the code comment at `src/osrt/model.py:925-927` says exactly this: "Per-head (head_dim) is the standard formulation; sharing the norm parameter across heads keeps the addition lightweight (~head_dim params per block) and matches Gemma2/Chameleon").

**Why:** the attention logit is `q·k / √d`. If q or k grows large in magnitude, the logits explode, the softmax saturates, and in bf16/fp8 the whole thing becomes numerically unstable — and any pathology in the attention output is inherited downstream by the MoE router (`src/osrt/model.py:922-924`). Normalising q and k to unit RMS bounds `|q·k|`, so logits cannot run away. This matters more here than usual because the model is Muon-trained, and Muon-trained models are known to be prone to logit growth (`ARCHITECTURE.md` §6.4; Kimi K2 added "QK-Clip" on top — OSRT relies on QK-Norm plus Muon's own stability).

### 5.2 RoPE (rotary position embedding)

The frequency tables are precomputed once at model init (`src/osrt/model.py:1239-1244`) over the full head dimension:

```python
cos, sin = compute_rope_freqs(
    config.max_position_embeddings,
    config.head_dim,            # = 64
    config.rope_theta,          # = 10000.0
    scaling=config.rope_scaling,
)
```

`compute_rope_freqs` (`src/osrt/model.py:47-77`) builds the standard inverse-frequency schedule `1 / θ^(2i/d)` and tiles cos/sin to shape `(1, seq_len, 1, dim)`. It supports NTK-style θ rescaling for context extension via `rope_scaling` (`src/osrt/model.py:58-63`), which is how the model is meant to stretch beyond its training context in mid-training (`ARCHITECTURE.md` §6.5).

`apply_rope` (`src/osrt/model.py:80-90`) is the rotate-half convention:

```python
d = x.shape[-1] // 2
x1, x2 = x[..., :d], x[..., d:]
x_rot = torch.cat([-x2, x1], dim=-1)
return x * cos + x_rot * sin
```

It splits the 64-dim head vector into two halves of 32 and pairs dim *i* with dim *i + 32* (not adjacent dims *i*/*i+1*). It also casts the fp32-precomputed cos/sin to the activation dtype so attention stays in bf16 under autocast (`src/osrt/model.py:81-86`).

> **Code-vs-spec note — "partial" RoPE is not actually partial.** `ARCHITECTURE.md` §6.5 calls this "Partial RoPE … applied to the last 64 dimensions only." But `head_dim` *is* 64, so "the last 64 of 64" is the **entire** head vector — and the implementation confirms this: `compute_rope_freqs` is called with the full `config.head_dim` (`src/osrt/model.py:1241`) and `apply_rope` rotates the whole vector (`src/osrt/model.py:80-90`). There is no `rotary_dim < head_dim` split anywhere in the code. **At this configuration, RoPE is full, not partial.** Making it genuinely partial would require a code change — computing the frequency table over a rotary *slice* of the head dim and leaving the remaining dims position-free. Treat §6.5's "partial" as aspirational, not a shipping feature.

---

## 6. Attention sink — what it was, why it was dropped, the dormant path

> **Status: DROPPED.** The shipping preset now sets `attention_sink=False` (`src/osrt/presets.py:54`). Attention runs through standard flash SDPA (§6.4). The sink code — the `sink_logits` parameter (`src/osrt/model.py:1062-1074`) and the `_attention_with_sink` method (`src/osrt/model.py:1187-1251`) — still **exists** behind the flag, but with the flag off it is never constructed or called. This section documents what it was and the OOM finding that retired it.

### 6.1 The idea

A standard softmax forces a query's attention weights to sum to exactly 1: the query *must* spread its attention over the available keys, even when none of them is relevant. An attention sink relaxes that. It adds one extra, learnable term to the softmax **denominator only** (`src/osrt/model.py:1062-1074`):

```python
self.attention_sink = config.attention_sink
if config.attention_sink:
    self.sink_logits = nn.Parameter(torch.zeros(self.heads))
```

The math (from the docstring at `src/osrt/model.py:1067`):

```
s_{h,i,j} = exp(z_{h,i,j}) / (Σ_k exp(z_{h,i,k}) + exp(sink[h]))
```

`sink_logits` is one scalar **per query head** (length 24 — `torch.zeros(self.heads)`), initialised to zero so the sink contributes `exp(0) = 1` to the denominator at step 0. Because the sink has no value vector, it never enters the numerator — it only ever *removes* probability mass. The effect: a head's real attention weights are now allowed to sum to **less than 1**, i.e. the head can attend to "nothing" when nothing is relevant (`ARCHITECTURE.md` §6.6). With the flag off, `sink_logits` is never even allocated (the `if config.attention_sink` guard at `src/osrt/model.py:1073` is false), and the model is 72 params lighter than it was with the sink on.

### 6.2 The dormant manual path

When the sink *was* enabled, the branch in `_attention` (`src/osrt/model.py:1165`) routed to a hand-written attention (`_attention_with_sink`, `src/osrt/model.py:1187-1251`) instead of SDPA. PyTorch's fused `F.scaled_dot_product_attention` (flash attention) computes a *plain* softmax and gives you no hook to add a term to its denominator, so the sink path had to materialise the score matrix explicitly.

The clever part is that it does **not** redo a "softmax with sink." It computes the ordinary masked softmax output `out` and the per-query log-sum-exp `lse = log Σ_k exp(z)`, then rescales (`src/osrt/model.py:1242-1251`):

```python
scores_f = scores.float()
lse = torch.logsumexp(scores_f, dim=-1)
attn_weights = torch.softmax(scores_f, dim=-1).to(v.dtype)
out = torch.matmul(attn_weights, v)

sink = self.sink_logits.float().view(1, H, 1)
rescale = torch.sigmoid(lse - sink).unsqueeze(-1).to(out.dtype)
return out * rescale
```

Adding `exp(sink)` to the denominator simply multiplies the sink-free output by `Σexp(z) / (Σexp(z) + exp(sink)) = sigmoid(lse - sink[h])`, because the sink's value is zero (`src/osrt/model.py:1196-1205`). This is an *exact* rescale, computed in fp32 for stability, not an approximation. GQA broadcasting is done explicitly with `repeat_interleave(self.group_size, dim=1)` on the KV heads (`src/osrt/model.py:1221-1223`), and the causal mask is built to match SDPA's semantics exactly (`src/osrt/model.py:1233-1237`). This code is still present but inert: with `attention_sink=False` the `if self.attention_sink` branch is never taken.

### 6.3 Why the sink was dropped — the seq-8192 OOM

That manual path is exactly what killed it. Flash/SDPA is a *fused* kernel: it never writes the full `S × total_len` score matrix to memory, which is what makes it memory-frugal at long context. But its softmax denominator is fixed — there is no API to inject `exp(sink[h])`. To get the sink you must compute the denominator yourself, which means materialising a `(B, H, S, total_len)` score matrix per head (`src/osrt/model.py:1226`).

At the **seq-8192 instruction phase** that matrix is roughly 12 GB at batch 2 — and because the block is gradient-checkpointed, the scores are *recomputed* in the checkpointed backward, so that cost is paid again. The measured result was an OOM: total memory exceeded 85 GB and the run died. The sink had been kept only because it happened to fit at the earlier seq-2048 phase, and it had shown no demonstrated quality benefit, so it was retired rather than worked around. Switching to flash SDPA (`attention_sink=False`) makes the **same** seq-8192 / batch-2 configuration fit at **35.9 GB** — flash never builds the score matrix, so there is nothing to recompute in the backward (`src/osrt/presets.py:47-54`).

### 6.4 What ships: flash SDPA

With `attention_sink=False` the block uses `F.scaled_dot_product_attention` with `enable_gqa=gqa` so SDPA broadcasts the KV groups internally without materialising repeated heads. There are two sub-cases (`src/osrt/model.py:1172-1183`):

- **Cached decode** (`past_len > 0 and S > 1`): an explicit triangular `attn_mask` is built so the `S` new query positions attend causally over the full `[0:total_len]` span (`src/osrt/model.py:1172-1179`).
- **Prefill / single-token decode**: `is_causal=(S > 1)` — causal for a multi-token prefill, unmasked for a single-token decode step (`src/osrt/model.py:1180-1183`).

Because the sink is off, `sink_logits` is never allocated and the manual path is dead code, so attention is bit-for-bit the standard fused kernel.

---

## 7. How it plugs into the block

`_attention` returns a **pre-residual contribution**, not an updated stream. Its return is `out_proj(attn) + adapter_out` plus the present cache (`src/osrt/model.py:990-994, 1054-1055`):

```python
attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
return self.out_proj(attn_out) + adapter_out, present_kv
```

(The `adapter_out` term is an additive HRA low-rank adapter applied to the sub-block — `src/osrt/model.py:996` — out of scope here; covered in the HRA doc. Just note that what attention returns already folds it in.)

The caller — `RecursiveBlock.forward` (`src/osrt/model.py:1135-1176`) — decides how to mix this contribution into the residual stream, and there are two paths:

- **Standard residual** (`src/osrt/model.py:1170-1175`): plain add.
  ```python
  f_attn, present_kv = self._attention(x, ...)
  x = x + f_attn
  x = x + self._moe(x, loop_idx, token_ids=token_ids)
  ```

(v6 also carried a multi-channel mHC path here, mixing the contribution through a learned doubly-stochastic matrix instead of a bare add. mHC was removed in v7 — roadmap §12.3 — so the plain add above is the only path.)

Attention's job ends at producing `f_attn`; the residual integration is the block's responsibility. This is why the attention method is written to return a contribution and stay agnostic about the stream shape.

---

## 8. Param / compute cost

Per `scripts/compute_budget.py`, the attention category totals:

```
attention            17,308,032
```

across the 3 physical blocks (run `python scripts/compute_budget.py`; the script categorises any parameter whose name contains `q_proj`, `kv_down`, `v_from_k`, `out_proj`, `norm_q`, `norm_k`, or `norm_attn` as "attention" — see `scripts/compute_budget.py:31`).

The per-projection breakdown is given in `ARCHITECTURE.md` §6.2:

| Projection | Params (per block) |
|---|---|
| `q_proj` (W_Q, 1536×1536) | 2.36M |
| `kv_down` (W_K_DOWN, 1536×512) | 0.79M |
| `v_from_k` (W_V_FROM_K, 512×512 + bias) | 0.26M |
| `out_proj` (W_O, 1536×1536) | 2.36M |
| **Four matrices** | **~5.77M / block** |

Three blocks × ~5.77M ≈ 17.3M, matching the aggregate above. The script's `17,308,032` is slightly larger than `3 × 5.77M` because it also folds in the three RMSNorm weight vectors (`norm_attn` over `dim`, `norm_q`/`norm_k` over `head_dim`) and the `v_from_k` bias, which the four-matrix figure omits. (These numbers are taken from `scripts/compute_budget.py` and `ARCHITECTURE.md` §6.2; they are not hand-derived here.)

For context, attention is a small slice of the 605M physical total — the routed experts dominate (`ARCHITECTURE.md` §14.2). The architectural wins of this block are about *KV-cache memory at inference*, not parameter count.

---

## 9. Known caveats / GPU-phase notes

- **Attention is now flash SDPA — the sink no longer constrains it.** The shipping preset sets `attention_sink=False` (`src/osrt/presets.py:54`), so every attention call goes through fused `F.scaled_dot_product_attention` and the `S × total_len` score matrix is never materialised (`src/osrt/model.py:1172-1183`). This is what retired the sink: its manual path materialised a `(B, H, S, total_len)` score matrix (`src/osrt/model.py:1226`), and at the seq-8192 instruction phase the gradient-checkpointed backward recompute of that matrix OOMed (>85 GB at batch 2); flash fits the same config at 35.9 GB (§6.3). The `O(S²)` attention-memory bottleneck this caveat used to warn about is gone with the default preset. The sink code path still exists behind `attention_sink` (default False) and would reintroduce the score-matrix cost only if re-enabled.

- **flex_attention is moot for the default config, recorded only for the dormant sink path.** The `_attention_with_sink` docstring (`src/osrt/model.py:1207-1215`) records that `flex_attention(return_lse=True)` was evaluated as the "flash + lse" route — the natural way to keep a fused kernel *and* recover the log-sum-exp needed for the sink rescale — but was rejected for the current target (torch 2.12 / CPU): it emitted a `return_lse` deprecation warning, materialised the full score matrix anyway without `torch.compile`, and needed a custom `mask_mod` to express GQA broadcasting. With the sink off this is no longer on the critical path — flash SDPA already gives the fused kernel with no score matrix. It remains listed as **GPU-phase** work in `ARCHITECTURE.md:1486` only as the route that *would* be needed if the sink were ever revived.

- **Partial RoPE is a no-op at this config** (see §5.2): `ARCHITECTURE.md` §6.5 describes partial rotary, but with `head_dim = 64` and "last 64 dims" it is full rotation. A future genuinely-partial variant would need code changes.

- **Context extension** is wired but unused by default: `compute_rope_freqs` supports NTK θ-rescaling via `rope_scaling` (`src/osrt/model.py:58-63`), and the forward path recomputes cos/sin on demand when a request exceeds the precomputed range (`src/osrt/model.py:1392-1407`). The default preset ships `rope_scaling=None`.

---

## Quick reference — code map

| Concept | Location |
|---|---|
| `RecursiveBlock.__init__` (projections, sink, norms) | `src/osrt/model.py:901-944` |
| `_attention` (main path) | `src/osrt/model.py:979-1055` |
| `_attention_with_sink` (manual sink path) | `src/osrt/model.py:1057-1121` |
| `compute_rope_freqs` / `apply_rope` | `src/osrt/model.py:47-90` |
| RoPE table init | `src/osrt/model.py:1239-1246` |
| `forward` (residual integration) | `src/osrt/model.py` |
| Preset (heads/kv/head_dim/sink) | `src/osrt/presets.py:22-47` |
| Config validation | `src/osrt/config.py:352-373` |
| Param budget | `scripts/compute_budget.py` |
| Design intent | `ARCHITECTURE.md` §6, §13 |
