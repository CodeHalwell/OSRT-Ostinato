# Quantization & Deployment

> **Updated to v7 (2026-09-01).** Config values, parameter counts, expert
> layout, tokenizer and optimizer recipe below are the committed v7 shape
> (`OSRT_V7`: 968,468,355 physical / 263,035,779 active). `file:line`
> citations drift as the code moves — `src/osrt/` is ground truth and
> `scripts/compute_budget.py` is the only source for any count. Passages that
> explain a *v6* choice are marked as such where they survive. Decisions and
> open gates: `specs/2026-08-11-v7-roadmap.md` §14, §16, §19.


> Part of the OSRT `docs/` architecture series. This chapter explains how a
> **968M-parameter (physical) mixture-of-experts model** is squeezed down to run on small /
> edge hardware (phones, a Raspberry Pi 5): the deployment memory budget and why
> the routed experts dominate it, the **implemented** int4 KV-cache quantizer in
> `src/osrt/quant.py`, the **planned** AlphaQ expert quantization, the full
> deployment stack, and the levers for hitting a tighter envelope.

A note on sourcing. Where this document states a *mechanic that exists in code*
it cites `src/osrt/quant.py` by line — that is the source of truth. The
parameter counts come from `scripts/compute_budget.py` (run live; numbers below
are its output). `ARCHITECTURE.md §13`–`§15` is cited for *design intent* and
the deployment plan. **Be careful about the word "implemented":** only the
int4 KV quantizer is code. The int8 base and FP4/AlphaQ expert quantization are
*design intent* in `ARCHITECTURE.md §14` — they are not coded yet. Each section
is labelled IMPLEMENTED or PLANNED, and section 7 has the full table. Where the
code and the spec disagree, the code wins and the discrepancy is flagged.

A naming note. The v6 lineage put parameter counts in names (OSRT-605M,
"OSRT-600M") and the counts drifted; v7 does not, and `scripts/compute_budget.py`
is the only source for any count. This document uses its v7 output —
**968,468,355 physical / 263,035,779 active** — for all memory math. Where an
older chapter or `ARCHITECTURE.md` quotes a different figure it is a v6 number;
regenerate, do not average.

---

## 1. Purpose — getting a 968M MoE onto edge hardware

OSRT is a mixture-of-experts model. Its *physical* parameter count (968M) is
large because it stores many routed experts, but only a fraction
(~263M, 27%) is *active* per token — the router picks the top-4 of 28 experts per
MoE block (`compute_budget.py` reports `ACTIVE / TOKEN 263,035,779`). That gap
is the whole point of MoE: lots of stored knowledge, cheap per-token compute.

But "cheap compute" does not mean "cheap memory". On a phone or a Pi, **every
stored weight has to live in RAM** (you cannot fault a top-4 expert in from disk
at decode latency without an inference system that explicitly pages — see §6).
At bf16, 968M params is ~1.9 GB of weights alone, before the KV cache and
activations. That is too big for a comfortable mobile resident set.

Quantization is the answer: store each weight in fewer bits. The goal of this
chapter's stack is to get the **resident weights to ~510 MB** as specified
(§2), and down toward **~150–250 MB** with the aggressive levers (§6) — small
enough to sit alongside an OS and an app. Two facts drive every decision:

1. **The routed experts are ~84% of the physical weights** (817.5M of 968M).
   Whatever you do to them dominates the budget. Expert quantization is *the*
   deployment lever.
2. **The KV cache grows with context** (`ARCHITECTURE.md §13.2`: 18 KB/token,
   72 MB at 4K). Unlike weights it is unbounded, so it gets its own quantizer.

---

## 2. The deployment memory budget (per-component)

The real per-component parameter breakdown, straight from
`scripts/compute_budget.py` (run `PYTHONPATH=src python3 scripts/compute_budget.py`):

```
  embedding            75,694,080    ( 7.8%)
  attention            17,308,032    ( 1.8%)
  shared_expert        53,084,160    ( 5.5%)
  routed_experts      817,496,064    (84.4%)   ← the dominant term
  router                  129,027
  mtp_heads             4,721,664    ( 0.5%)   ← DROPPED at deploy
  loop_emb                 27,648
  norms_misc                7,680
  ----------------------------------------------
  TOTAL PHYSICAL      968,468,355    (~968M)
  ACTIVE / TOKEN      263,035,779    (~263M, 27.2% of physical, excl. MTP)
```

Two things to read off this table.

**Routed experts are ~84% of the model.** (`compute_budget.py` reports 84.4%
for the v7 shape; `ARCHITECTURE.md §14.2` rounds to 84%. v6's figure was 71%
— the re-grain to 28 experts made the conclusion sharper, not different.)
If you only quantize one thing aggressively, quantize these. Every other
component combined is smaller than the routed experts alone.

**MTP heads are dropped at deployment.** The 4.72M multi-token-prediction head
params are a *training* aid (they sharpen representations by predicting two
tokens ahead). At inference you only need the main next-token head, so the MTP
heads contribute **0 bytes** to the deployed model. The active-per-token figure
above already excludes them.

### The on-disk estimate (~510 MB, the as-specified stack)

Applying the `ARCHITECTURE.md §14.1` per-component plan — int8 base, mixed-FP4
routed experts, bf16 for the small sensitive bits — gives (`ARCHITECTURE.md
§14.2`; decimal MB, 1 MB = 1,000,000 bytes, no allocator/metadata overhead):

```
Embedding (int8, 75.7M × 1 byte)                      76 MB
Attention (int8, 17.3M)                               17 MB
Shared experts (int8, 53.1M)                          53 MB
Routed experts (mixed FP4 @ ~3.5 bit avg, 817.5M):
    817.5M × 3.5 bits / 8  ≈ 358 MB (+~2% AlphaQ meta) ~365 MB
HRA adapter (post-training only; off in pretraining)   0 MB
router + norms + loop_emb (bf16, 0.16M × 2 bytes)     <1 MB
MTP heads (dropped at deploy)                          0 MB
  ----------------------------------------------------------
TOTAL ON DISK / RESIDENT                             ~510 MB
```

The routed-expert row (~365 MB) is over 70% of the total even at 3.5 bits —
exactly because they are 84% of the params. The embedding (76 MB) is the
second-biggest target precisely because it is the next-largest component.

### 510 MB vs the "~250 MB" headline — different scenarios, and a v6 number

`ARCHITECTURE.md §15.3` used to quote a "~250 MB" total inference footprint
(weights ~150–200 MB + ~5 MB KV + ~50 MB activations), derived for the v6
shape. It describes a **different scenario**, not a conflicting number:

- **~510 MB** = the full v7 stack *exactly as specified* in §14.1 (int8 base,
  FP4 routed at 3.5 bit). This is the honest baseline.
- **~150–250 MB** = the footprint *after* the tighter-envelope levers of §6
  **and** an active-only resident loading scheme. With v7's 3× larger routed
  pool the weight levers alone land near ~320 MB (§6 has the arithmetic); the
  headline is only reachable with the residency assumption.

Quote 510 MB when you mean "the spec'd weights"; quote ~150–250 MB only with
the residency assumption stated. Always say which.

---

## 3. KV cache quantization — IMPLEMENTED (`src/osrt/quant.py`)

This is the one piece of the deployment stack that **exists as code today**.
A critical caveat first, straight from the module docstring
(`src/osrt/quant.py:10-15`):

> "This is a STANDALONE deployment / RL-rollout utility. It is NOT wired into the
> training forward ... and is not enabled in `generate()` by default — the model
> code is unchanged."

So "implemented" means *the code exists and round-trips correctly* — a caller
(e.g. an offline rollout collector) has to call `quantize_kv_latent()` /
`dequantize_kv_latent()` explicitly. The default model does **not** quantize its
cache. Keep that distinction: implemented ≠ active.

### 3a. Why the cache needs compressing

The model caches only the **K_DOWN latent** — the compressed key — not full K
and V (`ARCHITECTURE.md §13.1`; V is recomputed from K via `V = W_V_FROM_K @ K`,
which already halves the cache). Even so, the cache is **18 effective layers ×
512 floats/token = 18 KB/token in bf16**, growing linearly with context:
**72 MB at 4K, 144 MB at 8K** (`ARCHITECTURE.md §13.2`). Unlike the weights,
this term has no ceiling — long contexts blow past the weight budget. int4 cuts
it 4× (bf16 → 4 bits): to **~9–18 MB at 4K** (`ARCHITECTURE.md §13.3`), and a
sliding window on top can reach ~2–5 MB.

The quantizer compresses the cached **K_DOWN latent** to symmetric int4. The
public entry points are `quantize_kv_latent()` (`quant.py:167`) and
`dequantize_kv_latent()` (`quant.py:224`).

### 3b. Random rotation for outlier mitigation

This is the "Turbo" in TurboQuant. Symmetric int4 has only **15 usable levels**.
A single large-magnitude ("outlier") channel in a block forces a large
quantization step, wasting precision on all the small channels — the classic
low-bit KV problem (`quant.py:17-33`).

The fix: apply a fixed **orthogonal rotation** to the block *before* quantizing.
A rotation mixes every channel into every other, so a lone outlier is *spread*
across the whole block and the per-block max — which sets the step size — drops
toward the block **RMS** rather than the **peak**. Because the rotation is
orthogonal it is exactly invertible: dequant rotates back with the transpose.

The rotation is built by `make_rotation()` (`quant.py:94`). For a power-of-two
block size it uses a **randomized Sylvester-Hadamard** matrix — a Hadamard
matrix with a random ±1 sign flip on the diagonal (`quant.py:81-86`):

```python
h = _hadamard_matrix(dim, device="cpu", dtype=torch.float32)
signs = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.float32) * 2.0 - 1.0  # ±1
r = h * signs.view(1, -1)
```

Two subtleties worth internalising:

- **Why the sign flip?** A *fixed* Hadamard matrix can accidentally align with
  the data axes and fail to spread an outlier. The random ±1 signs are what
  *randomize* it (`quant.py:96-100`) so it doesn't align with any particular
  channel layout.
- **Why Hadamard and not just a random matrix?** `H` is symmetric and orthogonal
  (`H @ H.T == I`), and normalizing by `1/sqrt(n)` makes it norm-preserving
  (`quant.py:62-74`). For non-power-of-two block sizes the code falls back to a
  general random orthogonal matrix — the Q-factor of a QR decomposition of a
  seeded Gaussian (`quant.py:87-90`). Both satisfy `R @ R.T == I`, so the
  inverse rotation is just `R.T` and no separate inverse is stored.

The rotation happens inside `quantize_kv_latent()` (`quant.py:198-204`): reshape
the last dim into blocks, then `blocked @ r`. On dequant it is undone with
`work @ r.T` (`quant.py:237`).

> **The rotation is internal to the round-trip — it never touches the cache
> layout.** Do not confuse it with RoPE. `ARCHITECTURE.md §13.4` stores the
> *un-rotated* K_DOWN in the cache (RoPE is applied at attention time so the
> linear KDV (Key-Derived Value) K→V relationship survives). The TurboQuant
> Hadamard rotation lives *entirely inside* `quantize_kv_latent` →
> `dequantize_kv_latent`; the dequantized latent comes back in the original
> basis. The cache stores int4 codes, not "rotated K".

### 3c. The symmetric int4 grid [−7, 7] and why −8 is dropped

int4 two's-complement spans [−8, 7] — 16 levels. The quantizer deliberately
uses only the **symmetric 15-level grid [−7, 7]**, setting `INT4_QMAX = 7`
(`quant.py:51`). The reasoning (`quant.py:45-50`):

> "We use the symmetric 15-level grid [-7, 7] (dropping the asymmetric -8 level)
> so that quantize(-x) == -quantize(x) exactly — the rotation produces
> zero-mean, near-symmetric blocks, and a symmetric grid avoids a half-step DC
> bias."

The rotation produces zero-mean, near-symmetric blocks (channels mixed together
average out). If you kept the asymmetric −8, the grid would have one more
negative level than positive, introducing a small **DC (mean) bias** on data
that is genuinely centred at zero. Dropping −8 costs one level but keeps the
quantizer mean-preserving — a good trade for zero-mean data. The scale is then
symmetric: `scale = max|.| / 7` per block, and codes round to `[−7, 7]`
(`quant.py:210-212`):

```python
amax = work.abs().amax(dim=-1, keepdim=True)
scale = (amax / INT4_QMAX).clamp_min(1e-12)
codes = torch.round(work / scale).clamp_(-INT4_QMAX, INT4_QMAX).to(torch.int8)
```

The `clamp_min(1e-12)` guards an all-zero block from dividing by a zero step
(it just quantizes to all zeros).

### 3d. Nibble packing (the 2× storage win)

The codes come out as an `int8` tensor — one int4 value per *byte*, which wastes
half the storage. When on-disk / in-RAM size actually matters, `pack_int4()`
(`quant.py:112`) packs **two int4 nibbles into one `uint8`**, halving the
footprint:

```python
nib  = (codes.to(torch.int16) & 0x0F).to(torch.uint8)  # 4-bit two's complement
low  = nib[..., 0::2]
high = nib[..., 1::2]
return (low | (high << 4)).to(torch.uint8)
```

`unpack_int4()` (`quant.py:127`) reverses it, sign-extending each 4-bit nibble
back to a signed int8 (`low = torch.where(low >= 8, low - 16, low)`,
`quant.py:133-134`). Without packing, int4 codes-in-int8 would be no smaller
than int8 storage — packing is what makes the int4 cache actually 4× smaller
than bf16.

### 3e. The rotation matrix is `lru_cache`'d (deterministic, build-once)

Building a Hadamard / QR matrix on every token would be wasteful. The CPU-side
build `_cached_rotation_cpu(dim, seed)` is wrapped in `@functools.lru_cache`
(`quant.py:77`):

```python
@functools.lru_cache(maxsize=32)
def _cached_rotation_cpu(dim: int, seed: int) -> Tensor:
    ...
```

The rotation is a pure function of `(dim, seed)`, so the cache is safe: the same
`(dim, seed)` always yields the *same* matrix. That determinism is load-bearing
for the round-trip — encode and decode rebuild the **identical** rotation from
the stored `seed` (`QuantizedKV.seed`, `quant.py:158`), so the matrix is never
persisted (no side channel, nothing matrix-sized on disk; `quant.py:31-33`).
`make_rotation()` just moves the cached CPU matrix to the requested device/dtype
(`quant.py:105-106`).

### The round-trip container

`QuantizedKV` (`quant.py:142`) bundles everything dequant needs: the int4
`codes`, the per-block `scale`, the rotation `seed`, the `block_size`, the
original `orig_dim` (for un-padding), and whether rotation was applied
(`rotated`). `kv_quant_rel_error()` (`quant.py:243`) reports the relative
reconstruction error `‖x − x̂‖ / ‖x‖` — a handy calibration check.

---

## 4. Expert quantization — PLANNED (AlphaQ)

> **Status: design intent, not yet coded.** A search of `src/` finds *no*
> AlphaQ, FP4, PL-Alpha-Hill, or int8-base implementation — `quant.py` (the
> int4 KV quantizer) is the only quantization code in the tree. Everything in
> this section is from `ARCHITECTURE.md §14.1/§14.3` and describes what the
> deployment plan *intends*, not what runs today.

The routed experts are 71% of the model (§2), so their bit budget decides the
deployment size. The plan is **AlphaQ**: a calibration-free, mixed-precision
allocation of bits across experts.

The core idea (`ARCHITECTURE.md §14.3`):

- For each routed expert weight matrix, compute the **PL Alpha Hill** metric —
  a heavy-tailed self-regularization measure from the weight's eigenvalue
  spectrum. A *heavy-tailed* spectrum signals a well-trained, important matrix;
  a *light-tailed* one signals a less critical matrix. Crucially this needs
  **no calibration data** — it reads the weights directly.
- An ILP solver allocates **2, 3, or 4 bits per expert per layer** under a
  **global budget averaging ~3.5 bits/expert**.
- **Heavy-tailed (important) experts get 4 bits; light-tailed ones get 2 bits.**
  Each up/gate/down projection is allocated independently.

The expected result is **near-lossless quality at a 3.5-bit average**
(`ARCHITECTURE.md §14.3` cites AlphaQ results on Qwen1.5-MoE, a similar
fine-grained many-expert regime). Note this is *mixed* precision — the §14.1 table's
"FP4 (MXFP4)" label is the base format, but AlphaQ then varies the per-expert
bit-width 2/3/4 around it. Describe it as **mixed FP4**, not uniform FP4.

This is where the ~365 MB routed-expert figure of §2 comes from
(817.5M × 3.5 bits / 8). Until AlphaQ is coded, that figure is a *projection*.

---

## 5. The full deployment stack — PLANNED (int8 base + FP4 routed + int4 KV)

The complete deployment plan layers three quantization regimes, each buying a
different thing. Only the last layer (int4 KV) is implemented today.

| layer | format | what it buys | status |
|---|---|---|---|
| **Base weights** (embedding, attention, shared experts) | int8, symmetric per-channel | 2× over bf16 on the ~157M "dense" params; int8 is near-lossless for these | PLANNED |
| **Routed experts** | mixed FP4 (MXFP4 + AlphaQ 2/3/4-bit) | the big win — ~5× over bf16 on the 84% term, ~365 MB | PLANNED |
| **KV cache** | int4 TurboQuant (rotation + symmetric grid + nibble pack) | 4× over bf16 on the *runtime* cache; bounds context growth | IMPLEMENTED (`quant.py`, standalone) |
| HRA adapters, router, loop-emb, norms | bf16 | kept full precision — small and sensitive | (no quant needed) |

**Why each layer, and the order to apply them:**

1. **int8 base first.** The embedding (76 MB after int8) and the dense
   attention / shared-expert weights are large but tolerate int8 essentially
   losslessly with per-channel scales. This is the cheap, safe 2× — do it first.
2. **FP4/AlphaQ on the routed experts** — the dominant lever. 4× to 5×
   compression on 84% of the model, allocated so the experts that matter keep
   their bits. This is what gets you from ~970 MB (int8-everything) down to
   ~510 MB.
3. **int4 KV at runtime, separately.** The KV cache is not a weight — it is
   produced during decode and quantized per-token by the caller. It does not
   change the on-disk weight size at all; it bounds the *runtime* memory so a
   long context does not dwarf the weights.

**Leave alone:** the post-training HRA adapter (v7 pretrains with HRA off),
router, loop embeddings, norms, and biases stay **bf16** (`ARCHITECTURE.md
§14.1`). They are small (router, norms and loop embeddings total under 1 MB;
the adapter is a low-rank delta) and quantization-sensitive — the HRA adapter
carries the post-training behaviour, the router decides expert selection, and
norms/biases are numerically delicate.
Spending bits to shrink them buys almost nothing and risks quality.

---

## 6. Levers to hit a tighter envelope (~150–250 MB)

The ~510 MB stack of §2 is the spec'd baseline. To reach the ~150–250 MB
footprint of `ARCHITECTURE.md §15.3`, pull these levers in priority order
(`ARCHITECTURE.md §14.2`):

1. **Routed experts → 2-bit average (~365 MB → ~210 MB).** They are 84% of the
   model, so this is the dominant lever by a wide margin. Cutting their average
   bit-width from 3.5 to ~2 saves ~155 MB — more than every other lever combined.
   The cost is quality: AlphaQ's whole point is to spend the bit budget where it
   matters, so a 2-bit average leans hard on light-tailed experts being cheap.
2. **Embedding → int4 (76 MB → ~38 MB).** The embedding is the second-largest
   component. int4 halves it again over int8; the tied embedding/LM-head is
   somewhat robust to this, but watch rare-token quality.
3. **HRA adapter folded (0 MB).** v7 pretrains with HRA off, so the baseline
   already carries no adapter bytes. If post-training ships a rank-256 adapter,
   **fold it into the base weights** (it is a low-rank delta) rather than
   shipping it in bf16 — or int8 it if it must stay separate for further tuning.

A fourth, *system-level* lever: **active-only resident loading.** Load just the
top-4 routed experts per layer into RAM and page the rest from disk/CPU. This is
an *inference-system* choice, not a *weight* choice — it changes the resident
set, not the file size — so state the assumption explicitly when you quote a
number that depends on it (`ARCHITECTURE.md §14.2`).

Stacking levers 1–3: routed ~210 MB + embedding ~38 MB + int8 base (attention
~17 + shared ~53) + folded HRA (0) + misc <1 ≈ **~320 MB** of weights. That
does **not** reach the ~150–200 MB weight band v6's `§15.3` quoted: the band was
derived for v6's 424.7M routed pool, and v7's is 817.5M. Getting there needs the
fourth lever. With only the top-4 of 28 experts resident per block (1/7 of the
routed bytes) the stack is ~140 MB with levers 1–3 applied and ~200 MB with the
baseline §2 formats. State the residency assumption whenever you quote either.

---

## 7. What's implemented vs planned

| piece | format / method | status | where |
|---|---|---|---|
| **KV-cache int4 quantizer** | TurboQuant random rotation + symmetric [−7,7] int4 + nibble pack | **IMPLEMENTED** (standalone utility; **not** wired into training or `generate()` by default) | `src/osrt/quant.py` |
| Randomized Sylvester-Hadamard rotation | `±1` sign-flipped Hadamard, QR fallback, `lru_cache`'d | **IMPLEMENTED** | `quant.py:61-106` |
| int4 nibble packing | 2 nibbles / `uint8` | **IMPLEMENTED** | `quant.py:112-136` |
| int8 base weights | symmetric per-channel QAT (embedding, attention, shared experts) | **PLANNED** (no code) | `ARCHITECTURE.md §14.1` |
| Routed-expert FP4 / AlphaQ | mixed 2/3/4-bit, PL-Alpha-Hill + ILP, ~3.5 bit avg | **PLANNED** (no code) | `ARCHITECTURE.md §14.3` |
| HRA / router / norms bf16 | keep full precision | **PLANNED** (deploy policy) | `ARCHITECTURE.md §14.1` |
| MTP-head drop at deploy | omit 4.72M head params | **PLANNED** (deploy policy) | `compute_budget.py`, §2 |

**The one-line summary:** the int4 **KV-cache** quantizer is real, tested code
(`src/osrt/quant.py`) — but it is a standalone utility a caller must invoke, not
something the default model does. The int8 base and the FP4/AlphaQ **expert**
quantization that actually shrink the *weights* to ~510 MB are **design intent
in `ARCHITECTURE.md §14`, not yet implemented**.

---

## 8. Saving & loading the deployed model — HF compliance (IMPLEMENTED)

Quantization decides how *small* the artifact is; this section is about the
*format* it ships in. The model is a `transformers.PreTrainedModel` subclass
(`OSRTPreTrainedModel`, `src/osrt/model.py:1339`) with `model_type = "osrt"`
(`config.py:32`), and `save_pretrained` / `from_pretrained` **round-trip
bit-exact** when the `osrt` package is installed — so a deployed checkpoint
reloads to the identical weights it was saved with.

**The persistent-RoPE fix (load-bearing for correctness).** The RoPE
`cos`/`sin` tables are registered with `persistent=True`
(`model.py:1389-1390`), so they are written into the checkpoint and restored on
load. They were previously `persistent=False`: HF's `from_pretrained` builds the
model skeleton on the **meta device** and then never materialises non-loaded
buffers, leaving `rope_cos`/`rope_sin` as **uninitialised garbage** —
**corrupting RoPE on every reloaded model**. Making them persistent costs ~2 MB
at the real config (negligible) and is what makes the round-trip actually
correct, not just structurally valid (rationale at `model.py:1382-1388`). The
on-the-fly recompute path still handles sequence lengths beyond the cached range
(`model.py:1558-1573`).

**Auto-class registration.** Importing the package registers the model with the
HF auto-classes (`model.py:2569-2576`):

- `AutoConfig.register("osrt", OSRTConfig)` and
  `AutoModelForCausalLM.register(OSRTConfig, OSRTForCausalLM)` let
  `AutoModelForCausalLM.from_pretrained(dir)` and `.from_config(cfg)` resolve the
  class **without naming it** (guarded by `try/except ValueError` so a re-import
  is a no-op).
- `OSRTConfig.register_for_auto_class()` and
  `OSRTForCausalLM.register_for_auto_class("AutoModelForCausalLM")` make
  `save_pretrained` write the `auto_map` into `config.json` — the hook that
  `trust_remote_code` loading reads.

**Deployment-relevant flags on the base class** (`model.py:1348-1351`):

- `supports_gradient_checkpointing = False` — checkpointing is managed
  internally via the private `OSRTModel._osrt_grad_ckpt` gate set by the trainer,
  **not** HF's `gradient_checkpointing_enable()`; advertising `False` stops HF
  attempting its own mechanism (which trips `post_init` and isn't what runs).
- `_no_split_modules = ["RecursiveBlock"]` keeps each recursive block intact
  under `device_map="auto"` sharding.
- A **custom `generate()`** is retained on `OSRTForCausalLM` (the latent-only KV
  cache and recursive decode are not expressible through HF's default loop).

**Caveat — full `trust_remote_code` without the package is a follow-up.**
The Auto* path above is fully functional *with the `osrt` package installed*.
Loading purely via `trust_remote_code` (no package) additionally needs
self-contained modeling/config files in the repo, because osrt's cross-module
imports (`fused_ce`, `hra`, `muon`) are **not auto-copied** by
`save_pretrained`. That consolidation is a separate follow-up
(`model.py:2564-2568`).

---

## 9. Cross-references

- **KV cache layout, size, and decode update:** `ARCHITECTURE.md §13.2` (cache
  growth: 18 KB/token, 72 MB at 4K) and `§13.4` (un-rotated K_DOWN is what gets
  cached; RoPE applied at attention time). A dedicated `docs/` inference chapter
  is forthcoming; until then §13 is the reference.
- **Why V is recomputed from K** (the halving that makes the K-only cache
  possible): `ARCHITECTURE.md §6.3`, `§13.1`.
- **Parameter budget & active-vs-physical:** `scripts/compute_budget.py`,
  `ARCHITECTURE.md §2.1`; HRA adapters `docs/04-hra-adapters.md`; routed experts
  `docs/03-moe-and-routing.md`.
- **Full memory math at inference:** `ARCHITECTURE.md §15.3`.

## Precision on Blackwell — what stock PyTorch actually offers (v7, roadmap §17.1)

The deployment target is the **RTX PRO 6000 Blackwell** (GB202, sm_120, 96GB),
occasionally B200. Its FP4/FP8 tensor cores are the reason the roadmap
un-rejected low-precision pretraining (§13.3). Three primary sources settle
what that buys for *this* model in stock PyTorch:

| path | dtype available | source |
|---|---|---|
| `torch.nn.functional.grouped_mm` — the routed-expert path, **84% of params** | **bf16 only** ("expected to be `torch.bfloat16` … SM≥80") | PyTorch docs |
| FP8 grouped (`_scaled_grouped_mm`) on Blackwell | open issue since June 2025, scoped to SM100, **no sm_120** | pytorch#156238 |
| NVFP4 grouped | **explicitly unsupported**; workaround is one `_scaled_mm` per expert at ~3.1× overhead | pytorch#187095 |

So the expert path is bf16 on the target card unless a third-party grouped
kernel is adopted (CUTLASS block-scaled grouped NVFP4; FlashInfer's
Blackwell-only FP8 grouped GEMM; DeepGEMM). The NVFP4 throughput case survives
for attention and the dense paths only. `grouped_mm` becoming a *public* API
does retire the "private `_grouped_mm`" portability risk the July review
flagged.

**Decode is not bandwidth-bound anyway.** At 263M active in bf16 a decode step
moves ~0.53 GB; on a ~1.8 TB/s card that is a ~3,400 tok/s ceiling against
~136 measured (§13.4) — about 4% of roofline. FP4 *weights* therefore do not
speed decode much; their value is footprint and training throughput. What
binds decode is sequential depth (18 effective layers) and launch count, which
is why the loop-count recommender (ch.08) and a persistent megakernel outrank
weight quantization on the §13.5 lever list.
