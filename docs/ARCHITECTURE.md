# ARCHITECTURE.md — OSRT v7 technical specification

> ## Config-value spec — updated to v7 (2026-09-01)
>
> Section numbering is **frozen**: `src/osrt/` comments cite `ARCHITECTURE.md
> §N` throughout, so §9–§18 keep their numbers and §8 (mHC) is a tombstone.
> Config values and parameter tables below are the **v7 committed shape**
> (`OSRT_V7`); prose that explains v6-era choices is marked as such where it
> survives. Where this file and the code disagree, **the code wins** —
> `scripts/compute_budget.py` for any count, `docs/specs/2026-08-11-v7-roadmap.md`
> §14/§16/§19 for the decisions.


**Scope:** the technical specification of the OSRT-600M model — every
layer, dimension, formula, and connection. The model is **implemented**
in `src/osrt/`; this doc describes that implementation. Where a number
or behaviour matters exactly, **the code is the source of truth** and
this doc is kept in sync with it (param counts via
`scripts/compute_budget.py`, behaviour via `src/osrt/model.py`).

**Companion docs:**
- [`README.md`](../README.md) — design philosophy, why each choice was made
- [`LEARNINGS.md`](LEARNINGS.md) — v5 lessons that shaped these choices
- [`RESEARCH.md`](RESEARCH.md) — external research cited
- `review/` — code reviews; `archive/` — pre-implementation plan reviews

**Reading order:** read README.md first for context, then this doc for
the technical details, then `src/osrt/` for the ground truth.

---

## Table of contents

1. [One-sentence overview](#1-one-sentence-overview)
2. [Parameter budget](#2-parameter-budget)
3. [Tokenizer specification](#3-tokenizer-specification)
4. [Embedding layer](#4-embedding-layer)
5. [Recursive transformer block](#5-recursive-transformer-block)
6. [Attention sub-block](#6-attention-sub-block)
7. [MoE sub-block](#7-moe-sub-block)
8. [Manifold-Constrained Hyper-Connections (mHC)](#8-manifold-constrained-hyper-connections-mhc)
9. [LM head and auxiliary heads](#9-lm-head-and-auxiliary-heads)
10. [Forward pass walkthrough](#10-forward-pass-walkthrough)
11. [Training losses](#11-training-losses)
12. [Inference path](#12-inference-path)
13. [KV cache structure](#13-kv-cache-structure)
14. [Quantization for deployment](#14-quantization-for-deployment)
15. [Total compute and memory math](#15-total-compute-and-memory-math)
16. [Architectural invariants](#16-architectural-invariants)

---

## 1. One-sentence overview

**OSRT** = **Optimized Sparse Recursive Transformer**: a recursive
sparse-MoE transformer with **3 physical decoder blocks applied 6 times
via depth recurrence** (giving 18 effective layers), **28 routed + 1
shared SiTU-GLU experts per block (top-4)**, **GQA attention with a KDV
(Key-Derived Value) compressed KV cache**, **Muon-optimized weights**,
and an optional **rank-256 HRA adapter for post-training** — totaling
**968M physical params, 263M active per token at inference** (27.2%
active fraction; see §2.1 for the full breakdown), ~2.4B FLOPs per
token (§2.3). mHC (manifold-constrained hyper-connections) was removed
in v7 (§8).

> ✅ **ACCOUNTING IS CODE-GENERATED & IMPLEMENTED.** This is no longer
> a paper spec — `src/osrt/` builds the model and all numbers in §2.1
> come from `PYTHONPATH=src python scripts/compute_budget.py`, which
> instantiates the canonical `OSRT_V7` preset
> (`src/osrt/presets.py`) on a meta device and counts real parameters.
> Re-run it after any config change.

> 🔧 **NAMING.** No parameter count appears in any name — repo, package,
> preset or directory (`CLAUDE.md`). The v6 lineage's names drifted from
> the real count four different ways, which is why the rule exists. The
> preset is `OSRT_V7`; §2.1 is the only place a number lives.

> 🔧 **NOT in the architecture:** "gated short convolutions" (claimed
> in an early draft) were never specified or implemented — the spec is
> attention + MoE only.

---

## 2. Parameter budget

### 2.1 Exact accounting (generated 2026-09-30)

> ✅ **GENERATED** — run `PYTHONPATH=src python scripts/compute_budget.py`
> (no args = the canonical `OSRT_V7` preset). The table below is that
> script's output with a per-row explanation added; nothing is
> hand-adjusted. Regenerate it, do not edit it. Do NOT pass loose CLI
> overrides expecting to reproduce the preset — the CLI starts from the
> full preset and only applies explicit `--override k=v` on top.

```
COMPONENT                                       PHYSICAL        ACTIVE / TOKEN (inference)
─────────────────────────────────────────────────────────────────────────────────────
Embedding (49,280 × 1,536, tied with LM head)    75,694,080     75,694,080
  -- one row per token at lookup; full matrix touched at the tied LM head

Attention × 3 blocks (GQA + KDV, §6)            17,308,032     17,308,032
  -- per block: q_proj (1536×1536) + kv_down (1536×512)
     + v_from_k (512×512 +b) + out_proj (1536×1536) + QK/attn norms
  -- ~5.77M/block; the KDV (Key-Derived Value) latent is what makes attention this lean
  -- shared across the 6 loop iterations (params counted once, run 6×)

Shared experts × 3 (SiTU-GLU, h=3,840)           53,084,160     53,084,160
  -- per block: 3 × 1,536 × 3,840 = 17,694,720; always active (absorbed HRA's 14,155,776 — E1)

Routed experts: 3 × 28 × (SiTU-GLU, h=2,112)    817,496,064    116,785,152
  -- per expert: 3 × 1,536 × 2,112 = 9,732,096; per block 272,498,688
  -- top-4 of 28 active per token → 4/28 = 14.3% routing density

Router × 3 (1,536 × 28, plus the scalar moe_gate)   129,027        129,027
Loop embeddings (18 × 1,536)                         27,648         27,648
Norms and misc                                        7,680          7,680

HRA adapters (OFF in pretraining — E1 §18.1)              0              0
  -- with use_hra=True: 18 × (1,536 × 256 + 256 × 1,536) = 14,155,776, fully active

MTP heads × 2 (§9.3)                              4,721,664              0
  -- training-time only; dropped at deploy → 0 active at inference

─────────────────────────────────────────────────────────────────────────────────────
TOTAL PHYSICAL                                  968,468,355  →  ~968M
ACTIVE / TOKEN (inference, excl. MTP)                          263,035,779  →  ~263M
ACTIVE FRACTION                                                    ≈ 27.2%
```

With the training-only MTP heads counted, the train-time active figure
is 267,757,443 (~268M). "Active" is a parameter count, not a FLOP count:
the three blocks each run six times, so per-token compute is ~2.4 GFLOPs
(§2.3), not 2 × 263M.

The canonical preset sets `attention_sink=False`, so no per-head sink
logits are instantiated (§6.6); the row that used to carry them is gone.

### 2.2 At-a-glance

- **Hidden dimension `d_model`**: 1,536
- **Vocab size**: 49,280 rows / 49,184 real (OSRT-Ostinato: SmolLM2 base + 32 specials)
- **Physical transformer blocks**: 3
- **Recursive loops**: 6 → 18 effective layers
- **Attention**: GQA 24 query heads / 8 KV heads / head_dim 64
- **MoE**: 1 shared expert (h=3,840) + **28 routed (h=2,112)**, top-4, Quantile Balancing, SiTU-GLU
- **HRA**: **off** in pretraining (E1, 2026-09-02); rank-256 adapter on the normalised input for SFT/GRPO
- **HRA adapter rank**: 256 (real high-rank, not LoRA-style 16)
- **HRA injection points**: 18 (implementation-defined; see §2.4)
- ~~**mHC expansion**~~ — removed in v7 (§8; roadmap §12.3). v6 ran a 4×
  residual stream width; v7 keeps one ladder slot as insurance, nothing more
- **Position encoding**: Partial RoPE (last 64 dims of Q and K)
- **Activation**: SiTU-GLU (FFN, v7; SwiGLU + clamp kept as the A/B), Sqrt(Softplus) (routing affinity)
- **Norm**: RMSNorm pre-norm on both sub-blocks, plus an RMSNorm reset between loops and before the head (§5.3; there is no post-sub-block norm)

### 2.3 FLOP count per token (forward pass, one inference)

Approximate, derived from the §2.1 active-param breakdown (2 FLOPs per
active MAC). Per effective layer (one block × one loop):

```
18 effective layers × (
    attention (q/kv_down/v_from_k/out, ~5.77M params)  : ~2 × 5.77M   = ~11.5M FLOPs
  + shared expert (17.69M params)                       : ~2 × 17.69M  = ~35.4M FLOPs
  + routed top-4 (4 × 9.73M = 38.93M active)            : ~2 × 38.93M  = ~77.9M FLOPs
  + router (43K) + norms                                :               ~1M FLOPs
)  ≈ 18 × ~126M  = ~2.27B FLOPs
+ embedding lookup (negligible) + tied LM head (~2 × 75.7M = ~151M)

TOTAL: ~2.4B FLOPs per token (forward); ~7.2B with backward

(Approximate — FLOP definitions vary, and attention's score/value
products are excluded. Use as ratios. For exact param counts see §2.1;
this estimate is hand-derived from them. HRA adds ~2 × 0.79M per
effective layer when it is on.)
```

### 2.4 HRA injection enumeration

The implementation injects **18 HRA adapter pairs** when `use_hra=True` —
one per *effective layer* (3 blocks × 6 loops = 18), applied on the
attention sub-block (`model.py::_attention`: `x_in @ adapter_a @ adapter_b`).
This is NOT per-projection (an early draft envisioned 132 across
Q/K/V/O + every expert + router); it is one parallel rank-256 path
per block forward. The canonical preset pretrains with HRA **off**
(E1, roadmap §18.1), so against `OSRT_V7` as shipped the count is 0.
Verify both:

```bash
PYTHONPATH=src python -c "from osrt.model import OSRTForCausalLM; \
from osrt.config import OSRTConfig; from osrt.presets import OSRT_V7; \
m = OSRTForCausalLM(OSRTConfig(**{**OSRT_V7, 'use_hra': True})); \
print(sum(p.numel() for n,p in m.named_parameters() if 'adapter' in n))"
# -> 14155776   (0 with the preset as shipped)
```

Total HRA params when on: 18 × (2 × 1,536 × 256) = 18 × 786,432 =
**14,155,776**. All fully active per token — the adapters sit on the
always-run attention path, not on the sparse routed experts, so there is
no top-k masking of HRA.

### 2.5 The name

`OSRT` = **Optimized Sparse Recursive Transformer**:
- **O**ptimized — Muon optimizer + AlphaQ + TurboQuant deployment stack
- **S**parse — MoE (top-4 of 28 routed + 1 shared per block)
- **R**ecursive — 3 physical blocks × 6 loops = 18 effective layers
- **T**ransformer — standard pre-norm decoder backbone

No parameter count appears in the repo, package, preset or directory
name, by rule (`CLAUDE.md`; §19). The v6 lineage carried four mutually
inconsistent stale counts at once — `OSRT-605M-A269M` (repo),
`nano-osrt-100m` (checkout), `OSRT_605M_A288M` (preset) and "~608M/~279M"
(pyproject) — against an actual 601M/278M. `scripts/compute_budget.py` is
the only source for a count: v7 instantiates at **968,468,355 physical /
263,035,779 active** (27.2%, §2.1). The old names survive only as history
in `CodeHalwell/OSRT-605M-A269M`.

---

## 3. Tokenizer specification

### 3.1 BPE configuration

- **Algorithm**: byte-level BPE (sentencepiece or HuggingFace
  tokenizers)
- **Vocab size**: 49,280 (49,184 real)
- **Encoding focus**: English + 6 multilingual (Arabic, Japanese,
  Korean, Spanish, French, German) + code (Python, JS, Rust, C++)
- **Pre-tokenization**: GPT-2 style regex (handles contractions,
  numbers, punctuation)

### 3.2 Special tokens (v7 contract — all 32 on disk)

The v7 tokenizer keeps SmolLM2's 49,152 base rows byte-for-byte (ids 0–49,151,
including SmolLM2's own `<|endoftext|>`/`<|im_start|>`/`<repo_name>`... control
tokens, which the pretraining stream never emits as control ids — raw text is
encoded with `split_special_tokens=True`) and appends the 32 OSRT specials at
49,152–49,183. `osrt.tokenizer_contract.validate_tokenizer_contract` pins the
size and these ids before any model is built; `osrt.presets.OSRT_V7` carries
the same ids into the model config (the v6 ids 0–13 that `OSRTConfig` still
defaults to are SmolLM2 control tokens in this tokenizer).

| token | id | role |
|---|---|---|
| `<|begin_of_text|>` | 49152 | BOS |
| `<|end_of_text|>` | 49153 | EOS (document separator in the packed stream) |
| `<|padding|>` | 49154 | PAD |
| `<|unknown|>` | 49155 | unk |
| `<|fim_prefix|>` / `<|fim_middle|>` / `<|fim_suffix|>` | 49156–49158 | FIM markers |
| `<|think|>` / `<|/think|>` | 49159 / 49160 | reasoning block |
| `<|answer|>` / `<|/answer|>` | 49161 / 49162 | answer block |
| `<|user|>` / `<|assistant|>` / `<|system|>` | 49163 / 49164 / 49165 | turn openers |
| `<|end_turn|>` | 49166 | end of an assistant turn |
| `<|tool_call|>` / `<|/tool_call|>` | 49167 / 49168 | tool invocation |
| `<|tool_result|>` / `<|/tool_result|>` | 49169 / 49170 | tool result |
| `<|image|>` / `<|audio|>` | 49171 / 49172 | reserved (vision / audio retrofit) |
| `<|reserved_21|>` … `<|reserved_31|>` | 49173–49183 | free slots |

Embedding rows 49,184–49,279 are padding to a multiple of 128; every loss and
every generation path slices logits to the real 49,184, so they are never
targets and never sampled.

### 3.3 Chat template

The canonical form has **no newlines between markers**. Every marker is a
single token, so a newline there would be an extra token the model has to
learn to emit at every boundary, and `<|end_turn|>` detection would depend
on whitespace. The pretraining stream (`osrt.data`, via
`osrt.chat_format.render_chat`) and the tokenizer's `chat_template`
(`osrt.chat_format.CHAT_TEMPLATE`, shipped in `tokenizer/tokenizer_config.json`)
produce this exact byte form; content is whitespace-trimmed. Each block
below is one contiguous string.

```
<|system|>{system_message}<|user|>{user_question}<|assistant|><|think|>{reasoning}<|/think|><|answer|>{final_answer}<|/answer|><|end_turn|>
```

Multi-turn (the system prefix is optional; a trailing user turn without a
reply is dropped in training, and a generation prompt ends with `<|assistant|>`):
```
<|system|>{system}<|user|>{q1}<|assistant|>{a1}<|end_turn|><|user|>{q2}<|assistant|>{a2}<|end_turn|>
```

Tool use (the tool markers are reserved for post-training; `tool` roles are
not rendered in pretraining):
```
<|user|>{question_needing_calc}<|assistant|><|think|>I need to compute 17 × 23.<|/think|><|tool_call|>calculator("17 * 23")<|/tool_call|><|tool_result|>391<|/tool_result|><|answer|>The answer is 391.<|/answer|><|end_turn|>
```

---

## 4. Embedding layer

### 4.1 Shape and tying

- `embedding_matrix ∈ ℝ^(49280 × 1536)`
- **Tied with LM head**: `lm_head.weight = embedding.weight`
- Total params: 100,663,296 (16.9% of model)

### 4.2 Initialization

- Truncated normal, std = 1 / √(1536) ≈ 0.0255
- LM head logits scale: divide by √(1536) at output for μP
  compatibility

### 4.3 Optimizer routing

- **AdamW** (not Muon — embedding is special, see §11.2)
- No weight decay on embedding (preserve representation norms per
  SmolLM3 convention)

---

## 5. Recursive transformer block

### 5.1 Structure

```
For each loop r ∈ {0, 1, 2, 3, 4, 5}:
    For each physical block b ∈ {0, 1, 2}:

        # Add loop conditioning (broken symmetry per-iteration)
        x = x + loop_emb[min(r, 7)]    # if b == 0 (start of loop)

        # mHC pre-block mixing (replaces standard residual)
        residual = x
        x_normed = RMSNorm_pre[b](x)

        # Attention sub-block
        x_attn = AttentionBlock[b](x_normed, cache=kv_cache[b])

        # mHC post-attention residual mixing
        x = mHC_mix(residual, x_attn, b)

        # mHC pre-FFN mixing
        residual_ffn = x
        x_normed = RMSNorm_post[b](x)

        # MoE FFN sub-block
        x_ffn = MoEBlock[b](x_normed)

        # mHC post-FFN residual mixing
        x = mHC_mix(residual_ffn, x_ffn, b)
```

### 5.2 Loop embeddings

```
loop_emb ∈ ℝ^(6 × 1536)
```

Added BEFORE the first physical block at each loop. This is the
**only parameter that differs across loop iterations** — the bias
that tells the model "you're on iteration r of 6."

Capping at `min(r, 7)` means hard wall at R=8. Model trained for R=6
will function (with quality degradation) at R=3-5; cannot safely
extend beyond R=6 without retraining loop embeddings.

### 5.3 Normalisation ("sandwich" in older text)

What the code does (`RecursiveBlock`, `OSRTModel`):
- `norm_attn` — RMSNorm on the block input before attention (pre-norm)
- `norm_moe` — RMSNorm on the residual before the MoE FFN (pre-norm)
- `norm_q` / `norm_k` — per-head QK-norm before RoPE
- `norm_loop` — RMSNorm reset of the residual stream between recursive loops
- `norm_out` — RMSNorm before the (tied) LM head and the MTP heads

Each stream norm is `nn.RMSNorm(1536)` (torch's default eps, i.e. dtype
epsilon) with learnable scale, no bias. **There is no post-sub-block norm**:
earlier revisions called this stack "sandwich RMSNorm" and cited Gemma 3,
whose sandwich is pre- *and* post-norm around each sub-block. What v6 proved
stable across 18 effective layers is pre-norm plus the per-loop reset; a true
post-norm variant would be a ladder arm, not a documented feature.

Gemma 3's "sandwich" placement validated for deep stacks; Huginn used
similar to survive 32+ recursive iterations.

### 5.4 HRA injection

87 HRA injection points across the model. At each point:
```
adapter_a ∈ ℝ^(1536 × 256)
adapter_b ∈ ℝ^(256 × 1536)
HRA_output(x) = x + adapter_b(adapter_a(x))    # low-rank residual
```

Injected into: Q/K/V projections, attention output, gate/up/down of
each expert, router projection. Trainable in all stages, especially
during RL (HRA-only training in GRPO stage).

---

## 6. Attention sub-block

### 6.1 GQA configuration

- **Query heads**: 24
- **Key/Value heads**: 8 (groups of 3 queries share a KV head)
- **Head dimension**: 64
- **Total Q dim**: 24 × 64 = 1,536
- **Total K dim**: 8 × 64 = 512
- **Total V dim**: 8 × 64 = 512

### 6.2 Projections

```
W_Q ∈ ℝ^(1536 × 1536)        # 2.36M params
W_K_DOWN ∈ ℝ^(1536 × 512)    # 0.79M params — to latent K
W_V_FROM_K ∈ ℝ^(512 × 512)   # 0.26M params — KDV: derive V from K
b_V ∈ ℝ^(512)                # bias for V derivation
W_O ∈ ℝ^(1536 × 1536)        # 2.36M params
```

Per block: ~5.76M params; across 3 blocks: ~17.3M. Plus HRA adapters.

> ✅ **DECISION MADE — KDV (Key-Derived Value).** Of the three options once on
> the table (a: cache one latent, derive V from it; b: widen latent then
> split; c: full DeepSeek MLA with decoupled-RoPE + matrix absorption), the
> implementation chose **(a)** — `model.py` caches a single 512-dim un-rotated
> latent (`kv_down`), reads K straight off it (identity reshape), and derives V
> via `v_from_k` (a learned `Linear(512→512)+bias`). The name for that contract
> — *Value derived from the Key latent* — is **KDV (Key-Derived Value)**.
> (`review/SYNTHESIS.md` Tier 1 #7.)
>
> **The justification is memory bandwidth, not expressivity.** Autoregressive
> decode is HBM-bandwidth-bound: each step streams the entire KV cache out of
> memory and does ~O(1) FLOP per byte loaded — far below an H100's ~300+
> FLOP/byte roofline ridge — so the tensor cores sit idle behind the load.
> Decode throughput therefore scales as 1 / (cache bytes per token per layer).
> KDV caches **512 scalars/token/layer** vs **1024** for a GQA K+V cache (≈2×
> fewer bytes ⇒ ≈2× the attention-bound decode throughput), and the `v_from_k`
> recompute is FLOPs that hide *for free* under the memory latency already being
> paid. The design deliberately spends idle compute to avoid HBM traffic —
> **"recompute, don't reload."** (Per-layer is a constant 2×; the larger
> bytes-moved levers for the recursive stack — cross-loop KV reuse and
> sequence-axis compression — are catalogued in
> `docs/specs/2026-06-16-cross-loop-kv-reuse.md`.)
>
> **KDV vs MLA on the metric that matters (cache bytes).** MLA-V2 caches its
> compressed latent **plus** a separate decoupled-RoPE channel
> (`d_c + d_h^R ≈ 512 + 64 = ~576` scalars/token/layer) precisely so the K
> up-projection can be *absorbed* into Q at inference — an absorption that saves
> decode *compute*. KDV forgoes absorption (it RoPEs the reshaped latent
> directly and recomputes K/V each step), costing only the idle FLOPs we don't
> care about, and in exchange caches **fewer bytes** (512 < ~576) with no
> decoupled channel to carry. On the bandwidth axis KDV is therefore marginally
> *leaner* than MLA, not a degraded version of it.
>
> **Implemented correctly: KDV operates on the UN-rotated latent.** RoPE is
> position-dependent, so the cache holds the un-rotated `c_kv`; both K (RoPE'd)
> and V (`v_from_k(c_kv)`) are recomputed from it at attention time. See
> `RecursiveBlock._attention` in `src/osrt/model.py` (the `c_kv_new = kv_down(h)`
> / `v_from_k(c_kv)` block).

### 6.3 V derived from K (Key-Derived Value / KDV, MLA-inspired)

```
K = W_K_DOWN @ x_normed          # [batch, seq, 512]
V = W_V_FROM_K @ K + b_V         # [batch, seq, 512] — KDV: derived from K
```

**Cache only the latent** (not K and V separately) to halve KV-cache bytes;
V is recomputed at decode via the learnable transform — the **Key-Derived
Value (KDV)** contract: at every token, V is a fixed learned affine function
of that token's cached key latent.

**Expressivity — the honest accounting.** It is *not* accurate to say KDV
"loses no expressivity." Split it into the two sides:

- **V side: free.** `v_from_k` is a full `512→512` map, so V mixes across the
  whole latent exactly as MLA's `W_UV` does — and that map is anyway
  *absorbable* into `out_proj` (`W_O · Σ_j a_j (W c_j) = (W_O W) · Σ_j a_j c_j`),
  so it costs no representational power beyond a per-block bias term. Nothing
  is lost here.
- **K side: a mild, accepted restriction.** K is the *identity* reshape of the
  latent, whereas MLA's `K = W_UK · c` is a learned projection of the *full*
  latent. Each KDV key head therefore sees only its own 64-dim slice, while an
  MLA key head sees a learned 64-dim view of all 512 dims. Formally KDV's
  attention-score function class is a **subset** of MLA's (MLA reproduces KDV
  by setting `W_UK` block-identity; KDV cannot reproduce a cross-slice MLA
  key). This was accepted on purpose: it is exactly what lets the cache hold
  the raw latent with RoPE folded in (no decoupled channel, fewer bytes — §6.2),
  and the lost cross-slice key mixing is judged marginal at this scale.
  Revisit only if attention quality stalls.

This is the same *family* as DeepSeek MLA's shared `c_KV` (one cached latent,
K and V both linear in it), tuned toward minimal cache bytes rather than
inference-time absorption — see the bandwidth argument in §6.2.

### 6.4 QK-Norm

Apply RMSNorm to each Q and K head independently before scaled dot-
product:
```
Q_head = RMSNorm(Q.view(batch, seq, 24, 64), dim=-1)
K_head = RMSNorm(K.view(batch, seq, 8, 64), dim=-1)
```

Prevents attention-logit explosion (Muon-trained models are
particularly prone; Kimi K2 added "QK-Clip" on top — we use just
QK-Norm and rely on Muon stability).

### 6.5 Partial RoPE

Apply RoPE to the **last 64 dimensions only** of Q and K head vectors:
- First 0 dims: position-free (content-only matching)
- Last 64 dims: rotary-encoded

Base θ = 10,000 (standard). Will be scaled via YaRN-style for context
extension in mid-training.

### 6.6 Attention sink — REMOVED (kept behind `attention_sink=False`)

> 🔧 **DROPPED for OOM at long context.** The attention sink was a
> learnable per-head sink logit added to the softmax DENOMINATOR only:
> ```
> sink_logits ∈ ℝ^(24)     # per RecursiveBlock; 3 × 24 = 72 params total
> s_{h,i,j} = exp(z_{h,i,j}) / (Σ_k exp(z_{h,i,k}) + exp(sink_logits[h]))
> ```
> letting a head's weights sum to < 1 (attend to "nothing" when no key
> is relevant). It is **off in the canonical preset**
> (`presets.py: attention_sink=False`) and the standard GQA path runs
> through `F.scaled_dot_product_attention` (flash) instead.
>
> **Why it was dropped:** SDPA cannot express the sink term, so
> `attention_sink=True` falls back to the manual `_attention_with_sink`
> path (`model.py`), which materialises the full `(B, H, S, total_len)`
> score matrix to compute the per-query log-sum-exp for the sink
> rescale. At the seq-8192 instruction phase that score matrix is
> recomputed inside the gradient-checkpointed backward (~12GB at batch
> 2) and the run measured OOM (>85GB on an 80GB H100). Flash never
> builds the score matrix → the **same** seq-8192/batch-2 config fits at
> ~35.9GB. The sink had no demonstrated benefit (it was kept only
> because it happened to fit at seq 2048), so it was removed in favour
> of v5's proven flash path which scales to every phase.
>
> **Code state:** the `attention_sink` config flag, the
> `_attention_with_sink` method, and the `sink_logits` parameter all
> still exist as a clean A/B knob. With the flag False (the canonical
> setting) the `sink_logits` `nn.Parameter` is **never instantiated**
> (`if config.attention_sink:` in `RecursiveBlock.__init__`), so the
> model is 72 params lighter (§2.1). See §6.7.

### 6.7 Scaled dot-product attention (flash GQA)

The canonical path is standard flash SDPA — no sink:
```
attn_output = F.scaled_dot_product_attention(Q, K, V,
                  is_causal=(S > 1), enable_gqa=(group_size > 1))
# (cached-decode with S>1 builds an explicit -inf causal mask shifted
#  by past_len instead of is_causal — model.py::_attention)
```

`enable_gqa=True` lets SDPA broadcast the 8 KV heads across the 24
query heads (8 groups of 3) without materialising repeated heads, and
flash never
builds the `(B, H, S, total_len)` score matrix — the property that
keeps seq-8192 in memory (§6.6).

When (and only when) `attention_sink=True`, the manual
`_attention_with_sink` path is used instead: it materialises the score
matrix, applies the same causal mask the SDPA path uses, and rescales
each head's output by `sigmoid(lse − sink[h])` — the exact log-sum-exp
equivalent of adding `exp(sink[h])` to the denominator. That path is
OFF in the canonical preset (§6.6).

### 6.8 Output projection

```
attn_output_concat = attn_output.view(batch, seq, 1536)
attn_block_output = W_O @ attn_output_concat
```

HRA adapter applied to `W_O` output additively.

---

## 7. MoE sub-block

### 7.1 Structure

Each MoE block has:
- 1 always-active shared expert (h=3,840; was 2,816 before E1 moved HRA's budget here)
- **28 routed experts** (h=2,112), top-4 active per token
- 1 router (linear projection + sqrt-softplus affinity + the Quantile Balancing bias)

> **v7:** 28 routed × h2,112, top-4 = 14.3% density, Quantile Balancing,
> SiTU-GLU. The re-grain holds active params while adding total; see
> `docs/03-moe-and-routing.md` §12 and roadmap §14.3. The paragraph below is
> the **v6** rationale, kept because it explains why 8 was chosen then.

v6 ran 8 routed (not the 12 of an early draft): top-2 of 8 = 25% routing
density vs 16.7% for top-2 of 12 — denser routing, more capacity per
token, less expert under-utilization at v6's 601M scale, each of the 8
wider (h=3,840) to absorb the capacity. v7 trades that for a finer grain:
more, narrower experts at a lower density (roadmap §14.3, gate G3).

### 7.2 Shared expert (SiTU-GLU)

```
w_gate ∈ ℝ^(1536 × 3840)       # 5.90M params
w_up ∈ ℝ^(1536 × 3840)         # 5.90M params
w_down ∈ ℝ^(3840 × 1536)       # 5.90M params

shared_output(x) = w_down @ (act(w_gate @ x) ⊙ (w_up @ x))
# act = the SiTU-GLU activation (v7 default); SiLU in the SwiGLU A/B arm
```

Per shared expert: 17,694,720 params. Across 3 blocks: 53,084,160.
(h=3,840 is where E1 reinvested HRA's 14,155,776 pretraining params —
roadmap §18.1; v6 used h=2,816.)

### 7.3 Routed experts (SiTU-GLU)

Per routed expert:
```
w_gate ∈ ℝ^(1536 × 2112)       # 3.24M params
w_up ∈ ℝ^(1536 × 2112)         # 3.24M params
w_down ∈ ℝ^(2112 × 1536)       # 3.24M params
```

Per expert: 9,732,096. Per block (28 experts): 272,498,688. Across 3
blocks: 817,496,064 — the dominant param term, 84.4% of physical. Top-4
routing keeps 4/28 = 14.3% of them (116,785,152) active per token.

### 7.4 Router

```
W_route ∈ ℝ^(1536 × 28)        # 43,008 params per block
b_route_bias ∈ ℝ^(28)          # per-expert bias for load balancing
                                # (not in gradient; nudged by load deviation)
```

Affinity score:
```
affinity = sqrt(softplus(W_route @ x))      # sqrt(softplus) — DeepSeek-V4; fp32
balanced_affinity = affinity + b_route_bias  # controller bias, selection only
top_k_indices = topk(balanced_affinity, k=4)

gates = affinity[top_k_indices] / sum(affinity[top_k_indices])
# DeepSeek-V3 §2.1.2: the bias picks WHICH experts; the gate that multiplies
# each expert's output comes from the ORIGINAL affinity. This is what
# `router_bias_in_gates=False` (the default) implements; the v6 behaviour
# (gates from the biased, Gumbel-noised distribution) is the True setting.
# The router matmul runs in fp32 regardless of autocast (Switch §2.4).
```

### 7.5 Hash routing for blocks 0 and 1

For physical blocks 0 and 1 (first 2 of 3), routing is HASH-based,
not learned:
```
expert_id = hash(token_id) mod 28    # mod num_routed_experts
# Always select this fixed expert, no learned router
```

Stabilizes early training (prevents collapse before router learns).
Block 2 uses normal learned routing.

> ✅ **DECISION MADE — loop-indexed top-1 hash, off by default.**
> Implemented in `model.py` as `expert_id = (token_id + loop_idx) %
> num_routed_experts` (loop-indexed → depth specialization, top-1).
> The number of early blocks that hash-route is `config.hash_routing_
> blocks` (default **0 = off**); it is a clean A/B knob, not on in the
> canonical preset. So in the trained config every block uses the
> learned router; hash routing is available for stability experiments.
> (Resolved `review/SYNTHESIS.md` Tier 1 #6: Q1 top-1, Q2 loop-indexed,
> Q3 hard binary at `hash_routing_blocks`.)

### 7.6 Aux-loss-free load balancing

The per-expert bias `b_route_bias[e]` steers expert **selection** only:
gates are computed from the pre-bias affinity (`router_bias_in_gates=False`,
DeepSeek-V3 semantics), and the bias is not in the gradient. Two
controllers exist (`router_balance_mode`); the canonical preset uses
**Quantile Balancing**, which roadmap §14.6 makes required, not optional:

```
# "quantile" (v7 preset; Kimi K3) — re-solved from accumulated router scores
p = top_k / num_routed                     # 4/28: the target selection fraction
for e in experts:
    t[e] = the score threshold with fraction p of expert e's OWN score mass above it
b_route_bias = mean(t) - t                 # centred (only differences steer top-k)
clamp(±router_balance_bias_max)
# every expert then presents the same fraction of its distribution above the
# common selection threshold, so load equalises in one shot — no rate to tune

# "heuristic" (config default; the legacy controller, tuned at E=8)
frac = EMA(clean load fraction per expert)         # router_balance_bias_ema_rate
b_route_bias -= router_balance_bias_update_rate × (frac - 1/num_routed)
clamp(±router_balance_bias_max)
```

Both accumulate the per-loop *clean* load (before any capacity drop) and
apply once per optimizer step (`MoELayer.apply_balance_update`). The
heuristic step was tuned at 8 experts; at 28 it has to move 3.5× as many
biases on 3.5× less load signal each, which is why the preset does not use
it (`config.py` warns when it is combined with more than 8 experts).
Combined with a small sequence-balance loss
(`router_seq_balance_loss_coeff` = 1e-4 in the preset) to prevent extreme
imbalance within single sequences.

### 7.7 MoE output

```
moe_output(x) = shared_output(x) + Σ_{i ∈ top4} weight_i × routed_output_i(x)
```

> ✅ **DISPATCH: grouped-GEMM (B4), loop retained as fallback.** Two
> dispatch implementations compute the routed sum, selected by
> `config.moe_grouped_gemm` (canonical preset: **True**; config default:
> False). Both produce identical weights, so checkpoints load under
> either path.
>
> - **`_dispatch_loop` (fallback):** the original per-expert
>   `(assign == ei).nonzero()` gather → run expert → `index_add` scatter,
>   with the capacity cap. Correct, but the data-dependent `.nonzero()`
>   is **the only `torch.compile` graph break in the model**
>   (`model.py`/`config.py` comments), so the model can't compile
>   fullgraph with it.
> - **`_dispatch_grouped` (B4, canonical):** flatten the (token, rank)
>   pairs, `argsort` by chosen expert, `bincount`→`cumsum` to per-expert
>   END offsets, one grouped SwiGLU over the sorted tokens
>   (`_grouped_ffn`), gate, then `index_add` scatter back per token. It
>   is **dropless** (no capacity cap — keeps every token in training) and
>   uses only fixed-shape ops (`argsort`/`bincount`/`cumsum`/`index_add`),
>   removing the lone graph break so the model compiles **fullgraph**.
>   The grouped matmul is `torch._grouped_mm` on CUDA (fused) and a
>   `_ref_grouped_mm` loop-of-matmuls reference on CPU (the kernel's CPU
>   backward is broken). Measured ~9-12% faster steady-state on H100
>   (gated by the gradient-checkpointing recompute), loss tracking the
>   loop path.

### 7.8 SwiGLU Clamping (stability)

Inside every SwiGLU (shared and routed):
```
gate_pre = w_gate @ x
up_pre = w_up @ x

# Apply DeepSeek-V4 stability clamps
gate_clamped = torch.clamp(gate_pre, max=10.0)         # cap upper
linear_clamped = torch.clamp(up_pre, min=-10.0, max=10.0)  # clamp both

output = w_down @ (SiLU(gate_clamped) ⊙ linear_clamped)
```

---

## 8. Manifold-Constrained Hyper-Connections (mHC) — REMOVED IN v7

> Section number retained deliberately: `src/osrt/` comments cite
> `ARCHITECTURE.md §N` throughout, so renumbering §9–§18 would invalidate
> them. The content is gone; the anchor stays.

mHC was removed in v7. See `docs/specs/2026-08-11-v7-roadmap.md` §12.3 for the
decision and §12.1-C2 for the evidence: the mHC paper's headline results are
**mHC-versus-HC**, not mHC-versus-a-plain-residual-stream, so the comparison
v7 actually needed was never published. Against that unmeasured benefit sat a
measured, threefold cost — 36 Sinkhorn projections per forward, a residual
stream 4x wider in activation memory at every one of 18 effective layers, and
an unresolved NaN/gradient-amplification warning that shipped enabled in the
v6 preset. Under the v7 requirement of fastest achievable inference (§13.1)
the decision could not have gone the other way, so no ablation was bought.

v5 ran Muon over 18 effective layers on a plain residual stream without loop
collapse, which is the closest thing to a control that exists.

## 9. LM head and auxiliary heads

### 9.1 Main LM head

Tied with embedding:
```
logits = embedding.weight @ x_final.transpose(-1, -2)
# Shape: [batch, seq, 49280] → sliced to 49184 real tokens before the loss
```

Final hidden state `x_final` comes from the LAST physical block of
the LAST loop iteration.

### 9.2 Auxiliary per-loop LM heads (architecture-fix knob)

The **same** LM head (tied with embedding) is applied to intermediate
loop outputs:
```
for r in range(6):
    if aux_loop_loss_weight > 0:
        x_loop_r = output of physical block 2 at loop r
        logits_loop_r = embedding.weight @ x_loop_r.transpose(-1, -2)
        # Use these for auxiliary cross-entropy losses
```

**Key insight:** the LM head is SHARED across all loop outputs (it
IS the embedding). No additional parameters. The per-loop training
signal alone makes intermediate loops produce coherent predictions.

This is what enables:
1. Architecture-fix: loops 1-5 actually contribute to predictions
2. Speculative decoding at inference: loop-3 output is a draft
   prediction, loop-6 verifies (~60-75% accept rate expected)

### 9.3 MTP (multi-token prediction) heads

Per DeepSeek-V3 / V4: predict tokens at offsets +1, +2 via separate
small heads on the FINAL loop output:
```
MTP_head_1 ∈ ℝ^(1536 × 49280)     # tied with embedding
MTP_head_2 ∈ ℝ^(1536 × 49280)     # tied with embedding
```

(Heads are tied with embedding too — no separate params, just
multiple uses of the LM head with different small projection layers
in between if needed.)

Used during training for the MTP loss; helps the main model learn
longer-range structure.

---

## 10. Forward pass walkthrough

> ✅ **PSEUDOCODE IS ILLUSTRATIVE; the implementation in `model.py` /
> `mhc.py` is the source of truth.** The three bugs an early draft of
> this pseudocode contained are all **fixed in the real code** (see
> `review/SYNTHESIS.md` Tier 0 #3 and `review/code-review.md`):
>
> **Bug 1 — `expand()` aliasing in mHC init:** FIXED. The real code
> uses `.repeat(...)` (not `.expand()`), so the per-channel loop-bias
> write doesn't alias the other channels. (`tests/test_mhc.py`.)
>
> **Bug 2 — mHC mixing shape arithmetic:** FIXED. `mhc.py` uses
> explicit `torch.einsum` for the input view and residual update
> (e.g. `torch.einsum("bsc,bscd->bsd", a, X)`), so shapes are correct
> by construction.
>
> **Bug 3 — final collapse:** FIXED. There is a dedicated learnable
> collapse head `mhc_collapse` (a length-`n_hc` parameter initialised
> uniform to `1/n_hc`), not a reused dynamic `A_l`. Final hidden =
> `einsum("c,bscd->bsd", mhc_collapse, X)`.
>
> The pseudocode below reads the OLD buggy form in places; treat it as
> a conceptual sketch and defer to `model.py`/`mhc.py` for exact ops.

Detailed pseudocode for one forward pass on a batch of `B` sequences
of length `L`:

```python
def forward(input_ids, kv_cache=None, training=False):
    # Step 1: Embedding lookup
    x = embedding(input_ids)              # [B, L, 1536]

    # Step 2: Initialize mHC residual stream (4× width)
    X = x.unsqueeze(-2).expand(-1, -1, 4, -1)    # [B, L, 4, 1536]

    # Step 3: Recursive loop
    per_loop_outputs = []
    for r in range(6):
        # Add loop bias to channel 0 (or to all channels — design choice)
        loop_bias = loop_emb[min(r, 7)]
        X[:, :, 0, :] = X[:, :, 0, :] + loop_bias

        for b in range(3):
            # mHC pre-attention
            X = X.reshape(B, L, 6144)
            A_l, B_l, C_l = generate_mHC_params(X, layer_id=(r, b, 'attn'))
            x_view = (A_l @ X.reshape(B, L, 4, 1536).transpose(2, 3)).squeeze(-1)

            # Pre-norm
            x_normed = RMSNorm_pre[b](x_view)

            # Attention sub-block
            x_attn = AttentionBlock[b](x_normed, kv_cache=kv_cache[r, b])

            # mHC update of residual
            X = (B_l @ X.reshape(B, L, 4, 1536).transpose(2, 3)).squeeze(-1) \
                + (C_l @ x_attn.unsqueeze(-2)).squeeze(-2)

            # mHC pre-FFN
            A_l_ffn, B_l_ffn, C_l_ffn = generate_mHC_params(X, layer_id=(r, b, 'ffn'))
            x_view = (A_l_ffn @ X.transpose(-1, -2)).squeeze(-1)

            # Post-norm
            x_normed = RMSNorm_post[b](x_view)

            # MoE sub-block
            if b < 2:
                x_moe = MoEBlock[b](x_normed, routing='hash')
            else:
                x_moe = MoEBlock[b](x_normed, routing='learned')

            # mHC update
            X = (B_l_ffn @ X) + (C_l_ffn @ x_moe.unsqueeze(-2)).squeeze(-2)

        # End of loop r — capture output for aux LM head
        if training and aux_loop_loss_weight > 0:
            x_end_of_loop_r = (A_l @ X.transpose(-1, -2)).squeeze(-1)
            per_loop_outputs.append(x_end_of_loop_r)

    # Step 4: Final output — extract from mHC residual stream
    x_final = (A_l @ X.transpose(-1, -2)).squeeze(-1)    # [B, L, 1536]

    # Step 5: LM head (tied with embedding)
    logits = x_final @ embedding.weight.T               # [B, L, 49280]

    return {
        'logits': logits,
        'per_loop_outputs': per_loop_outputs,           # for aux losses
        'kv_cache': kv_cache,
    }
```

The pseudocode is illustrative; the actual implementation should:
- Use efficient batched matrix ops
- Fuse RMSNorm + Linear where possible
- Use Flash Attention or similar for the attention block
- Cache mHC parameter generations across the recursion (they only
  depend on the residual state, which evolves)

---

## 11. Training losses

### 11.1 Main loss

Standard next-token cross-entropy on the final logits:
```
L_main = CrossEntropy(logits, targets, ignore_index=-100)
```

Label masking: -100 on the prefix (system + user prefix during SFT),
real token IDs on the assistant target.

### 11.2 Aux per-loop LM-head loss

For each intermediate loop output:
```
L_aux_loop_r = CrossEntropy(per_loop_logits_r, targets) × aux_loop_loss_weight
```

Total aux loss:
```
L_aux_total = Σ_{r=1}^{5} L_aux_loop_r
            # Note: loop 6 IS the main loss; we add r=1..5
```

`aux_loop_loss_weight = 0.05` during pretrain/MOPD/SFT.
`aux_loop_loss_weight = 0.03` during GRPO (preserve training but
don't dominate policy gradient).

### 11.3 MoE auxiliary balance loss (small)

Sequence-wise balance loss to prevent extreme intra-sequence
imbalance:
```
L_balance = α_balance × Σ_blocks Σ_experts f_i × p_i
# α_balance = 0.0001 (small; main balancing comes from b_route_bias)
```

### 11.4 MTP loss

```
L_MTP_1 = CrossEntropy(mtp_logits_1, targets_shifted_by_1) × β_mtp
L_MTP_2 = CrossEntropy(mtp_logits_2, targets_shifted_by_2) × β_mtp
# β_mtp = 0.3 most of training, decayed to 0.1 at LR decay
```

### 11.5 Router z-loss (insurance against logit blow-up)

```
L_z = mean(logsumexp(router_logits) ** 2) × γ_z
# γ_z = 0.001
```

### 11.6 Total training loss

```
L_total = L_main + L_aux_total + L_balance + L_MTP_1 + L_MTP_2 + L_z
```

### 11.7 Decoupled Top-K KD (during MOPD)

For knowledge distillation from teacher (LFM2 method):
```
L_DTK_per_token = KL(Bern(P_T(T)) || Bern(P_S(T)))                         # binary mass
                + P_T(T) × KL_τ(P_T(·|T) || P_S(·|T))                       # top-K conditional

# where T = teacher's top-K (K=32) token set, τ = temperature
# applied only to the conditional term
```

Replaces standard CE on teacher response during MOPD distillation.
Provides ~32× denser supervision per token.

---

## 12. Inference path

### 12.1 Generation modes

```python
def generate(
    input_ids,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.95,
    loops=6,                      # adjustable: 3-6
    eos_token_id=1,
    stop_token_ids=[10, 14, 18],  # </answer>, end_turn, /tool_result
):
    # Prefill phase: full forward pass over the prompt
    output = forward(input_ids, kv_cache=None)
    kv_cache = output['kv_cache']

    # Speculative decoding via loop-3 draft head (optional)
    if speculative_decoding_enabled:
        return generate_speculative(input_ids, kv_cache, loops)

    # Standard autoregressive decode
    generated = []
    for step in range(max_new_tokens):
        next_logits = output['logits'][:, -1, :]
        next_logits = next_logits / temperature
        # top-p sampling
        probs = top_p_filter(softmax(next_logits), p=top_p)
        next_token = sample(probs)

        if next_token in stop_token_ids or next_token == eos_token_id:
            break

        generated.append(next_token)
        output = forward(next_token, kv_cache=kv_cache, loops=loops)

    return generated
```

### 12.2 Variable loop count (controllable inference compute)

`generate(loops=K)` runs only K of the 6 trained loops. Trained for
6, but the aux per-loop LM head training makes loops 3-5 also
produce coherent outputs. Quality vs speed trade-off:

| loops | speed | quality |
|---|---|---|
| 3 | 2× faster than 6 | ~85% of full quality |
| 4 | 1.5× faster | ~93% of full quality |
| 5 | 1.2× faster | ~98% of full quality |
| 6 | baseline | full quality |

### 12.3 Speculative decoding via loop-3 draft

```python
def generate_speculative(input_ids, kv_cache, K_draft=4):
    """
    Draft K tokens using loop-3 output of main model.
    Verify all K with single forward pass at loop-6.
    Commit accepted prefix.
    """
    # Draft K tokens cheaply
    drafts = []
    for k in range(K_draft):
        x_loop_3 = forward_partial(loops=3, cache=kv_cache)
        logits = embedding.weight @ x_loop_3.T
        draft_token = greedy(logits)
        drafts.append(draft_token)

    # Verify with full forward
    full_logits = forward(drafts, loops=6, cache=kv_cache)
    full_predictions = greedy(full_logits)

    # Accept matching prefix
    accept_prefix = []
    for k in range(K_draft):
        if drafts[k] == full_predictions[k]:
            accept_prefix.append(drafts[k])
        else:
            # Reject from here; emit the verifier's prediction for position k
            accept_prefix.append(full_predictions[k])
            break

    return accept_prefix
```

Expected accept rate: 60-75% per draft (the loop-3 head is trained to
predict the same thing the loop-6 head predicts via the aux loss).
Net speedup: ~1.8-2.4× on generation.

> **v7 status.** The loop-3 draft above is an *autoregressive* drafter, the
> family DSpark's Table 1 shows losing to parallel drafters at every scale
> (roadmap §15.3). It stays available as `spec_drafter="loops"`. The default
> recommendation is §12.3b.

### 12.3b Speculative decoding via the MTP heads (`spec_drafter="mtp"`, 2026-09-02)

The two MTP heads (§9.3) are a parallel multi-position drafter that the
model already trained. `generate(speculative=True, spec_drafter="mtp")`
routes to `_generate_speculative_mtp`, which costs **one forward per round**
over `1 + mtp_heads` tokens and needs no draft-side cache:

```python
# round r: pending = last committed token (not yet cached); drafts d1..dK
# were proposed by the heads at the previous round's accepted position.
verify_input = [pending, d1, ..., dK]                 # K = mtp_heads
logits, hidden = forward(verify_input, cache, loops=6, output_hidden_states=True)
preds = greedy(logits)          # preds[i] = token after verify_input[:i+1]
accept = first i where drafts[i] != preds[i], else K
commit drafts[:accept] + [preds[accept]]              # 1 + accept tokens
cache = cache[: cache_len + accept + 1]               # stale tail sliced off
drafts = [greedy(embedding @ head_k(hidden[accept])) for k in 1..K]
#        ^ the heads at the accepted position predict the tokens that
#          FOLLOW the token just committed (offsets +2, +3 from there)
```

The first round's drafts come from a prefill over the whole prompt, whose
last hidden gives the first committed token (main head) and the heads'
proposals for the two after it. Per forward the loop commits `1 + a` tokens,
`a ∈ [0, K]`: the **worst case is exactly plain greedy** (one token per
forward) and the ceiling is `1 + K` = 3× at two heads. Output is
token-identical to greedy decoding (`tests/test_mtp_speculative.py`).
`model.last_spec_stats` records rounds, forwards, drafts offered/accepted,
acceptance rate and tokens per forward.

Why this matters more for OSRT than for a dense model of the same size: decode
is depth/launch bound (roadmap §13.2 roofline: ~136 tok/s measured against a
~3,400 tok/s bandwidth ceiling on v6), so each forward costs roughly the same
whatever it produces, and tokens-per-forward is the lever. The measured
acceptance rate on a real checkpoint is recorded in roadmap §15.7; a
DSpark-style post-hoc parallel drafter (§15.4) is the follow-on once the trunk
exists.

---

## 13. KV cache structure

### 13.1 Per-token cache contents

For OSRT-600M, **we cache only the K_DOWN (latent K) output**, not
full K or V:

```
cache_per_token_per_effective_layer = K_DOWN ∈ ℝ^512    # 8 KV heads × 64
```

V is recomputed at decode time via `V = W_V_FROM_K @ K + b_V`. This
halves the cache size vs caching both K and V.

### 13.2 Cache layout

```
kv_cache: dict
    keys: (loop_idx, block_idx) ∈ {0..5} × {0..2}
    values: tensor of shape [batch, seq, 512]

# Total cache entries: 18 effective layers × 512 floats per token
# Per token, BF16: 18 × 512 × 2 = 18,432 bytes = 18 KB
# At 4K context: 4096 × 18 KB = 72 MB raw
# At 8K context: 8192 × 18 KB = 144 MB raw
```

### 13.3 Compression stack at deployment

The §13.1 baseline already excludes V (K-only), so the KDV row
in earlier drafts was double-counting. Corrected table — apply
compression *once* against the K-only baseline:

| step | format | size at 4K |
|---|---|---|
| Standard GQA reference (K+V, BF16) — for comparison only | BF16 | 144 MB |
| **§13.1 baseline (K_DOWN only, BF16)** | BF16 | **72 MB** |
| + TurboQuant int4 on K_DOWN | int4 | 9-18 MB |
| + Sliding window (if applicable, 1K window over 4K context) | int4 + SW | 2-5 MB |

Final deployment cache: **~9-18 MB at 4K context** (TurboQuant only),
**~2-5 MB at 4K context** (with 1K sliding window). The previous
"< 5 MB" headline required *both* TurboQuant and sliding window;
state both assumptions when quoting it.

### 13.4 Cache update during decode

After each new token, append the **un-rotated** K_DOWN to all 18 caches.
RoPE is applied at attention time, NOT before caching (otherwise the
linear K→V relationship in KDV is broken — see §6.2 callout):

```
for r in range(6):
    for b in range(3):
        # Compute un-rotated latent K — DO NOT apply RoPE here
        new_K_down = W_K_DOWN[b] @ new_x_in_loop_r_block_b
        kv_cache[(r, b)] = concat(kv_cache[(r, b)], new_K_down, axis=seq)

# At attention time:
#   K_unrot = kv_cache[(r, b)]          # cached un-rotated K
#   V       = W_V_FROM_K[b] @ K_unrot + b_V[b]   # KDV: derive V from un-rotated K
#   K       = apply_rope(K_unrot, position_ids)  # then rotate K for QK math
#   Q       = apply_rope(W_Q[b] @ x_new, position_ids)
#   attn    = softmax(Q @ K.T / sqrt(d)) @ V
```

Each loop iteration computes FRESH K at that loop (the input differs
from loop r-1's output). No K sharing across loops — they're
genuinely different representations.

---

## 14. Quantization for deployment

### 14.1 Per-component quantization plan

| component | format | method |
|---|---|---|
| Embedding (tied) | int8 | symmetric per-channel QAT |
| Attention W_Q, W_K_DOWN, W_O | int8 | symmetric per-channel QAT |
| Shared experts | int8 | symmetric per-channel QAT |
| Routed experts | **FP4 (MXFP4)** | AlphaQ-allocated bit budget |
| HRA adapters | bf16 | kept full precision (small, sensitive) |
| Router projections | bf16 | kept full precision |
| Loop embeddings | bf16 | kept full precision |
| LayerNorms / biases | bf16 | always bf16 |
| K cache (per layer) | **int4** | TurboQuant random-rotation + per-block |

### 14.2 Memory budget at deployment

Numbers below are decimal MB (1 MB = 1,000,000 bytes), no allocator
overhead, no per-tensor quantization metadata. Param counts are the
real §2.1 figures (MTP heads dropped at deploy, HRA off):

```
Embedding (int8, 75.7M params × 1 byte):              76 MB
Attention (int8, 17.3M params):                       17 MB
Shared experts (int8, 53.1M params):                  53 MB
Routed experts (FP4 @ ~3.5 bit avg, 817.5M params):
    817.5M × 3.5 bits / 8 ≈ 358 MB (+~2% AlphaQ meta) ~365 MB
HRA adapter (post-training only; off in pretraining)    0 MB
router + norms + loop_emb (bf16, 0.16M params):        <1 MB
(MTP heads dropped at deploy)                            0 MB

TOTAL ON DISK (all loaded into RAM):                 ~510 MB
```

To fit a tighter envelope, the levers are:

- Routed experts to 2-bit average (~365 MB → ~210 MB) — they're 84%
  of physical, so this is the dominant lever
- Embedding to int4 (76 MB → ~38 MB)
- A post-training HRA adapter folded into the base weights (0 MB) or int8-ed

Stacked, those land near **~320 MB** of weights. The ~150–250 MB band
that v6's 424.7M routed pool reached is not reachable for v7 on weight
formats alone: it needs the "active-only resident" scheme (loading just
the top-4 routed experts per block and paging the rest from disk/CPU),
which is an inference-system choice, not a weight choice — state the
assumption when quoting it. With it, the stack is ~140 MB at 2-bit
routed and ~200 MB with the baseline formats. `docs/09` has the arithmetic.

### 14.3 AlphaQ bit allocation (routed experts)

Per AlphaQ:
- Compute PL Alpha Hill metric for each routed expert weight matrix
- ILP solver allocates bits ∈ {2, 3, 4} per layer per expert under
  global budget of 3.5 bits average
- Heavy-tailed experts (high importance) → 4 bits
- Light-tailed experts (less critical) → 2 bits
- Layer-wise allocation (each up/gate/down independently)

Expected quality: near-lossless at 3.5-bit average (per AlphaQ
results on Qwen1.5-MoE, a fine-grained many-expert regime; v7's 28
routed per block, top-4, is closer to it than v6's 8 were).

---

## 15. Total compute and memory math

### 15.1 Training compute and budget

Per token, per forward pass: ~2.4B FLOPs (§2.3)
Backward is ~2× forward: ~4.8B FLOPs
Total per token per training step: ~7.2 BFLOPs

**Base-pretrain budget (`train_config.py::PretrainConfig`):** one
continuous WSD schedule, with the phase boundaries derived from
`total_steps` so the two cannot disagree:
```
total_steps  = 18,000         # ≈5.43B tokens; WSD decays over the last 15% (wsd_decay_frac)
warmup_steps = 400            # ~2% — spins up Muon + the balance bias
peak_lr      = 6e-4  →  min_lr = 6e-5    # AdamW groups; Muon 3e-3 → 3e-4 on the same shape
```
Phases: foundation seq-2048 to step 900 (131K tok/step, 0.12B tokens),
knowledge seq-4096 to 15,300 (270K tok/step, 3.89B), instruction
seq-8192 to 18,000 (524K tok/step, 1.42B) — **≈5.43B tokens**, ~3.9e19
training FLOPs. Whether the token requirement tracks active or total
params is still open (gate G3a); wall-clock is measured, not estimated
— the smoke run and the notebook's probe step report tok/s on the actual
card (`RUNBOOK.md`). Post-training (SFT / RL) is being redesigned for v7
and is not part of this schedule (roadmap §14).

**Memory:** gradient (activation) checkpointing is on (the trainer flips
the private `OSRTModel._osrt_grad_ckpt` gate), the fused linear-CE
(`fused_cross_entropy_chunks` > 0, with `fused_main_head` covering the
main head) removes the (B, S, vocab) fp32 logit peak, and `torch.compile`
is on by default (`compile_enabled`). The v6 H100 figures that used to
sit here (seq-8192/batch-2 at ~35.9 GB, seq-4096/batch-6 at ~59 GB) were
measured on the 601M v6 model and do not transfer to v7's 968M on the
96 GB RTX PRO 6000; `train_main.py` scales the micro-batch to the GPU's
memory and the notebook's probe step measures the real footprint.

### 15.2 Inference compute per token (generation)

Just the forward pass: ~2.4B FLOPs (§2.3). Decode is bandwidth- and
latency-bound, not FLOP-bound: at bf16 a decode step reads the ~263M
active parameters (~0.53 GB); on a ~1.8 TB/s card that is a ~3,400 tok/s
single-stream ceiling against the ~136 tok/s recorded in
`docs/09-quantization-deployment.md`. Sequential depth (18 effective
layers) and launch count bind first, which is why the loop-count
recommender and a persistent megakernel outrank weight quantization on
the decode lever list. On CPU (Snapdragon 8 Elite class, int8) expect
tens of tokens/sec.

### 15.3 Memory at inference (full deployment)

Weights: ~510 MB as specified (§14.2); ~320 MB with the §14.2 levers;
~140–200 MB only with active-only expert residency
KV cache (4K context, int4 TurboQuant): ~9–18 MB (§13); ~2–5 MB with a 1K sliding window
Activations (transient): ~50 MB
Total: **~380 MB** with the levers and the full KV cache. The v6 "~250 MB,
fits comfortably on phones / Raspberry Pi 5" headline needs the residency
assumption for v7 — say which when quoting it.

---

## 16. Architectural invariants

These are the design properties that MUST hold for the architecture
to function as designed. Violating any of these is a bug.

### 16.1 Recursion correctness

- `loop_embeddings.shape[0] >= 6` — must have a bias for each loop
- Recursive forward MUST apply the SAME 3 physical blocks 6 times
  (not 18 different block instances)
- `aux_loop_loss_weight > 0` during training keeps the recursion
  meaningful; if 0, training MUST monitor for loop collapse

> ✅ **Collapse telemetry (implemented).** The model records, per
> effective layer (loop × block, 18 of them), the relative residual
> update `||Δx|| / ||x||` and the hidden norm `||x||`
> (`OSRTModel.last_loop_update_norm` / `last_loop_hidden_norm`,
> `model.py`). A deep loop whose update → 0 has collapsed to a no-op.
> The trainer (`train.py::_collect_moe_metrics`) emits these to W&B and
> stdout as `loop/update_norm_l{0..17}`, `loop/hidden_norm_l{0..17}`
> (plus `loop/update_norm_{mean,min,last}`), alongside a routed-expert
> health count `moe/dead_experts_total`. All of it is gated on
> `telemetry_enabled` (toggled per-step by `set_moe_telemetry`), so the
> `.item()` syncs never run on normal compiled steps — keeping the B4
> fullgraph clean (§7.7).

### 16.2 mHC stability

- `B_l` MUST satisfy `||B_l||_2 ≤ 1` at every step (doubly stochastic)
- Sinkhorn-Knopp MUST converge within `t_max=20` iterations
- `A_l, C_l` MUST be non-negative (sigmoid-bounded)

### 16.3 Routing correctness

- Per training step, `Σ_i b_route_bias[i]` MUST remain bounded
- Blocks with `block_idx < hash_routing_blocks` use hash routing; the
  canonical preset sets `hash_routing_blocks=0` so EVERY block uses the
  learned router (hash routing is an off-by-default A/B knob — §7.5)
- Aux-loss-free bias `b_route_bias` MUST NOT receive gradient
- `affinity = sqrt(softplus(W_route @ x))` — NEVER negative

### 16.4 Attention correctness

- QK-Norm MUST apply per-head, not flattened
- Partial RoPE applies to LAST 64 dims only
- V derivation MUST use `V = W_V_FROM_K @ K + b_V` (not from x)
- Canonical path MUST be flash SDPA (`attention_sink=False`); the sink
  is removed (§6.6). IF `attention_sink=True` is ever re-enabled, the
  sink logits MUST be added to the denominator, not the numerator

### 16.5 KV cache correctness

- Cache stores only K_DOWN (the latent), not V
- 18 separate cache entries per token (one per effective layer)
- Each loop's K is computed FRESH from that loop's input
- TurboQuant int4 applied to cached entries, not to the live forward
  pass

### 16.6 Tied LM head correctness

- `lm_head.weight = embedding.weight` (literal reference, not copy)
- Auxiliary per-loop heads use the SAME tied weight
- Logits computation: `x_final @ embedding.weight.T`

### 16.7 Gradient routing correctness

- Muon optimizer handles ALL 2D matrices in attention, experts, HRA,
  mHC W_pre/W_res/W_post
- AdamW handles: embedding, LM head (tied), RMSNorm gains, biases,
  router_bias accumulator
- Weight decay applied via decoupled scheme (Muon paper)
- `b_route_bias` NEVER in optimizer (heuristic update only)

---

## 17. Implementation notes (carried from v5 optimizations)

Five performance + security patterns proven in v5 that should bake
into v6 from the start. Full original commits preserved on the
`archived/v5-optimizations` branch.

### 17.1 Vectorized repetition penalty (`generate()`)

**Pattern:** never loop over generated token IDs in Python — it forces
CPU-GPU sync every step and hardcodes batch_size=1.

```python
# WRONG (v5 original — slow, batch-broken):
if repetition_penalty != 1.0:
    for token_id in set(generated[0].tolist()):
        if next_logits[0, token_id] > 0:
            next_logits[0, token_id] /= repetition_penalty
        else:
            next_logits[0, token_id] *= repetition_penalty

# RIGHT (vectorized, ~12-45× faster, batch-safe):
if repetition_penalty != 1.0:
    vocab_size = next_logits.shape[-1]
    mask = torch.zeros(
        (generated.shape[0], vocab_size),
        dtype=torch.bool, device=next_logits.device,
    )
    clamped = generated.clamp(max=vocab_size - 1)
    mask.scatter_(1, clamped, True)
    mask &= generated < vocab_size
    next_logits = torch.where(
        mask,
        torch.where(
            next_logits > 0,
            next_logits / repetition_penalty,
            next_logits * repetition_penalty,
        ),
        next_logits,
    )
```

Origin: commit `e370ff5` (closes 7 v5 issues).

### 17.2 RoPE direct concatenation (no intermediate full-size tensor)

**Pattern:** for element-wise math on sliced tensors, compute the
per-slice results and concatenate them directly. Avoid allocating an
intermediate full-size tensor.

```python
# WRONG (v5 original — extra allocation):
def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    x_rot = torch.cat([-x2, x1], dim=-1)   # full-size intermediate
    return x * cos + x_rot * sin

# RIGHT (direct concatenation of rotated halves):
def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos1, cos2 = cos[..., :d], cos[..., d:]
    sin1, sin2 = sin[..., :d], sin[..., d:]
    return torch.cat([x1 * cos1 - x2 * sin1, x2 * cos2 + x1 * sin2], dim=-1)
```

Reduces memory bandwidth — especially impactful on GPU where RoPE is
applied every layer × every loop. With 18 effective layers, this
matters.

Origin: commit `b71f2bd`.

### 17.3 MoE router counting: `torch.bincount`, not `F.one_hot(...).sum()`

**Pattern:** computing per-expert assignment fractions doesn't need a
3D one-hot intermediate.

```python
# WRONG (v5 original — large 3D intermediate):
dispatch_one_hot = F.one_hot(top_idx, num_classes=self.num_routed)
f = dispatch_one_hot.float().sum(dim=(0, 1)) / (N * self.top_k)

# RIGHT (direct integer count):
f = torch.bincount(top_idx.view(-1), minlength=self.num_routed).float() / (
    N * self.top_k
)
```

Same pattern applies to `raw_balance_f`, `dispatch_f`, etc. across
the MoE router code.

Origin: commit `10f8274`.

### 17.4 Sequence-balance loss: `scatter_add_` over `F.one_hot`

**Pattern:** when you need per-batch grouping (2D aggregation), use
`scatter_add_` into a pre-zeroed tensor instead of a 4D one-hot
intermediate.

```python
# WRONG (v5 original — large 4D intermediate B×S×K×E):
seq_one_hot = (
    F.one_hot(raw_balance_top_idx, num_classes=self.num_routed)
    .float()
    .view(B, S, self.top_k, self.num_routed)
)
f_seq = seq_one_hot.sum(dim=(1, 2)) / (S * self.top_k)

# RIGHT (direct scatter-add into B×E, ~20× faster):
f_seq = torch.zeros(
    B, self.num_routed, dtype=torch.float32,
    device=raw_balance_top_idx.device,
)
ones = torch.ones_like(raw_balance_top_idx.view(B, -1), dtype=torch.float32)
f_seq.scatter_add_(1, raw_balance_top_idx.view(B, -1), ones)
f_seq = f_seq / (S * self.top_k)
```

Origin: commit `096bc7f`.

### 17.5 Regex ReDoS prevention in reward functions

**Pattern:** when matching whitespace adjacent to a newline boundary,
use `[ \t]*` or `[^\S\n]*` instead of `\s*`. `\s*` includes `\n`,
which creates overlapping backtracking paths and O(N²) ReDoS
vulnerability.

```python
# WRONG (v5 original — HIGH severity ReDoS, catastrophic backtracking
# on adversarial input with alternating spaces and newlines):
numbered = re.findall(
    r"(?:^|\n)\s*(?:\d+[\.\):]|step\s+\d+)",
    thinking, re.IGNORECASE,
)

# RIGHT (horizontal whitespace only at newline boundary):
numbered = re.findall(
    r"(?:^|\n)[ \t]*(?:\d+[\.\):]|step\s+\d+)",
    thinking, re.IGNORECASE,
)
```

Apply to ALL regex in reward functions, parsers, and tokenizer-
adjacent code where input is model-generated or user-supplied.

Origin: commit `88074b5`.

### 17.6 General lesson — the "v5 optimization patterns" branch

The branch `archived/v5-optimizations` (on remote and reachable via
`git checkout archived/v5-optimizations`) preserves the full original
commits with their exact code patches and discussion. Reference it
when implementing the equivalent v6 paths.

The five patterns above plus the six Gradio UX improvements (Stop
button, Accordion settings, multiline input, branded empty state,
input length validation, payload size validation) are the engineering
work worth not re-discovering.

---

## 18. Decisions from plan review — status

Two outside reviews (`archive/agy-plan-reviewed.md`,
`archive/codex-plan-review.md`, synthesized in
`archive/SYNTHESIS.md`) flagged decisions before implementation.
**Most are now RESOLVED in code.** Tracked here so the history is
clear.

### ✅ Resolved (implemented)

- **Repo / package** — `src/osrt/` exists (the package was renamed
  `nano_osrt` → `osrt`); `pytest` collects and 144 tests pass.
- **KDV (Key-Derived Value) vs MLA** (§6.2) — chose KDV: cache one
  un-rotated 512-d latent, K read off it, V = `v_from_k(latent)`.
- **mHC final collapse** (§8/§10) — dedicated learnable `mhc_collapse`
  head, not a reused `A_l`. mHC dimensional bugs fixed (einsum +
  `.repeat`). `use_mhc=True` in the preset, pending GPU stability test.
- **Hash routing** (§7.5) — loop-indexed top-1, `hash_routing_blocks=0`
  (off) in the canonical preset.
- **Parameter accounting** (§2.1) — generated by `compute_budget.py`
  from the instantiated preset: 601M physical / 278M active.
- **HRA injection count** — 18 (one per effective layer, attention
  path), not the early 87/132 guesses (§2.4).
- **Loss naming** — distinct knobs exist: `aux_loop_loss_weight`,
  `router_aux_loss_coeff`, `router_z_loss_coeff` (§11 / `config.py`).
- **HF (transformers) compliance** — `model.py`: `OSRTConfig` and
  `OSRTForCausalLM` are registered with `AutoConfig` /
  `AutoModelForCausalLM` (+ `register_for_auto_class`) so
  `from_pretrained` / `from_config` work without naming the class; a
  bit-exact `from_pretrained` round-trip is verified; `rope_cos`/
  `rope_sin` are `persistent=True` buffers (saved in the checkpoint —
  fixes meta-init garbage on reload); `_no_split_modules =
  ["RecursiveBlock"]` for device-map sharding; and a custom `generate`
  is retained. Gradient checkpointing deliberately uses a **private**
  runtime gate (`OSRTModel._osrt_grad_ckpt`, flipped by the trainer)
  rather than HF's mechanism, so `supports_gradient_checkpointing =
  False` (the HF name would collide with our custom recursion — see
  `config.py` note).

### ⚠ Partially resolved

- **Tokenizer** (§3) — rebuilt with IDs 0-13; **missing 14-20**
  (end_turn / tool / image / audio). Blocks tool-use + multimodal
  until added.
- **Speculative decoding** (§12) — greedy path implemented; it is NOT
  distribution-preserving (no accept/reject sampling). Fine as a
  greedy accelerator; document it as such, don't call it standard
  speculative sampling.

### ⏳ Open (GPU-phase / future)

- **Tier 1 cost** (`README.md` §12) — reconciled to spot-pricing
  assumption; revisit with real GPU-hour numbers once a run exists.
- **mHC stability under sustained training** — flagged NaN-prone on
  CPU pre-flight; profile on GPU before trusting (see `presets.py`
  comment + `review/architecture-optimization-2026-06-08.md` B5).
- **GPU-phase optimizations — mostly LANDED (2026-06-08/09).**
  - **B4 grouped-GEMM MoE** — landed and ON in the canonical preset
    (`moe_grouped_gemm=True`); removes the lone graph break → fullgraph
    compile, dropless, ~9-12% faster. Loop path retained as fallback
    (§7.7).
  - **B2 fused linear-CE** — landed and available for the aux/MTP heads
    (`fused_cross_entropy_chunks`, routed through `osrt.fused_ce` in
    `train.py`); default 0 = off, opt-in per stage (§15.1).
  - **B1 flex-attention sink** — superseded: the attention sink itself
    was **dropped** (`attention_sink=False`) for OOM at seq-8192;
    canonical path is plain flash SDPA (§6.6).
  - **B3 MLA decode V-recompute** — already implemented in `_attention`
    (V is recomputed from the cached un-rotated latent every step; §6.2,
    §13.4).
  See `review/architecture-optimization-2026-06-08.md`.

---

## Document changelog

- **2026-06-07** — initial creation, captures complete OSRT-600M
  architecture spec as planned in README.md
- **2026-06-07** — added §17 implementation notes ported from v5
  optimization commits on `archived/v5-optimizations` branch
- **2026-06-07** — mechanical fixes from plan review
  (`review/SYNTHESIS.md`): vocab typo, K-only KV cache double-count
  removed, deployment memory math reconciled, V-from-K + RoPE
  ordering clarified, tokenizer-spec-vs-disk mismatch flagged. Added
  inline `DECISION REQUIRED` callouts on §6.2 (V-from-K), §7.5
  (hash routing), §10 (mHC pseudocode bugs). Added §18 listing open
  decisions.
- **2026-06-08** — **synced doc to the landed implementation.** §2.1
  regenerated from the instantiated preset (601M physical / 278M
  active, was the hand-derived 607M/288M; `compute_budget.py` itself
  fixed to load the real preset rather than MHA defaults). §7 prose
  corrected to 8 experts / h=3,840 / shared h=2,816. §2.4 HRA = 18
  attention-path adapters. §3 tokenizer status (14/21 built; first 3
  IDs corrected to PAD=0/BOS=1/EOS=2 to match disk). All
  `DECISION REQUIRED` callouts (§6.2, §7.5, §10) converted to
  DECISION MADE with the implemented choice. §18 rewritten as a
  resolved/partial/open status list. §1 reframed from spec to
  implemented.
- **2026-06-09** — **synced to the 2026-06-08/09 code changes.**
  Attention sink DROPPED (§6.6 reframed as removed-behind-flag with the
  seq-8192 OOM reasoning; §6.7 reframed to flash SDPA; §16.4 invariant
  updated; §2.1/§2.5 physical & active both −72 → 601,444,393 /
  278,217,769, "attn sink" dropped from the misc line, table hand-
  adjusted pending a compute_budget.py regen). MoE dispatch documented
  as grouped-GEMM B4 (canonical, fullgraph, dropless) with the
  `.nonzero()` loop as fallback (§7.7); stale 12-expert numbers fixed in
  §7.4/§7.5 and the duplicate §7.6. Collapse telemetry
  (`loop/update_norm_l*`, `loop/hidden_norm_l*`, `moe/dead_experts_total`)
  added to §16.1. HF-compliance (Auto* registration, bit-exact round-
  trip, persistent rope buffers, `_no_split_modules`,
  `supports_gradient_checkpointing=False` + private gate) added to §18
  Resolved; §18 Open updated (B4 landed+ON, B2 fused-CE landed+available
  opt-in, B1 superseded by the sink drop, B3 already implemented).
  Budget/config refreshed in §15.1
  (~$100, total_steps 3,500, warmup 400, cosine 6e-4→6e-5, ~455M
  foundation tokens; seq-8192/b2 ~35.9GB). NOTE: the new data mix
  (`train_config.py`: FineWeb-Edu / Nemotron-CC-Math / Nemotron-Code /
  Cosmopedia etc., dropped CodeParrot + Wikipedia) is a TRAINING config,
  not an architecture spec, so it has no home in this doc — see
  `train_config.py::PretrainConfig.phases`.
- **2026-06-09** — **named the V-from-K contract: KDV (Key-Derived
  Value).** §6.2 callout, §6.3 heading, §13.4 code comment, §15
  decision log, and the top-level blurb all adopt the formal name; the
  `v_from_k` projection is now annotated as "**KDV: derive V from K**"
  in `docs/02-attention.md` §3, with a one-line contract: `V = W·c_kv + b`
  on the un-rotated cached latent. Doc-only — no model code or
  parameter naming changes (the `v_from_k` attribute stays as is).
