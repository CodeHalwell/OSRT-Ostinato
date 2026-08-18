# ARCHITECTURE.md — OSRT-600M technical specification

**Scope:** the technical specification of the OSRT-600M model — every
layer, dimension, formula, and connection. The model is **implemented**
in `src/osrt/`; this doc describes that implementation. Where a number
or behaviour matters exactly, **the code is the source of truth** and
this doc is kept in sync with it (param counts via
`scripts/compute_budget.py`, behaviour via `src/osrt/model.py`).

**Companion docs:**
- [`README.md`](README.md) — design philosophy, why each choice was made
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

**OSRT** = **Optimized Sparse Recursive Transformer**. OSRT-600M is a
recursive Mixtral-style sparse MoE transformer with **3 physical
decoder blocks applied 6 times via depth recurrence** (giving 18
effective layers), using **HRA adapters (high-rank, rank 256)**,
**GQA attention with a KDV (Key-Derived Value) compressed KV cache**,
**manifold-constrained hyper-connections**, and **Muon-optimized
weights** — totaling **601M physical params, 278M active per token
at inference** (46.3% active fraction; see §2.1 for full breakdown),
~2.5B FLOPs equivalent per token. "600M" in the name rounds the
physical count.

> ✅ **ACCOUNTING IS CODE-GENERATED & IMPLEMENTED.** This is no longer
> a paper spec — `src/osrt/` builds the model and all numbers in §2.1
> come from `PYTHONPATH=src python scripts/compute_budget.py`, which
> instantiates the canonical `OSRT_605M_A288M` preset
> (`src/osrt/presets.py`) on a meta device and counts real parameters.
> Re-run it after any config change.

> 🔧 **NAMING vs REALITY.** The preset is named `OSRT_605M_A288M` and
> the repo `OSRT-605M-A269M`; both numbers predate the corrected count.
> The instantiated model is **601M physical / 278M active (inference)**.
> The names are kept (renaming the repo breaks clones; renaming the
> preset churns code) — trust §2.1, not the names.

> 🔧 **NOT in the architecture:** "gated short convolutions" (claimed
> in an early draft) were never specified or implemented — the spec is
> attention + MoE only.

---

## 2. Parameter budget

### 2.1 Exact accounting (generated 2026-06-08)

> ✅ **GENERATED** — run `PYTHONPATH=src python scripts/compute_budget.py`
> (no args = the canonical `OSRT_605M_A288M` preset). The table below is
> a transcription of that output, with two figures hand-adjusted by −72
> for the dropped attention sink (see the note after the table). Do NOT
> pass loose CLI overrides expecting to reproduce the preset — the CLI
> starts from the full preset and only applies explicit `--override k=v`
> on top.

```
COMPONENT                                       PHYSICAL        ACTIVE / TOKEN (inference)
─────────────────────────────────────────────────────────────────────────────────────
Embedding (65,536 × 1,536, tied with LM head)   100,690,944    100,690,944
  -- one row per token at lookup; full matrix touched at the tied LM head

Attention × 3 blocks (GQA + KDV, §6)            17,308,032     17,308,032
  -- per block: q_proj (1536×1536) + kv_down (1536×512)
     + v_from_k (512×512 +b) + out_proj (1536×1536) + QK/attn norms
  -- ~5.77M/block; the KDV (Key-Derived Value) latent is what makes attention this lean

mHC mixers (Sinkhorn/Birkhoff, §8)                  921,766        921,766
  -- per-sub-block A/B/C generators; shared across loop iterations

Shared experts × 3 (SwiGLU, h=2,816)             38,928,384     38,928,384
  -- per block: 3 × 1,536 × 2,816 = 12,976,128; always active

Routed experts: 3 × 8 × (SwiGLU, h=3,840)       424,673,280    106,168,320
  -- per expert: 3 × 1,536 × 3,840 = 17,694,720
  -- top-2 of 8 active per token → 2/8 = 25% routing density

HRA adapters (rank 256, 18 injection points)     14,155,776     14,155,776
  -- adapter_a (1,536 × 256) + adapter_b (256 × 1,536) = 786,432 each
  -- ONE rank-256 parallel adapter per effective layer (3 blocks ×
     6 loops = 18), applied on the attention sub-block input
     (model.py _attention: x_in @ adapter_a @ adapter_b). Fully
     active — no sparse split.

MTP heads × 2 (§9.3)                              4,721,664              0
  -- training-time only; dropped at deploy → 0 active at inference

Router + loop embeddings + norms                    ~45,857        ~45,857

─────────────────────────────────────────────────────────────────────────────────────
TOTAL PHYSICAL                                  601,444,393  →  ~601M
ACTIVE / TOKEN (inference, excl. MTP)                          278,217,769  →  ~278M
ACTIVE FRACTION                                                    ≈ 46.3%
```

(With the training-only MTP heads counted, the train-time active figure
is ~283M; the 278M headline is the inference forward.)

> 🔧 **Attention sink DROPPED (−72 params).** The earlier table (and the
> repo/preset names) carried the per-head learnable sink logits
> (3 blocks × 24 heads = 72 params, in BOTH columns). The canonical
> preset now sets `attention_sink=False` (`presets.py`), and the sink
> `nn.Parameter` is only created `if config.attention_sink:`
> (`model.py::RecursiveBlock.__init__`) — so it is no longer
> instantiated. Physical and active both drop by 72
> (601,444,465 → 601,444,393; 278,217,841 → 278,217,769) and "attn
> sink" is removed from the misc-params line. These two figures were
> hand-adjusted by −72 from the last `compute_budget.py` output pending
> a clean regen; re-run the script to refresh the full table. See §6.6.

### 2.2 At-a-glance

- **Hidden dimension `d_model`**: 1,536
- **Vocab size**: 65,536 (BPE)
- **Physical transformer blocks**: 3
- **Recursive loops**: 6 → 18 effective layers
- **Attention**: GQA 24 query heads / 8 KV heads / head_dim 64
- **MoE**: 1 shared expert (h=2,816) + **8 routed (h=3,840)**, top-2
  - 8 (not 12) for denser routing — see §2.5
- **HRA adapter rank**: 256 (real high-rank, not LoRA-style 16)
- **HRA injection points**: 18 (implementation-defined; see §2.4)
- **mHC expansion**: 4× residual stream width (enabled in canonical
  preset; pending GPU-phase stability test)
- **Position encoding**: Partial RoPE (last 64 dims of Q and K)
- **Activation**: SwiGLU (FFN), Sqrt(Softplus) (routing affinity)
- **Norm**: RMSNorm pre + post sandwich

### 2.3 FLOP count per token (forward pass, one inference)

Approximate, derived from the §2.1 active-param breakdown (2 FLOPs per
active MAC). Per effective layer (one block × one loop):

```
18 effective layers × (
    attention (q/kv_down/v_from_k/out, ~5.77M params)  : ~2 × 5.77M  = ~11.5M FLOPs
  + shared expert (~12.98M params)                      : ~2 × 12.98M = ~26M FLOPs
  + routed top-2 (2 × 17.69M/8 experts ≈ 4.42M active)  : ~2 × 4.42M  = ~8.8M FLOPs
  + HRA adapter (786K params)                           : ~2 × 0.79M  = ~1.6M FLOPs
  + mHC + norms                                         : ~1M FLOPs
)  ≈ 18 × ~49M  = ~880M FLOPs
+ embedding lookup (negligible) + tied LM head (~2 × 100.7M = ~200M)

TOTAL: ~1.1B FLOPs per token (forward); ~3.3B with backward

(Approximate — FLOP definitions vary. Use as ratios. For exact param
counts see §2.1; this FLOP estimate is hand-derived from them.)
```

### 2.4 HRA injection enumeration

The implementation injects **18 HRA adapter pairs** — one per
*effective layer* (3 blocks × 6 loops = 18), applied on the attention
sub-block (`model.py::_attention`: `x_in @ adapter_a @ adapter_b`).
This is NOT per-projection (an early draft envisioned 132 across
Q/K/V/O + every expert + router); it is one parallel rank-256 path
per block forward. Verify in code:

```bash
PYTHONPATH=src python -c "from osrt.model import OSRTForCausalLM; \
from osrt.config import OSRTConfig; from osrt.presets import OSRT_605M_A288M; \
m = OSRTForCausalLM(OSRTConfig(**OSRT_605M_A288M)); \
print(sum(p.numel() for n,p in m.named_parameters() if 'adapter' in n))"
# -> 14155776
```

Total HRA params: 18 × (2 × 1,536 × 256) = 18 × 786,432 = **14,155,776**.
All fully active per token — the adapters sit on the always-run
attention path, not on the sparse routed experts, so there is no
top-k masking of HRA.

### 2.5 Why "OSRT-600M" (name vs physical count)

`OSRT` = **Optimized Sparse Recursive Transformer**:
- **O**ptimized — Muon optimizer + AlphaQ + TurboQuant deployment stack
- **S**parse — MoE (top-2 of 8 routed + 1 shared per block)
- **R**ecursive — 3 physical blocks × 6 loops = 18 effective layers
- **T**ransformer — standard pre-norm decoder backbone

`600M` rounds the **physical** parameter count of **601,444,393**
(601M). Active per token at inference is **278M** (46.3%). (The count
dropped by 72 from 601,444,465 when the per-head attention-sink logits
were removed — see §2.1 / §6.6.)

**Naming note:** the GitHub repo is `OSRT-605M-A269M` and the canonical
preset `OSRT_605M_A288M`; both numbers were locked at earlier points
before the count was generated cleanly (the "605/607" came from a
compute_budget CLI run that fell back to MHA defaults; the "288"
included an attention overcount + the training-only MTP heads). The
instantiated model is 601M / 278M. The names are kept — renaming the
repo breaks clones, renaming the preset churns `app.py`/training
imports — but **§2.1 is authoritative, not the names.** An alias
`OSRT_605M_A279M` is also kept for back-compat with older imports.

---

## 3. Tokenizer specification

### 3.1 BPE configuration

- **Algorithm**: byte-level BPE (sentencepiece or HuggingFace
  tokenizers)
- **Vocab size**: 65,536
- **Encoding focus**: English + 6 multilingual (Arabic, Japanese,
  Korean, Spanish, French, German) + code (Python, JS, Rust, C++)
- **Pre-tokenization**: GPT-2 style regex (handles contractions,
  numbers, punctuation)

> 🔧 **PARTIALLY BUILT — `tokenizer/tokenizer.json` has 14 of 21
> spec tokens.** The on-disk v6 tokenizer was rebuilt with the correct
> base IDs (PAD=0, BOS=1, EOS=2, FIM 4-6, think/answer 7-10,
> user/assistant/system 11-13). **Still missing IDs 14-20:**
> `<|end_turn|>`, `<|tool_call|>`/`<|/tool_call|>`,
> `<|tool_result|>`/`<|/tool_result|>`, `<|image|>`, `<|audio|>`.
> Basic chat works; **tool-use and multimodal will silently mis-tokenize
> until these are added** (the strings get byte-BPE'd into fragments).
> Add them + a `tokenizer_contract_test.py` asserting
> `tok("<|end_turn|>") == [14]` before any tool-use / vision training.
> The IDs below are the full v6 contract (✓ = on disk now).

### 3.2 Special tokens (reserved IDs — v6 contract)

IDs 0-13 are ✓ on disk (`tokenizer/tokenizer.json` + the HF config
`bos=1, eos=2, pad=0`); IDs 14-20 are the contract but NOT yet built.

| token | id | role | on disk? |
|---|---|---|---|
| `<|padding|>` | 0 | PAD | ✓ |
| `<|begin_of_text|>` | 1 | BOS | ✓ |
| `<|end_of_text|>` | 2 | EOS | ✓ |
| `<|unknown|>` | 3 | unk | ✓ |
| `<|fim_prefix|>` | 4 | FIM prefix marker | ✓ |
| `<|fim_middle|>` | 5 | FIM middle marker | ✓ |
| `<|fim_suffix|>` | 6 | FIM suffix marker | ✓ |
| `<|think|>` | 7 | reasoning block open | ✓ |
| `<|/think|>` | 8 | reasoning block close | ✓ |
| `<|answer|>` | 9 | answer block open | ✓ |
| `<|/answer|>` | 10 | answer block close | ✓ |
| `<|user|>` | 11 | user turn open | ✓ |
| `<|assistant|>` | 12 | assistant turn open | ✓ |
| `<|system|>` | 13 | system prompt open | ✓ |
| `<|end_turn|>` | 14 | turn separator (ChatML style) | ✗ missing |
| `<|tool_call|>` | 15 | tool invocation open | ✗ missing |
| `<|/tool_call|>` | 16 | tool invocation close | ✗ missing |
| `<|tool_result|>` | 17 | tool result open | ✗ missing |
| `<|/tool_result|>` | 18 | tool result close | ✗ missing |
| `<|image|>` | 19 | reserved for vision retrofit | ✗ missing |
| `<|audio|>` | 20 | reserved for future audio | ✗ missing |

IDs 21-31 reserved for future expansion. Real vocab begins at id 32.

### 3.3 Chat template

```
<|system|>{system_message}
<|user|>{user_question}
<|assistant|><|think|>{reasoning}<|/think|><|answer|>{final_answer}<|/answer|>
<|end_turn|>
```

Multi-turn:
```
<|system|>{system}
<|user|>{q1}<|assistant|>{a1}<|end_turn|>
<|user|>{q2}<|assistant|>{a2}<|end_turn|>
```

Tool use:
```
<|user|>{question_needing_calc}<|assistant|>
<|think|>I need to compute 17 × 23.<|/think|>
<|tool_call|>calculator("17 * 23")<|/tool_call|>
<|tool_result|>391<|/tool_result|>
<|answer|>The answer is 391.<|/answer|><|end_turn|>
```

---

## 4. Embedding layer

### 4.1 Shape and tying

- `embedding_matrix ∈ ℝ^(65536 × 1536)`
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

### 5.3 Sandwich RMSNorm

Two RMSNorm layers per physical block per sub-block:
- `RMSNorm_pre[b]` before attention
- `RMSNorm_post[b]` before MoE FFN

Each is `RMSNorm(d_model=1536, eps=1e-6)` with learnable scale, no
bias. Total norm params per block: 2 × 1536 = 3,072.

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
- 1 always-active shared expert (h=2,816)
- **8 routed experts** (h=3,840), top-2 active per token
- 1 router (linear projection + sqrt-softplus affinity)

8 routed (not the 12 of an early draft): top-2 of 8 = 25% routing
density vs 16.7% for top-2 of 12 — denser routing, more capacity per
token, less expert under-utilization at 601M scale. Each of the 8 is
wider (h=3,840) to absorb the capacity.

### 7.2 Shared expert (SwiGLU)

```
w_gate ∈ ℝ^(1536 × 2816)       # 4.33M params
w_up ∈ ℝ^(1536 × 2816)         # 4.33M params
w_down ∈ ℝ^(2816 × 1536)       # 4.33M params

shared_output(x) = w_down @ (SiLU(w_gate @ x) ⊙ (w_up @ x))
```

Per shared expert: ~12.98M params. Across 3 blocks: 38.93M.
(h=2,816 chosen by `compute_budget.py` to land the overall ~601M
target; revisit at GPU phase.)

### 7.3 Routed experts (SwiGLU)

Per routed expert:
```
w_gate ∈ ℝ^(1536 × 3840)       # 5.90M params
w_up ∈ ℝ^(1536 × 3840)         # 5.90M params
w_down ∈ ℝ^(3840 × 1536)       # 5.90M params
```

Per expert: ~17.69M. Per block (8 experts): 141.56M. Across 3 blocks:
424.67M (the dominant param term — ~71% of physical; ~25% active per
token via top-2 routing).

### 7.4 Router

```
W_route ∈ ℝ^(1536 × 8)         # 12,288 params per block
b_route_bias ∈ ℝ^(8)           # per-expert bias for load balancing
                                # (not in gradient; nudged by load deviation)
```

Affinity score:
```
affinity = sqrt(softplus(W_route @ x))      # sqrt(softplus) — DeepSeek-V4
balanced_affinity = affinity + b_route_bias  # static bias for balancing
top_2_indices = argmax(balanced_affinity, k=2)

normalized_weights = softmax(balanced_affinity[top_2_indices])
# (DeepSeek-style: bias only in TOP-K selection, not in gating weights)
```

### 7.5 Hash routing for blocks 0 and 1

For physical blocks 0 and 1 (first 2 of 3), routing is HASH-based,
not learned:
```
expert_id = hash(token_id) mod 8     # mod num_routed_experts
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

Per-expert balancing bias `b_route_bias[i]` accumulates per training
step:
```
mean_load = (1/8) × total_tokens_in_batch
for i in range(8):
    deviation = expert_load[i] - mean_load
    if deviation > 0:
        b_route_bias[i] -= γ              # nudge down
    else:
        b_route_bias[i] += γ              # nudge up
# γ = 0.001 (per DeepSeek-V3)
# This bias is HEURISTIC — not in the gradient
```

Combined with a small sequence-balance loss (weight 0.0001) to prevent
extreme imbalance within single sequences.

### 7.6 Aux-loss-free load balancing

> 🔧 **Duplicate heading** — this section repeats §7.6 above (8 experts);
> kept as-is to preserve numbering. Numbers below corrected from the
> stale 12-expert form.

Per-expert balancing bias `b_route_bias[i]` accumulates per training
step:
```
mean_load = (1/8) × total_tokens_in_batch
for i in range(8):
    deviation = expert_load[i] - mean_load
    if deviation > 0:
        b_route_bias[i] -= γ              # nudge down
    else:
        b_route_bias[i] += γ              # nudge up
# γ = 0.001 (per DeepSeek-V3)
# This bias is HEURISTIC — not in the gradient
```

Combined with a small sequence-balance loss (weight 0.0001) to prevent
extreme imbalance within single sequences.

### 7.7 MoE output

```
moe_output(x) = shared_output(x) + Σ_{i ∈ top2} weight_i × routed_output_i(x)
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

## 8. Manifold-Constrained Hyper-Connections (mHC)

### 8.1 Residual stream expansion

Standard transformers: residual stream is `[batch, seq, d_model]`
(1536 channels).

mHC expands by factor `n_hc = 4`:
```
X ∈ ℝ^(batch × seq × n_hc × d_model)    # 4 channels × 1,536 = 6,144 total
```

Per channel still operates on `d_model=1536`; the inner layers
(attention, MoE) consume one 1,536-dim view and produce one 1,536-dim
output. The 4× expansion is in the **residual** stream only.

### 8.2 Mixing matrices (dynamic)

Per mHC application (one per attention sub-block, one per MoE
sub-block), three mixing matrices:

```
A_l ∈ ℝ^(1 × 4)       # input mapping: residual → layer input
B_l ∈ ℝ^(4 × 4)       # residual transformation (the constrained matrix)
C_l ∈ ℝ^(4 × 1)       # output mapping: layer output → residual
```

Update rule:
```
X_{l+1} = B_l @ X_l + C_l @ F_l(A_l @ X_l)
        = (residual mixing) + (layer contribution)
```

Where `F_l` is either Attention or MoE depending on which sub-block.

### 8.3 Birkhoff polytope constraint on B_l

`B_l` is constrained to be **doubly stochastic** (rows and columns
sum to 1, all entries ≥ 0). This is the Birkhoff polytope of n×n
matrices.

Constraint enforcement:
```
# Start with unconstrained ~B_l (computed as in §8.4)
# Project onto manifold via Sinkhorn-Knopp iteration
M_0 = exp(~B_l)
for t in range(t_max=20):
    M = T_r(T_c(M))     # alternating row/column normalization
B_l = M
```

The doubly-stochastic constraint guarantees `||B_l||_2 ≤ 1` (spectral
norm bounded). This means the residual transformation is
**non-expansive** — guaranteed numerical stability across the
forward pass and backprop. Closed under multiplication, so deep stacks
(our 18 effective layers) stay stable.

### 8.4 Dynamic parameter generation

`A_l`, `B_l`, `C_l` are dynamically generated per token:

```
# Flatten + normalize the residual stream
X_flat = vec(X_l) ∈ ℝ^(1 × (4 × 1536))      # (1 × 6144)
X_normed = RMSNorm(X_flat)

# Generate raw (unconstrained) parameters
~A_l = α_pre × (X_normed @ W_pre) + S_pre        # (1 × 4)
~B_l = α_res × Mat(X_normed @ W_res) + S_res     # (4 × 4)
~C_l = α_post × (X_normed @ W_post)^T + S_post   # (4 × 1)
```

Where:
- `W_pre ∈ ℝ^(6144 × 4)` (dynamic component for A_l)
- `W_res ∈ ℝ^(6144 × 16)` (dynamic component for B_l, reshaped to 4×4)
- `W_post ∈ ℝ^(6144 × 4)` (dynamic component for C_l)
- `S_pre, S_res, S_post`: static learnable biases
- `α_pre, α_res, α_post`: learnable gating factors, initialized small

### 8.5 Sigmoid bounds on A_l and C_l

```
A_l = σ(~A_l)         # bounded [0, 1]
C_l = 2σ(~C_l)        # bounded [0, 2]
```

Prevents signal cancellation. The factor 2 on C_l preserves the
ability to scale layer contributions.

### 8.6 Cost (~720K params, ~6.7% overhead)

Total mHC params per injection point (one for attn, one for FFN, per
block, per loop ITERATION):
- 18 effective layers × 2 sub-blocks = 36 mHC applications
- But the dynamic generation matrices are SHARED across the loop
  iterations (per physical block)
- Net added: ~720K trainable params + ~6.7% wall-clock overhead

---

## 9. LM head and auxiliary heads

### 9.1 Main LM head

Tied with embedding:
```
logits = embedding.weight @ x_final.transpose(-1, -2)
# Shape: [batch, seq, 65536]
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
MTP_head_1 ∈ ℝ^(1536 × 65536)     # tied with embedding
MTP_head_2 ∈ ℝ^(1536 × 65536)     # tied with embedding
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
    logits = x_final @ embedding.weight.T               # [B, L, 65536]

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
| mHC matrices | bf16 | kept full precision |
| Loop embeddings | bf16 | kept full precision |
| LayerNorms / biases | bf16 | always bf16 |
| K cache (per layer) | **int4** | TurboQuant random-rotation + per-block |

### 14.2 Memory budget at deployment

Numbers below are decimal MB (1 MB = 1,000,000 bytes), no allocator
overhead, no per-tensor quantization metadata. Param counts are the
real §2.1 figures (MTP heads dropped at deploy):

```
Embedding (int8, 100.7M params × 1 byte):            101 MB
Attention (int8, 17.3M params):                       17 MB
Shared experts (int8, 38.9M params):                  39 MB
Routed experts (FP4 @ ~3.5 bit avg, 424.7M params):
    424.7M × 3.5 bits / 8 ≈ 186 MB (+~2% AlphaQ meta) ~190 MB
HRA adapters (bf16, 14.2M params × 2 bytes):          28 MB
mHC + router + norms + loop_emb (bf16):               ~2 MB
(MTP heads dropped at deploy)                            0 MB

TOTAL ON DISK (all loaded into RAM):                 ~377 MB
```

To fit a tighter (~150-250 MB) envelope, the levers are:

- Routed experts to 2-bit average (~190 MB → ~110 MB) — they're 71%
  of physical, so this is the dominant lever
- Embedding to int4 (101 MB → 50 MB)
- HRA folded into base weights post-RL (eliminates 28 MB) or int8-ed

A "active-only resident" figure (loading just the top-2 routed experts
per layer + paging the rest from disk/CPU) is an inference-system
choice, not a weight choice — state the assumption when quoting it.

### 14.3 AlphaQ bit allocation (routed experts)

Per AlphaQ:
- Compute PL Alpha Hill metric for each routed expert weight matrix
- ILP solver allocates bits ∈ {2, 3, 4} per layer per expert under
  global budget of 3.5 bits average
- Heavy-tailed experts (high importance) → 4 bits
- Light-tailed experts (less critical) → 2 bits
- Layer-wise allocation (each up/gate/down independently)

Expected quality: near-lossless at 3.5-bit average (per AlphaQ
results on Qwen1.5-MoE; our 8-experts-per-block is a similar regime).

---

## 15. Total compute and memory math

### 15.1 Training compute and budget

Per token, per forward pass: ~810M FLOPs (§2.3)
Backward is ~2× forward: ~1.6B FLOPs
Total per token per training step: ~2.4 BFLOPs

**Base-pretrain budget (`train_config.py::PretrainConfig`):** the LR
cosine horizon is sized to a **~$100 Modal H100 run** (≈ $3.95/hr ≈ 25
H100-hr):
```
total_steps  = 3,500          # LR-anneal target; cosine self-terminates here
warmup_steps = 400            # ~11% — spins up Muon + the MoE balance bias
peak_lr      = 6e-4  →  min_lr = 6e-5    # one continuous cosine, no re-warm
```
At ~5k tok/s on the seq-2048 foundation phase (131K tok/step) that is
**~455M tokens** by step 3,500, when the cosine has fully decayed
peak→min_lr and the run self-terminates at the budget with a clean,
annealed base. Long-context (4096/8192) phases and SFT/RL are separate
chunks layered on top (`PretrainExtendConfig`, `SFTConfig`, etc.).

**Memory (80GB H100):** gradient (activation) checkpointing is required
to fit (the trainer flips the private `OSRTModel._osrt_grad_ckpt` gate),
and the fused linear-CE for the aux/MTP heads is **available** to cut
the (B, S, vocab) fp32 logit peak (`fused_cross_entropy_chunks`,
routed through `osrt.fused_ce` in `train.py`; default 0 = off, opt-in
per stage). With the canonical preset, seq-8192/batch-2 sits at
**~35.9GB** (the flash SDPA path — the dropped attention sink's manual
path OOMed here, §6.6). The knowledge phase (seq-4096) runs batch-6 at
~59GB. `torch.compile` is on by default in the trainer
(`compile_enabled`, default True).

### 15.2 Inference compute per token (generation)

Just the forward pass: ~810M FLOPs
On H100 (consumer-equivalent at int8): ~200 GFLOPs effective
→ ~250K tokens/sec single-token decode (unrealistic — bandwidth bound)
→ Realistic on CPU (Snapdragon 8 Elite, int8): ~50-100 tokens/sec

### 15.3 Memory at inference (full deployment)

Weights: ~150-200 MB
KV cache (4K context, full stack): ~5 MB
Activations (transient): ~50 MB
Total: **~250 MB** — fits comfortably on phones / Raspberry Pi 5.

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
