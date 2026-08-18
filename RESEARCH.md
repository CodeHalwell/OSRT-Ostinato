# RESEARCH.md — External research summarised

**Scope:** every paper, technical report, and research synthesis that
shaped OSRT-600M's design. Each entry: what it is, the key findings,
and what we adopted (vs deferred or rejected).

**Companion docs:**
- [`README.md`](README.md) — the v6 design that cites these
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — v6 technical spec
- [`LEARNINGS.md`](LEARNINGS.md) — what v5 (363M) taught us

---

## Table of contents

1. [Muon optimizer](#1-muon-optimizer)
2. [DeepSeek-V2 / V3 / V4](#2-deepseek-v2--v3--v4)
3. [LFM2 (Liquid AI)](#3-lfm2-liquid-ai)
4. [Ouro / Huginn (recurrent / looped LMs)](#4-ouro--huginn-recurrent--looped-lms)
5. [SmolLM2 / SmolLM3](#5-smollm2--smollm3)
6. [Gemma 3 / Gemma 4](#6-gemma-3--gemma-4)
7. [Qwen3 / Qwen2.5-MoE](#7-qwen3--qwen25-moe)
8. [DeepSeekMoE](#8-deepseekmoe)
9. [OLMoE](#9-olmoe)
10. [AlphaQ (calibration-free MoE quantization)](#10-alphaq-calibration-free-moe-quantization)
11. [TurboQuant + QJL (Google)](#11-turboquant--qjl-google)
12. [Nemotron-CC + Nemotron 3 Ultra](#12-nemotron-cc--nemotron-3-ultra)
13. [Recurrent MoE + Muon synthesis (Yang et al. report)](#13-recurrent-moe--muon-synthesis)
14. [Small-language-model frontier survey](#14-small-language-model-frontier-survey)
15. [Other named references](#15-other-named-references)

---

## 1. Muon optimizer

**Sources:**
- Keller Jordan, "Muon" GitHub repo (Oct 2024)
- Moonshot AI, "Muon is Scalable for LLM Training" (arXiv 2502.16982, Feb 2025)
- Dao-AILab, "Gram Newton-Schulz" repo
- "AdaMuon" / "Newton-Muon" variants (arXiv 2025)
- "Muon: Training and Trade-offs with Latent Attention and MoE" (arXiv 2509.24406)

### Key idea

Replace AdamW for 2D hidden-layer matrices with a SGD-momentum
optimizer that orthogonalizes its update direction via Newton-Schulz
iteration on the polar decomposition. The update lies on the Stiefel
manifold (orthogonal matrices).

Algorithm:
```
M_t = β·M_{t-1} + G_t                 # momentum
M̃_t = β·M_t + G_t                     # Nesterov
U_t = NewtonSchulz(M̃_t / ||M̃_t||_F)   # orthogonalize via NS5
W_{t+1} = W_t - η·(U_t + λ·W_t)        # update + decoupled weight decay
```

### Findings

- **~2× compute efficiency vs AdamW** at compute-optimal training
  (Moonshot Moonlight: 3B/16B MoE on 5.7T tokens)
- **~50% memory savings** vs AdamW (no second-moment vector)
- **Implicit spectral norm regularization** — Muon = steepest descent
  under spectral norm; controls singular values of weight matrices
- **Aligns with μP** (maximal update parametrisation) for HP transfer
- **Gram Newton-Schulz variant** (Dao-AILab): iterates on small
  Gram matrix X·X^T instead of full X. ~2× FLOP reduction for
  non-square matrices on Hopper/Blackwell
- **Hybrid Newton-Schulz** (DeepSeek-V4): 10 iterations in 2 stages —
  first 8 at (3.4445, −4.7750, 2.0315) for rapid convergence, last
  2 at (2, −1.5, 0.5) to stabilize singular values at 1

### What we adopted

- Muon for all 2D hidden matrices (attention QKV+out, FFN gates,
  router projection). AdamW for embeddings, LM head, RMSNorm, biases
- Moonlight recipe: weight decay 0.01-0.10, update-RMS alignment
- DeepSeek-V4's hybrid Newton-Schulz coefficients
- Gram Newton-Schulz if implementing fresh; standard NS5 acceptable

### What we deferred

- AdaMuon (element-wise Adam moments on orthogonalized direction)
- Newton-Muon (right preconditioning via input second moment)

---

## 2. DeepSeek-V2 / V3 / V4

**Sources:**
- DeepSeek-V2 paper (arXiv 2405.04434)
- DeepSeek-V3 technical report (arXiv 2412.19437)
- DeepSeek-V4 preview announcement (DeepSeek API news,
  https://api-docs.deepseek.com/news/news260424 — released 2026-04-24)
- DeepSeek-V4-Flash Hugging Face card
  (huggingface.co/deepseek-ai/DeepSeek-V4-Flash) — primary public spec
  for mHC, CSA/HCA, Muon, sqrt(softplus) routing, hash routing
- HF Transformers DeepSeek-V4 model docs
  (github.com/huggingface/transformers/.../model_doc/deepseek_v4.md)
- DeepSeek-V3.2 (arXiv 2512.02556)

> 🔧 **CITATION CORRECTION (2026-06-07).** Earlier drafts cited
> arXiv 2511.23404 as the DeepSeek-V4 technical report. That arXiv
> ID is the **LFM2 Technical Report** (see §3). DeepSeek-V4 has no
> arXiv preprint yet — the primary sources are the DeepSeek news
> page and the Hugging Face model card linked above. See
> `review/SYNTHESIS.md` Tier 1 #10.

### Key innovations across versions

**V2:** introduced **MLA (Multi-Head Latent Attention)** — cache a
small latent c_KV instead of full K/V. Up-projections W_UK, W_UV
reconstruct K, V from c_KV at decode. Brilliant matrix-absorption
trick at decode time:
```
Q_eff = Q @ W_UK^T              # absorb K up-proj into Q
scores = Q_eff @ c_KV^T          # latent acts as K
weighted = softmax(scores) @ c_KV  # latent acts as V
out = weighted @ W_UV            # absorb V up-proj into output
```
K and V never materialize at decode. Cache is ~70 elements/token
vs ~1024 for standard.

**V3:** added **aux-loss-free load balancing** for MoE. Per-expert
bias `b_i` nudged ±γ=0.001 by load deviation, NOT in the gradient.
Replaces traditional aux loss (which distorts the LM gradient).
Trained 671B-total / 37B-active on ~15T tokens.

**V4 (preview, Dec 2025):**
- **Hybrid attention**: CSA (Compressed Sparse Attention) + HCA
  (Heavily Compressed Attention) interleaved
- CSA: compress every m=4 tokens → 1 entry, then sparse top-k
  attention via "lightning indexer"
- HCA: compress every m'=128 tokens → 1 entry, dense attention
- **mHC (Manifold-Constrained Hyper-Connections)**: residual
  mapping constrained to doubly-stochastic matrices via
  Sinkhorn-Knopp. Guarantees non-expansive mapping → stable across
  deep stacks
- **Hash routing for first MoE layers**: deterministic hash function
  for early layers, learned routing for later
- **Sqrt(Softplus) routing affinity** (replaces V3's Sigmoid)
- **Multi-teacher On-Policy Distillation (OPD)** replaces mixed RL
  stage. Train specialists separately, distill into unified model
  via reverse-KL on student's own outputs
- **SwiGLU Clamping** [-10, 10] for training stability
- **Anticipatory Routing**: decouple routing from forward pass, use
  historical θ_{t-Δt} for routing while current θ_t for features
- **FP4 QAT** during post-training (MXFP4 format, lossless FP4→FP8
  dequant via STE)
- Achievement: 10% of V3.2's KV cache at 1M context

### What we adopted

- **MLA + matrix absorption** for attention (alternative to GQA)
- **Aux-loss-free balancing** (γ=0.001)
- **HCA-style sequence compression** for KV cache reduction
- **Multi-teacher OPD** as the post-training pipeline (replaces our
  planned multi-env GRPO)
- **mHC** for residual connections in deep recursive stack
- **Hybrid Newton-Schulz** coefficients for Muon
- **Sqrt(Softplus) routing**, **Hash routing for first 2 MoE layers**
- **SwiGLU Clamping** for stability
- **FP4 QAT** for post-training expert quantization

### What we deferred

- CSA (lightning indexer + sparse top-k) — useful only at >32K context
- Anticipatory Routing — engineering-heavy, defer unless instability
- DeepSeek's specific MoE expert count (256+ experts not needed at 600M)
- Million-token context capability (out of scope at $280)

---

## 3. LFM2 (Liquid AI)

**Source:** "LFM2 Technical Report" (Liquid AI, Nov 2025) —
**arXiv: 2511.23404**. Previously miscredited to DeepSeek-V4
elsewhere in this doc; corrected as of 2026-06-07.

### Family

- LFM2-350M, 700M, 1.2B, 2.6B dense
- LFM2-8B-A1B MoE (8.3B total / 1.5B active)
- LFM2-VL (vision-language), LFM2-Audio, LFM2-ColBERT (retrieval)
- All support 32K context, optimized for on-device CPU/mobile

### Key findings

- **Hardware-in-the-loop architecture search** on real devices
  (Samsung S25 Snapdragon, AMD Ryzen) → most layers should be
  **gated short convolutions** + minority **GQA** blocks. SSM /
  linear-attention variants did NOT help when paired with attention.
- **Gated short conv block**:
  ```
  (B, C, h̃) = Linear(h)
  y = B ⊙ h̃
  z = Conv_k(y)             # depthwise 1D, k=3
  o = Linear_out(C ⊙ z)
  ```
- **65,536 vocab BPE** held constant across all sizes (English + 6
  multilingual + code; FIM + tool-calling + ChatML tokens)
- **GQA configurations**: 16q/8kv (350M), 24q/8kv (700M), 32q/8kv
  (1.2B/2.6B), all head_dim=64. QK-Norm everywhere.
- **Three-stage post-training**: SFT → length-normalized DPO with
  on-policy CLAIR refinement → model merging (TIES/DARE/DELLA)
- **Decoupled Top-K knowledge distillation** during pretraining:
  decompose KL via chain rule into (binary mass term) + (conditional
  Top-K KL with temperature on the conditional only). Avoids support
  mismatch when tempering a Top-K truncated distribution.
- **Curriculum learning** via ensemble difficulty scoring: 12 models
  rate each SFT example, train easy → hard
- **Quick Instruction tokens** for auxiliary tasks (web search
  trigger, intent recognition) — reuse KV cache, no separate small
  model needed
- **Up to 2× faster prefill/decode on CPUs** vs similarly sized
  attention-heavy baselines
- **LFM2-2.6B benchmarks**: 82.41% GSM8K, 79.56% IFEval at 11T tokens

### What we adopted

- **Vocab 65,536** and **hidden 1536** for OSRT-600M (matching
  LFM2-700M)
- **Decoupled Top-K KD** as MOPD objective improvement (~32× denser
  supervision per token vs plain CE)
- **Curriculum learning** via ensemble difficulty scoring (~$5
  one-time cost)
- **Length-normalized DPO + CLAIR** for any future preference learning
- **Model merging** at end of post-training (multiple SFT/alignment
  variants → TIES/DARE/DELLA merge)
- **GQA 24q/8kv head=64** (their 700M config)
- **QK-Norm** confirmed standard

### What we deferred

- **Gated short conv replacement for attention** — biggest single
  architecture change; risk vs reward unclear at our scale, but
  promising. Listed as v7 consideration.
- **Hardware-in-the-loop search** at our scale (would need ~$1K of
  proxy training runs)
- **LFM2-Audio architecture** — bookmark if we go multimodal-audio
- **LFM2-ColBERT** retrieval — not in our use case

---

## 4. Ouro / Huginn (recurrent / looped LMs)

**Sources:**
- Huginn-3.5B / "Recurrent Transformer" (Geiping et al., arXiv 2502.05171, Feb 2025)
- Ouro / LoopLM (Zhu et al., arXiv 2510.25741, Oct 2025)
- Universal Transformers (Dehghani et al., 2019)
- RecurrentGemma / Griffin (Google DeepMind)
- CART: Context-Anchored Recurrent Transformer (ResearchGate 2024)

### Key idea

Apply a small block of weights repeatedly (depth recurrence) to give
effective depth without proportional parameter cost. The "weight
tying across depth" + "iteration-specific conditioning" pattern.

### Findings

- **Huginn-3.5B**: Prelude (2 blocks) → Recurrent core (4 blocks
  applied N times) → Coda (2 blocks). Trained with log-normal
  Poisson distributed iteration counts + truncated backprop
  (~8 iterations). At 132 unrolls, 8 physical blocks act like
  132 layers.
- **Ouro-1.4B / 2.6B (R4)**: looped LM, 7.7T tokens. **Beats dense
  Qwen3-4B on GSM8K (78.92 vs 72.86) and MATH500 (82.40 vs 59.60).**
  Reports 2-3× parameter-efficiency gains.
- **Validation of recursive thesis**: parameter efficiency via depth
  recurrence is real and reproducible
- **Instability at high recursion** (Ouro paper) — R4 is sweet spot,
  R8+ needs careful stabilization
- **Latent overthinking** can hurt at very complex tasks (probing
  study arXiv 2507.02199 — "Latent Chain-of-Thought?")
- **Loop embeddings** (Universal Transformer depth embedding) are
  critical — pure weight-tying without loop conditioning degenerates
- **CART** (Context-Anchored Recurrent Transformer): compute K/V
  once in prelude, recurrent core does cross-attention to frozen
  K/V. Trades expressivity for KV cache reduction.

### What we adopted

- **Recursive 3 × 6 design validated** (v5 used this; v6 keeps it)
- **Loop embeddings** kept from v5
- **Cap at R=6** (don't chase higher per Ouro instability findings)
- **Aux per-loop LM head losses** (architectural-fix knob from v5)

### What we deferred / rejected

- **CART-style frozen K/V** — too expressivity-limiting; kept full
  KV cache and attacking with GQA + AlphaQ + TurboQuant instead
- **Huginn-style 32+ unrolls** — too unstable, way past Ouro's R4
  sweet spot
- **Variable loop count at training** — Ouro's adaptive depth
  allocation; could be a v7 feature

---

## 5. SmolLM2 / SmolLM3

**Sources:**
- "SmolLM2" (Allal et al., arXiv 2502.02737)
- SmolLM3-3B (HuggingFace blog + report)

### Key findings

- **SmolLM2-1.7B at 11T tokens** = 6,500 tokens/param ratio. Way
  past Chinchilla. Validates over-training for SLMs.
- **Multi-stage manual mixture rebalancing**: looked at stagewise
  eval results, adjusted data mix. Single-pass mix never worked.
- **Created new datasets** when public ones too small or low-quality
  (FineMath, Stack-Edu, SmolTalk)
- **WSD schedule**: 2000-step linear warmup → long stable phase →
  decay. SmolLM3 uses this.
- **Total cost ~$250K** for full SmolLM2 (1.7B at 11T tokens)
- **SmolLM3 staging**:
  - Stage 1 (0→8T): web 85% (FineWeb-Edu + DCLM + FineWeb2), code 12%
  - Stage 2 (8→10T): higher-quality math/code
  - Stage 3 (decay): upweight highest-quality math/code
- **Settings**: AdamW (β1=0.9, β2=0.95, ε=1e-8), weight decay 0.1
  EXCLUDING embeddings, gradient clip 1.0

### What we adopted

- **WSD schedule** for OSRT-600M
- **Multi-stage data mixture with annealing**
- **No weight decay on embeddings** (preserve representation norms)
- **Token budget aiming for ≥1000 tokens/param**
- **SmolLM-style staged data progression** (broad → math/code → anneal)

### What we deferred

- Full SmolLM2 token volume (11T) — out of budget. v6 plan is 3T
  ideal, 12B-100B at $280.

---

## 6. Gemma 3 / Gemma 4

**Sources:**
- Gemma 3 technical report (Google DeepMind, 2025)
- Gemma 3 270M model card
- Gemma 4 (preview)

### Key findings (Gemma 3)

- **270M model: 170M / 270M = 63% on EMBEDDINGS** because of 256K
  vocab. Only 100M in transformer blocks. Vocab tax disaster at
  small scale.
- **Specialization-first framing** for 270M: "strong fine-tuning
  base", NOT a frontier generalist
- **5:1 local:global attention layout** with 1024-token sliding
  window. Cuts KV cache O(n × window) not O(n²).
- **Pre-norm + post-norm (sandwich) RMSNorm** on every layer
  (replaces Gemma 2's soft-capping)
- **QK-norm** on every attention block
- **RoPE θ raised to 1M on global layers** for 128K context
- **Training: knowledge distillation** during pretraining
- **Pretraining at 32K context**, extend to 128K near end (not full-
  context day 1)
- **Quantization-aware variants released** for deployment

### Key findings (Gemma 4)

- **Dedicated multi-token prediction draft models** for speculative
  decoding
- **Configurable thinking modes** (think/no-think)
- **270M trained on 6T tokens = 22,000 tokens/param ratio** — the
  current SLM frontier

### What we adopted

- **Sandwich RMSNorm** (pre + post) in recurrent block — critical for
  our 18-effective-layer stack
- **QK-norm** standard
- **Context progression** (pretrain at 4K, extend to 8K in decay)
- **Local/global attention layout** (5:1) — noted as v7 consideration
- **Knowledge distillation in pretraining** — defer at $280, but
  noted as +$300 upgrade in OSRT_600M tier 3 budget
- **Speculative decoding via MTP heads** (we already train them via
  aux_loop_loss)

### What we rejected

- **256K vocab** — disastrous at small scale (Gemma 3 270M's mistake)

---

## 7. Qwen3 / Qwen2.5-MoE

**Sources:**
- Qwen3 technical report (arXiv 2505.09388)
- Qwen2.5-MoE (Qwen blog)
- Qwen1.5-MoE-A2.7B

### Key findings

- **Single-checkpoint controllable reasoning** (think/no-think
  switching via thinking budget)
- **GQA + QK-Norm + RMSNorm + SwiGLU + RoPE** universal across sizes
- **Tied embeddings** at ≤4B (0.6B, 1.7B, 4B); untied at 8B+
- **Vocab 151,669** — large but Qwen pays for the breadth
- **Qwen3-MoE dropped shared experts** (vs Qwen1.5-MoE which had
  them); uses global-batch load balancing
- **Qwen3 corpus: 36T tokens over 119 languages and dialects**, PDF-
  derived text + synthetic
- **Scaling-law-guided HP selection**
- **Qwen3 MoE configurations**: 128 experts, top-8 (much finer-
  grained than DeepSeek's 256/8 or our 12/2)
- **Qwen3-30B-A3B Instruct-2507** uses **global-batch load balancing
  loss** (not per-microbatch)

### What we adopted

- **GQA + QK-norm + RMSNorm + SwiGLU + RoPE + tied embed** all
  confirmed
- **Global-batch load balancing** for routing (add to v6 stack)
- **Single-checkpoint controllable reasoning** (we get this via
  loop_count knob + think/no-think SFT mix)

### What we rejected

- **128 experts top-8** — too fine-grained at 600M; we use 8 experts
  top-2
- **151K vocab** — too expensive at 600M (would be 200M+ embedding)
- **Drop shared experts** — DeepSeekMoE argument for shared experts
  more persuasive at sub-1B scale

---

## 8. DeepSeekMoE

**Source:** Dai et al. "DeepSeekMoE" (arXiv 2401.06066, Jan 2024)

### Key idea

Two innovations on top of standard MoE:

1. **Fine-grained expert segmentation** — split each FFN expert into
   m smaller experts, activate m× more. Increases combinatorial
   routing flexibility at constant params/FLOPs.
2. **Shared expert isolation** — reserve K_s always-on experts to
   absorb common cross-domain knowledge. Routed experts can then
   specialize harder, no longer need to relearn common patterns.

Formula: `h = Σ(shared FFN) + Σ(g · routed FFN) + residual`

Specific results:
- **DeepSeekMoE 16B**: 2 shared + 64 routed (top-6), each expert
  0.25× standard FFN size. Deliberately small balance factor
  α=0.001.
- **DeepSeek-V2**: 2 shared + 160 routed (top-6), expert intermediate
  dim 1536. 236B total / 21B active.

### What we adopted

- **Shared + routed expert hybrid** (1 shared + 8 routed top-2 in
  our 600M)
- **DeepSeekMoE-style fine granularity** but stopped at 8 experts
  (vs their 64 / Qwen3's 128) — at 600M, denser top-2-of-8 routing
  (25% active) beats finer top-2-of-12; the "many tiny experts"
  thesis pays off at 16B+, not here
- **Small aux loss coefficient** (we'll use aux-loss-free per V3
  instead)

### What we rejected

- **64+ routed experts** — too fine at our scale; routing overhead
  exceeds specialisation benefit

---

## 9. OLMoE

**Source:** Muennighoff et al. "OLMoE" (arXiv 2409.02060)

### Key findings

- **OLMoE-1B-7B**: 6.9B total / ~1.3B active, **64 experts top-8**
- **No shared experts** (vs DeepSeek family)
- **Dropless token-choice routing**
- Trained on **5.1T tokens** with **load-balance loss 0.01** +
  **router z-loss 0.001**
- **Fully open** (data, code, logs, checkpoints) — best open
  reference for MoE training
- 18 ablations: fine-grained, sharedless designs strong; MoEs trained
  ~2× faster than dense at equal active params

### What we adopted

- **Router z-loss 0.001** (cheap insurance against router-logit
  blow-up)
- **OLMoE as the open MoE reference** for any debugging

### What we rejected

- **No shared experts** — at sub-1B scale, DeepSeek's shared-expert
  argument more persuasive

---

## 10. AlphaQ (calibration-free MoE quantization)

**Source:** Yang et al. "AlphaQ: Calibration-Free Bit Allocation for
Mixture-of-Experts Quantization" (DeLTa Workshop @ ICLR 2026)

### Key idea

Allocate bit-widths across MoE experts and layers WITHOUT a
calibration dataset. Use Heavy-Tailed Self-Regularization (HT-SR)
theory: experts with heavier-tailed weight spectra are better-trained
and merit higher bits.

Method:
- **PL Alpha Hill** metric: power-law exponent of weight matrix
  eigenvalue distribution. Smaller α = heavier tails = more
  important.
- **FARMS** (Fixed-Aspect-Ratio Matrix Subsampling) — corrects
  aspect-ratio bias in spectral metrics
- **ILP solver** for global bit allocation under budget constraint
- **Layer-wise allocation beats expert-wise** — each up/gate/down
  projection gets independent bit-width

### Findings

- **Qwen1.5-MoE at 3.5 bits/expert = BF16 accuracy** (4× memory
  compression, ZERO degradation)
- At 2.5 bits: ~2.5% accuracy drop on average
- **Especially strong on fine-grained MoEs** (DeepSeekV2-Lite,
  Qwen1.5-MoE) where per-expert variance is large
- Avoids the calibration-overfitting problem: math-calibrated bit
  allocation destroys code performance and vice versa

### What we adopted

- **AlphaQ post-training** for OSRT-600M routed experts
- 3.5 bits/expert target (matches BF16 quality on similar
  fine-grained MoEs)
- ~1 day engineering with their open-source code
- **Pairs naturally with our recursive arch** — per-expert spectral
  analysis done ONCE, applies across all 6 loop reuses

### Why it's especially good for us

Calibration-based methods (PMQ, MxMoE) overfit to whatever
calibration set you pick. Our model serves diverse domains, so
calibration-free is the right answer. The 14-experts-per-block
fine-grained design matches AlphaQ's strong-domain exactly.

---

## 11. TurboQuant + QJL (Google)

**Sources:**
- "TurboQuant" paper (Google Research)
- "QJL: Quantized Johnson-Lindenstrauss" paper (Google Research)

### Key idea (TurboQuant)

KV-cache compression via random rotations + per-block quantization.
Compresses KV cache 4-8× with near-lossless quality (~0.01
perplexity delta).

### Key idea (QJL)

Quantized Johnson-Lindenstrauss random projections for routing
matrices. Keeps the inference path int4-friendly while preserving
routing decisions.

### What we adopted

- **TurboQuant int4 on KV cache** for OSRT-600M deployment
- **QJL on routing matrices** to complete the int4 inference path
- Pairs with AlphaQ (expert weights) + int8 QAT (dense paths) for
  full deployment stack

### Why it matters DOUBLY for us

Our recursive arch caches K/V from EACH effective layer (18 caches
per token vs 3 unique blocks). KV cache memory is the dominant
inference cost. TurboQuant 4-8× reduction is more impactful for us
than for non-recurrent models.

---

## 12. Nemotron-CC + Nemotron 3 Ultra

**Sources:**
- Su et al. "Nemotron-CC" (arXiv 2412.02595)
- Nemotron-CC-Math-v1 dataset (we have access)
- Nemotron 3 Ultra model release

### Key findings

- **Nemotron-CC**: classifier ensembling (FineWeb-Edu + DCLM style)
  + synthetic rephrasing + relaxed heuristics
- "Four times more unique real tokens than DCLM"
- 15T-token Llama-3-8B trained on Nemotron-CC beats Llama-3.1-8B by
  +5 MMLU (70.3 vs 65.3)
- Key insight: aggressive filtering (FineWeb-Edu drops 90%) starves
  long-horizon training of unique tokens. Nemotron-CC's ensemble
  approach mitigates this.
- **Nemotron-CC-Math-v1**: gated dataset (we have access), used in
  pretrain_extend
- **Nemotron 3 Ultra**: 550B-A55B free tier on OpenRouter, but ~100s/
  rollout too slow for our distillation pipeline

### What we adopted

- **Math-first Nemotron/Cosmopedia pretrain mix** — the v6 mix in
  `train_config.py` (2026-06-08) makes NVIDIA Nemotron sets the
  backbone and drops the old codeparrot-clean + Wikipedia streams:
  - **Foundation** (seq 2048): fineweb-edu 0.40, Nemotron-CC-Math-v1
    `4plus` 0.25, Nemotron-Pretraining-Code-v2
    `Synthetic-Question-Answering` 0.20, Cosmopedia `web_samples_v2`
    0.15
  - **Knowledge** (seq 4096): Nemotron-CC-Math-v1 `4plus` 0.25,
    Nemotron-Pretraining-Specialized-v1 `STEM-SFT` 0.15 +
    `Math-Textbooks` 0.15 + `InfiniByte-Reasoning` 0.10, fineweb-edu
    0.15, Nemotron Code `Synthetic-Question-Answering` 0.10, Cosmopedia
    `openstax` 0.10
  - **Instruction** (seq 8192): smoltalk, Evol-Instruct-Code,
    OpenHermes-2.5, Nemotron-Post-Training-Dataset-v1 (math + stem)
  - Most NVIDIA Nemotron sets are **gated** (Nemotron-CC-Math,
    Nemotron-Pretraining-*, Nemotron-Post-Training) — access approval
    required.
- **Nemotron-CC ensemble approach** as Tier 3 upgrade path for v6
  pretraining

### What we rejected

- **Nemotron 3 Ultra as teacher** for MOPD — too slow (100s/rollout
  vs DeepSeek's 13s)
- **codeparrot-clean + Wikipedia** as pretrain streams — dropped from
  the v6 base mix in favour of the Nemotron synthetic-QA code +
  Nemotron/Cosmopedia knowledge streams above (they survive only in
  the mid-training extend configs).

---

## 13. Recurrent MoE + Muon synthesis

**Source:** User-supplied research synthesis report (Nov 2026):
"Co-Designing Recurrent MoE Architectures and Orthogonal Optimizers
for Small Language Models"

### Scope

A 363M recurrent MoE with shared+routed experts + Muon optimizer.
Comprehensive ablation across (optimizer × aux loss) cells.

### Key findings

**Cell C (Muon + 0.10 aux loss)** is the production recipe:
- Final task loss: **3.43** (vs Lion's ~7.4)
- Load balance factor: 1.02 (near-perfect)
- Clean expert minimum activation: 0.105 (no starved experts)

**Cell A (Lion + 0.10 aux):** task loss ~7.4, partial late-warmup
collapse. Lion's first-order updates can't regulate singular value
distribution → representation collapse.

**Cell B (Lion + 0.00 aux):** catastrophic collapse by step 500.
Without explicit balancing pressure, first-order optimizers can't
resist the natural geometric attractor of representation collapse.

**Cell D (Muon + 0.00 aux):** task loss ~4.7, B-style router collapse.
Even Muon's orthogonal updates don't bypass the need for explicit
load balancing.

**Architecture details validated:**
- Loop embeddings (LIE) critical for breaking weight-tying symmetry
- `min(r, 7)` cap → can't safely loop past R=8 (hard wall)
- Sigmoid scoring with amplification factor γ ∈ {2.448, 2.827}
- DeepSeek V4's Sqrt(Softplus) replaces this in newer designs
- Auxiliary-loss-free balancing (SMEBU / sequence-wise score
  corrections) — DeepSeek V3 production validates

### Mathematical framework

- **Muon as LION-K under nuclear norm**: implicit spectral
  regularization, bounds ||W||_2
- **Maximal Update Parametrization (μP)** alignment: requires
  ||W||_2 = Θ(√(d_out/d_in)). Muon controls UPDATE spectral norm
  but not weight drift → need decoupled weight decay
- **Update scale adjustments per parameter type** for consistent RMS

### What we adopted

- **Cell C as our recipe** (Muon + 0.10 aux loss; will trial 0.01 and
  loss-free in v6)
- **Per-parameter update scaling** (Moonlight + this paper agree)
- **All stability requirements** for Muon scaling (weight decay,
  RMS alignment)

### Open questions surfaced

- Aux coefficient sweep at our 600M scale (paper used 363M)
- Loss-free balancing under recursion (paper used non-recurrent)
- Per-loop routing accounting (novel for sparse MoE + recurrence)

---

## 14. Small-language-model frontier survey

**Source:** User-supplied research summary (Nov 2026): "Building a
Cutting-Edge Small Language Model Around Your 363M Recurrent MoE"

### Comparative table compiled (frontier July-Dec 2025):

| model | active params | tokens | gsm8k |
|---|---|---|---|
| Qwen3-0.6B | 0.6B dense | distilled from larger | ~45% |
| SmolLM2-1.7B | 1.7B dense | 11T | strong |
| SmolLM3-3B | 3B dense, NoPE on every 4th | 11.2T | competitive |
| **Gemma 3 270M** | 270M dense, 256K vocab | 6T | ~35% |
| Llama 3.2 1B | 1B dense | 9T | distilled |
| OLMoE-1B-7B | 1.3B active / 6.9B total | 5.1T | strong |
| **Qwen3-30B-A3B** | 3B active / 30B total | many | competitive 30B |
| Phi-4-Mini | 200K vocab, GQA | synthetic-heavy | strong reasoning |
| Ministral 3 | cascade distillation | varies | competitive |
| MiniCPM5-1B | 24 layers, think/no-think | varies | controllable reasoning |
| LFM2 (350M-2.6B + MoE) | varies | 10-12T | LFM2-2.6B: 82% |
| Huginn-3.5B | 3.5B recurrent | 800B | pioneer |
| Ouro 1.4B/2.6B | recurrent LoopLM, R4 | 7.7T | beats Qwen3-4B |

### Frontier convergence (what everyone agrees on)

- Decoder-only, RMSNorm pre-norm (Gemma adds post-norm)
- GQA universal
- RoPE universal
- SwiGLU universal
- Tied embeddings at ≤4B
- QK-norm for stability
- WSD schedules dominant
- Massive over-training (1000-22000 tokens/param)
- Multi-stage data with quality annealing
- AdamW → Muon migration in progress

### Frontier splits (still debated)

- Shared vs no-shared experts (DeepSeek/Qwen1.5 = shared; Qwen3/
  OLMoE = no-shared)
- Fine-grained (many tiny) vs coarse (few large) experts
- Aux-loss vs loss-free balancing
- MTP draft models vs simple sampling

### Where our 363M was novel

- **Shared-expert MoE + depth recurrence + tied embeddings** — no
  released frontier model combines all three. Each ingredient
  validated separately.
- Risk: novel combo = no recipe to copy from
- Coarse experts (8/top-2) against fine-grained trend
- High aux coefficient (0.10) vs production norm (0.001-0.01)
- 3 unique attention blocks (more aggressive than Huginn's 8)

### What we adopted from this survey

- Frontier convergence defaults (all confirmed in OSRT-600M)
- Pretraining at thousands of tokens/param (target ≥1000)
- Multi-stage data + WSD + annealing
- Distillation IN pretraining (Tier 3 upgrade path)

### What this survey explicitly recommended we don't change

- **Keep shared experts** (DeepSeekMoE argument persuasive at sub-1B)
- **Don't expand vocab aggressively** (Gemma 3 270M's tax disaster)
- Add GQA + QK-Norm if missing
- Trial smaller aux coefficient (we'll do this in v6)

---

## 15. Other named references

Models / papers cited in design discussions but not deeply
summarized:

- **Mixtral 8×7B** (Jiang et al., arXiv 2401.04088): 47B total / 13B
  active, 2/8 routing, no shared experts
- **Phi-4-Mini** (Abouelenin et al., arXiv 2503.01743): 200K vocab,
  multilingual via mixture-of-LoRAs
- **Granite 4.0** (IBM Research): hybrid Mamba-Transformer
- **TinyLlama-1.1B**: trained on ~1T tokens, GPT-2-class quality
- **Mamba / Mamba2** (Gu & Dao): selective state-space model,
  alternative to attention
- **S4 / S5** (Gu et al.): structured state-space sequence models
- **Liquid Time-Constant (LTC)** Networks (Hasani et al.): LFM
  ancestry
- **CART** (Capps et al., 2024): context-anchored recurrent
  transformer, alternative K/V design
- **modded-nanoGPT** (KellerJordan): GPT-2 speedrun benchmark,
  reference for Muon + modern stack tuning
- **MiniCPM** (paper): popularized WSD schedule for small models
- **Chinchilla** (Hoffmann et al., 2022): scaling law (~20 tokens/
  param compute-optimal)
- **"Beyond Chinchilla-Optimal"** (arXiv 2401.00448): inference-
  efficient SLMs benefit from massive over-training (up to 10,000
  tokens/param)
- **OpenHermes-2.5** (Teknium): instruction dataset with system
  prompts (used for system_sft data collection)
- **UltraChat** (HuggingFaceH4): chat dataset (mostly no system
  prompts)
- **Llama 3** (Grattafiori et al.): 8B trained 15T tokens,
  reference dense baseline

---

## Document changelog

- **2026-06-07** — initial creation, captures all research integrated
  through DeepSeek-V4 / LFM2 / AlphaQ
- **2026-06-09** — §12 updated to the implemented math-first
  Nemotron/Cosmopedia pretrain mix (`train_config.py`); recorded the
  drop of codeparrot-clean + Wikipedia from the v6 base mix
