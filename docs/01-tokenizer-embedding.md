# Tokenizer & Embedding

> **Updated to v7 (2026-09-01).** Config values, parameter counts, expert
> layout, tokenizer and optimizer recipe below are the committed v7 shape
> (`OSRT_V7`: 968,468,355 physical / 263,035,779 active). `file:line`
> citations drift as the code moves — `src/osrt/` is ground truth and
> `scripts/compute_budget.py` is the only source for any count. Passages that
> explain a *v6* choice are marked as such where they survive. Decisions and
> open gates: `specs/2026-08-11-v7-roadmap.md` §14, §16, §19.


*Chapter 1 of the `docs/` architecture series. Read [`00-overview.md`](00-overview.md)
first. Like every chapter, this one is grounded in the real artifacts on disk
(`tokenizer/`, `src/osrt/`) and cites `file:line`; where the code or the on-disk
tokenizer disagrees with the older spec (`ARCHITECTURE.md`), this chapter trusts
the artifacts and flags the drift.*

---

## 1. Purpose — the input/output interface

A language model never sees text. It sees integers, and it emits a probability
distribution over integers. Two components do that translation:

1. **The tokenizer** turns a UTF-8 byte string into a list of integer **token
   IDs** (encode), and turns IDs back into a string (decode). It lives entirely
   *outside* the neural network — it is a deterministic lookup-and-merge table,
   not a learned tensor. Artifact: `tokenizer/tokenizer.json`.

2. **The embedding matrix** is the first learned layer. It maps each token ID to
   a `dim`-dimensional vector (the "loop-0" input the recursive stack consumes),
   and — because it is **weight-tied** — the *same* matrix is reused at the very
   end as the LM head that turns the final hidden state back into a distribution
   over token IDs. Artifact: `model.embedding` in `src/osrt/model.py:1237`.

So the embedding is literally the bridge in *both* directions: IDs → vectors at
the front, vectors → ID-logits at the back. Everything in between (attention,
MoE, recursion) operates purely on continuous vectors.

> ### The one fact that organizes this chapter
> The model's embedding matrix has **49,280 rows** (one per ID `0…49279`), of which
> the tokenizer **defines 49,184 real tokens**; the last 96 rows are tensor-core padding
> (IDs `0…32767`). These are two different artifacts that *do not yet agree*:
> the network is built for a 64K vocabulary, the trained tokenizer is 32K. Keep
> this split in mind throughout — most of the surprises below flow from it.

---

## 2. The OSRT-Ostinato tokenizer

v7 does not use a custom-trained vocabulary. It uses **SmolLM2's 49,152-token
byte-level BPE** (HuggingFaceTB, Apache-2.0) extended with the 32 OSRT special
tokens below — **49,184 real tokens**, padded to **49,280 embedding rows** (a
multiple of 128 for tensor cores). The build is reproducible:
`scripts/build_tokenizer_v7.py`; the lineage is in `tokenizer/README.md` and
`tokenizer/osrt_vocab.json`.

### Why this one — the digit finding (roadmap §16)

The decision was made on measurement, not reputation, and the measurement
inverted the obvious guess. v6's custom 65,536 BPE did not *over-split*
numbers — it made them **atomic**:

| property | v6 custom 65,536 | Ostinato (SmolLM2 base) |
|---|---|---|
| 1 / 2 / 3-digit numbers | **100 / 100 / 96.7% single-token** | one token per digit |
| context consistency (11 contexts) | 75% | **100%** |
| place value | frequency-merged: `1234567 → 123·4567` | `1·2·3·4·5·6·7` |
| tied embedding @ dim 1536 | 100.7M | **75.7M** — and it is *active* |

Atomic numbers meant the model had to memorise ~1000 unrelated symbols rather
than compose digits — and GSM8K arithmetic lives almost entirely in that 1–3
digit range. Cost of the swap, measured on the real pretraining mix
(`scripts/probe_tokenizer_fertility.py`): **+6.0% tokens**, concentrated in the
math slice at +10.4%. A synthetic number-dense sample gives +36%; that figure
is a worst case and should not be quoted.

Alternatives measured: SmolLM3 and DeepSeek-V3 (128K, 3-digit **left-to-right**
groups — consistent but misaligned, `34567 → 345·67`), Qwen3 (151K,
single-digit but +132M active), Mistral (32K). SmolLM2 was the only candidate
that won on arithmetic, parameter count *and* decode cost at once.

### What "byte-level BPE" means

Bytes, not characters, are the atoms, so any UTF-8 string tokenizes with no
`<unk>` fallback. Merges are learned pairs of byte sequences. The base 49,152
merges are SmolLM2's, unchanged — which is what keeps **SmolLM2-1.7B a
same-tokenizer teacher** (rows `0…49,151` align byte-for-byte). That alignment
survives *adding* tokens; it does not survive retraining the merges. The
tokenizer is **extend-only**.

### The 107 free slots

96 padding rows (49,184 → 49,280) plus 11 `<|reserved_N|>` placeholders can be
filled at **zero parameter cost**. Whether they buy measurable fertility on this
mix is unmeasured. **Freeze before the trunk run** — vocabulary added after
training begins leaves untrained rows in a checkpoint worth months of compute.

## 3. Special tokens

All 32 are present and pinned. `src/osrt/tokenizer_contract.py` hard-validates
the real size (49,184) and every structural id at launch — a swapped or
half-built tokenizer cannot start a run, because a checkpoint trained at one
vocabulary cannot load at another.

| token | id | role |
|---|---|---|
| `<\|begin_of_text\|>` | 49152 | bos |
| `<\|end_of_text\|>` | 49153 | eos |
| `<\|padding\|>` | 49154 | pad |
| `<\|unknown\|>` | 49155 | unk |
| `<\|fim_prefix\|>` | 49156 | fill-in-middle |
| `<\|fim_middle\|>` | 49157 | fill-in-middle |
| `<\|fim_suffix\|>` | 49158 | fill-in-middle |
| `<\|think\|>` | 49159 | opens reasoning |
| `<\|/think\|>` | 49160 | closes reasoning |
| `<\|answer\|>` | 49161 | opens the final answer |
| `<\|/answer\|>` | 49162 | closes the final answer |
| `<\|user\|>` | 49163 | role |
| `<\|assistant\|>` | 49164 | role |
| `<\|system\|>` | 49165 | role |
| `<\|end_turn\|>` | 49166 | turn boundary |
| `<\|tool_call\|>` | 49167 | tool use |
| `<\|/tool_call\|>` | 49168 | tool use |
| `<\|tool_result\|>` | 49169 | tool use |
| `<\|/tool_result\|>` | 49170 | tool use |
| `<\|image\|>` | 49171 | reserved modality |
| `<\|audio\|>` | 49172 | reserved modality |
| `<\|reserved_21\|>` | 49173 | reserved slot |
| `<\|reserved_22\|>` | 49174 | reserved slot |
| `<\|reserved_23\|>` | 49175 | reserved slot |
| `<\|reserved_24\|>` | 49176 | reserved slot |
| `<\|reserved_25\|>` | 49177 | reserved slot |
| `<\|reserved_26\|>` | 49178 | reserved slot |
| `<\|reserved_27\|>` | 49179 | reserved slot |
| `<\|reserved_28\|>` | 49180 | reserved slot |
| `<\|reserved_29\|>` | 49181 | reserved slot |
| `<\|reserved_30\|>` | 49182 | reserved slot |
| `<\|reserved_31\|>` | 49183 | reserved slot |

IDs are contiguous from 49,152 because the specials are appended to the
SmolLM2 base in the order above. `eos` is 49,153 and `pad` is 49,154; both are
wired as the tokenizer's designated tokens and picked up by `train_main`.

### The chat contract (open-only-tag convention)

```
<|system|>{persona}<|user|>{question}<|assistant|><|think|>…<|/think|><|answer|>N<|/answer|><|end_turn|>
```

Round-trips byte-identically through the tokenizer, and numbers inside the
answer tags split per digit: `<|answer|>391<|/answer|>` →
`<|answer|>·3·9·1·<|/answer|>`. That is the property the reward extraction
and the arithmetic both depend on.

## 4. Consequence for the model

There is no "missing tokens" gotcha in v7 — that section described v6's 32K
on-disk artefact and is retired. What remains true: **the embedding matrix has
49,280 rows, the tokenizer defines 49,184.** Rows 49,184–49,279 are padding —
never looked up by any input, and sliced out of every logit before the loss
(`real_vocab_size`). They sit at init forever and cost nothing but memory.

## 5. The embedding matrix

### Shape and the tied LM head

The embedding is a plain `nn.Embedding`:

```python
# src/osrt/model.py:1237
self.embedding = nn.Embedding(config.vocab_size, config.dim)   # 49280 × 1536
```

So the matrix is **49,280 × 1,536** (`vocab_size × dim`). The forward pass uses
it as a lookup — one row per input token — to produce the loop-0 hidden state
(`src/osrt/model.py:1351`, `x = self.embedding(input_ids)`).

It is **weight-tied** with the LM head. There is **no separate `lm_head`
parameter** anywhere in the model; the output projection is computed directly
from the embedding weight:

```python
# src/osrt/model.py:1658  — the tied LM head
logits = F.linear(hidden, self.model.embedding.weight)
```

`F.linear(x, W)` computes `x @ Wᵀ`, so this projects the final `dim`-vector onto
all 49,280 rows of the embedding and yields one logit per row (sliced to 49,184 real tokens) — the inverse of
the lookup, using the very same numbers. The class docstring states the intent
plainly: *"LM head is weight-tied to embeddings (via `F.linear` with
`embedding.weight`)"* (`src/osrt/model.py:1569`).

#### Why tie, and the parameter saving

If the input embedding and the output projection were **untied**, you would store
**two** `49,280 × 1,536` matrices ≈ **151 M** parameters. Tying stores **one** and
reuses it, **saving ≈ 76 M parameters** (`49,280 × 1,536 = 75,694,080`). For a
~601 M-parameter model that is the difference between spending ~16.6 % vs ~33 % of
the budget on the I/O interface alone — params that are far better spent on the
recursive MoE blocks.

Tying is also well-motivated, not just thrifty: the input and output spaces are
the *same* token space, so sharing a representation is a sensible inductive bias
(input "what does token *t* mean" and output "how likely is token *t*" are dual
views of the same vector). The auxiliary per-loop heads and the Multi-Token
Prediction heads reuse this same tied weight as well — no extra projection params
(`ARCHITECTURE.md:749-777`).

> The docstring at `src/osrt/model.py:1570` says tying *"saves ~50M params … for
> 32K×1536 embedding."* That figure is for the **32K** case
> (`32768 × 1536 ≈ 50M`); for the **64K** preset that this model actually builds,
> the saving is **~100M**. Same logic, different vocab.

### The output-side slice: `real_vocab_size`

When computing the training loss, the logits are sliced to `real_vocab_size`
before cross-entropy:

```python
# src/osrt/model.py:1675
shift_logits = logits[..., :-1, :self.config.real_vocab_size]
```

`real_vocab_size` is `49184` in the preset — 96 below `vocab_size`. This slice is
the hook that *would* let you compute logits over the full padded vocab but only
score the "real" prefix of it — useful if the embedding were padded beyond the
true vocabulary for alignment. Here they are equal, so the slice is a no-op, but
it documents the intended separation between *physical* vocab (embedding rows) and
*scored* vocab.

### Initialization

The embedding is initialized by the shared `_init_weights` hook:

```python
# src/osrt/model.py:1222-1228
elif isinstance(module, nn.Embedding):
    custom_std = getattr(module, "_osrt_init_std", None)
    nn.init.normal_(module.weight, mean=0.0,
                    std=custom_std if custom_std is not None else std)
```

where `std = self.config.initializer_range` (`model.py:1217`), and
`initializer_range` defaults to **0.02** (`config.py:256`). The token embedding
carries no `_osrt_init_std` override, so it is drawn from a plain **Gaussian,
mean 0, std 0.02**. (The per-loop *loop embeddings* — a different tensor, see §6
— *do* set `_osrt_init_std = 0.1` via `loop_embedding_init_std`,
`model.py:213-214` / `config.py:248`.)

> **Discrepancy — ARCHITECTURE.md §4.2** specifies *"truncated normal,
> std = 1/√1536 ≈ 0.0255"* and *"divide logits by √1536 at output for μP."* Both
> diverge from the code: the implementation uses **non-truncated `normal_` with
> std 0.02** (not 0.0255), and the tied LM head is a **bare `F.linear` with no
> √1536 logit scaling** (`model.py:1658`). Trust the code; the μP logit-scale is
> *not* implemented.

---

## 6. How the embedding feeds the recursion

The embedding output is the **loop-0 input** to the recursive stack. In
`OSRTModel.forward`:

```python
# src/osrt/model.py:1351-1355
x = self.embedding(input_ids)
```

Two things to notice:

1. **It is the only entry point.** Every one of the 6 recursive loops over the 3
   physical blocks (18 effective layers) starts from this single embedding
   lookup; the blocks then transform `x` in place. The embedding therefore sets
   the *scale and geometry* of the entire residual stream — which is why its init
   and its no-weight-decay treatment matter.
2. **Single residual stream.** The embedding output feeds one `dim`-vector
   per token straight into the recursive blocks. (v6 expanded this into a
   4-channel mHC stream; mHC was removed in v7 — roadmap §12.3.) The
   loop-conditioning that breaks symmetry between passes lives in
   [`05-recursion.md`](05-recursion.md).

For the full depth-recurrence story — loop embeddings, per-pass adapters, the
tied LM head applied at intermediate loops — see [`05-recursion.md`](05-recursion.md)
and [`06-heads-and-losses.md`](06-heads-and-losses.md).

---

## 7. Parameter cost

Counts come from `scripts/compute_budget.py`, which instantiates the canonical
`OSRT_605M_A288M` preset on a meta device and sums real parameters
(`scripts/compute_budget.py:50-58`).

### The exact numbers

- **Token-embedding matrix**: `49,280 × 1,536 = 75,694,080` parameters (`compute_budget.py` reports 75,721,728 for the category, which also catches 18 small per-layer vectors). This is
  the one tensor that serves both the input lookup and (tied) the LM head.
- **`compute_budget.py` "embedding" line**: **100,690,944**. This is *slightly
  larger* than the matrix itself — by exactly **27,648 = 3 × 6 × 1,536**.

#### Why the budget number is bigger than the matrix (resolved, not hand-waved)

`100,690,944 − 100,663,296 = 27,648 = num_blocks(3) × recursive_loops(6) × dim(1536)`
— the size of the per-block **loop embeddings**. The loop-embedding tensor
(`nn.Embedding(recursive_loops, dim)`, `model.py:213`) lives inside `MoELayer`,
which lives inside each `RecursiveBlock`, which is instantiated `num_blocks=3`
times (`model.py:1250-1252`) — so there are `3 × 6 × 1536 = 27,648` loop-embedding
params in total.

They land in the **"embedding"** budget bucket only because of category
*ordering*: `compute_budget.py:29-40` checks `"embedding" in name` **before**
`"loop_emb" in name`, and the parameter is named `…loop_embeddings.weight`, whose
name contains the substring `embedding`. So the first rule wins and the dedicated
`loop_emb` bucket never fires. It is a *bucketing* artifact, not a second copy of
the vocabulary — the actual token-embedding matrix is exactly **100,663,296**.

> Take-away: cite **100,663,296** for the token-embedding/tied-LM-head matrix.
> The **100,690,944** figure is `compute_budget.py`'s embedding line, which also
> sweeps in the 3× loop embeddings (`+27,648`). The product in some notes that
> (v6 history — the v7 matrix is 49,280 × 1,536 = 75,694,080.) The v6 note that
> read "65,536 × 1,536 = 100,690,944" was arithmetically wrong; the true product
> is 100,663,296.

### Share of the model — the embedding tax

Against the preset's **~601M physical** parameters
([`00-overview.md`](00-overview.md), from `compute_budget.py`), the token
embedding is **≈ 7.8 %** of the model (`75,694,080 / 968,468,355`) — down from 16.6% in v6, because the vocab shrank while the experts grew. (The
"embedding" *budget line*, 100,690,944, is ~16.7 %; ARCHITECTURE.md §4.1 quotes
16.9 % against an older total. The spread is just numerator/denominator choice.)

This ~16–17 % is the **deliberate "embedding tax" target**. The lesson from
recent small models:

- **LFM2-700M** showed that small models should put their parameters into
  *blocks* (~85 %), not vocabulary — capacity in the reasoning machinery pays
  off more than a bigger lookup table.
- **Gemma-3-270M** is the cautionary tale: a large vocabulary on a tiny model
  left **~63 %** of parameters tied up in the embedding — an "embedding
  catastrophe" where most of the model is a dictionary, not a thinker.

OSRT's ~16.6 % sits firmly in the LFM2 regime, and **tying** is what keeps it
there: an untied 64K×1536 head would roughly *double* the embedding tax. The
choice of a 64K (not 128K+) vocabulary is the other half of the same bet.

> **Caveat on totals**: do **not** read `model.py:23`'s docstring figure
> (`362,720,259`) as the model total — that docstring describes the **default
> `OSRTConfig`** (vocab 32,768, MHA, no MTP), *not* the 605M preset. For the
> preset, regenerate with `PYTHONPATH=src python scripts/compute_budget.py`
> (`compute_budget.py:14`).

---

## Summary

- The **tokenizer** is byte-level BPE (`train_tokenizer.py:233-237`); the
  **embedding** is a single `49,280 × 1,536` matrix,
  **weight-tied** as the LM head (`model.py:1658`), saving ~100M params.
- **Embedding tax ≈ 16.6 %** of ~601M physical — the LFM2 "params into blocks"
  regime, far from Gemma-3-270M's ~63 % embedding catastrophe.
- **Live discrepancies to fix** (code/file vs `ARCHITECTURE.md`):
  1. Tokenizer on disk is **32K**, model is built for **64K** — IDs 32768–65535
     are dead embedding rows until the tokenizer is retrained.
  2. Special tokens **14–20** (tool-use + multimodal) are **not on disk**; those
     strings byte-BPE into fragments → silent mis-tokenization for tool/vision.
  3. There are **no reserved IDs 21–31**; real (non-special) tokens start at
     **ID 14**, contra ARCHITECTURE.md §3.2.
  4. Embedding init is **`normal_(std=0.02)`**, not truncated-normal 0.0255; and
     there is **no √1536 μP logit scaling**, contra ARCHITECTURE.md §4.2.
