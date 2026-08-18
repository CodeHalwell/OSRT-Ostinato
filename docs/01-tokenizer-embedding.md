# Tokenizer & Embedding

> **v7 status.** The architecture this chapter describes is current, but its
> **`file:line` citations, parameter tables and config values were written
> against v6** and have not been regenerated. mHC references have been removed
> (roadmap §12.3); expert counts, vocab and param figures may still be stale.
> Regenerate counts with `scripts/compute_budget.py`; `src/osrt/` is ground
> truth where they disagree.


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
> The model's embedding matrix has **65,536 rows** (one per ID `0…65535`), but
> the tokenizer **currently shipped on disk only defines 32,768 tokens**
> (IDs `0…32767`). These are two different artifacts that *do not yet agree*:
> the network is built for a 64K vocabulary, the trained tokenizer is 32K. Keep
> this split in mind throughout — most of the surprises below flow from it.

---

## 2. Byte-level BPE and the vocabulary size

### What "byte-level BPE" means

The tokenizer is **byte-level Byte-Pair Encoding (BPE)**. Two ideas stacked:

- **Byte-level**: the base alphabet is the 256 possible byte values, not
  Unicode characters. Any input — English, Japanese, emoji, raw binary — is
  first reduced to its UTF-8 bytes, so there is **no out-of-vocabulary
  failure**: in the worst case a string falls back to its individual bytes. The
  pre-tokenizer is configured as `ByteLevel` with `add_prefix_space=false`,
  `use_regex=true` (`tokenizer/tokenizer.json:134-139`) — the GPT-2-style regex
  splits on word/number/punctuation boundaries before merging.
- **BPE**: starting from bytes, the trainer greedily merges the most frequent
  adjacent pair, over and over, until the vocabulary reaches the target size.
  Frequent sequences (`" the"`, `"def "`, `"tion"`) become single tokens;
  rare ones stay fragmented. The learned merge list lives under `"merges"`
  (`tokenizer/tokenizer.json:33006`).

The trainer is in `scripts/train_tokenizer.py`: it builds a
`Tokenizer(models.BPE())` with a `ByteLevel` pre-tokenizer and decoder
(`scripts/train_tokenizer.py:233-237`), trained on a ~2 GB sample of the
pre-training mix — 45 % FineWeb-Edu, 20 % OpenWebMath, 20 % CodeParrot-clean,
15 % Wikipedia (`scripts/train_tokenizer.py:83-108`). Training the tokenizer on
the *same* distribution the model will see keeps the merges optimal for that
data; the math-dense OpenWebMath share is deliberate — the model is math-first,
so LaTeX/symbols/numerics must tokenize well rather than fall back to near
byte-level (`train_tokenizer.py:76-82`). Note this is the **tokenizer's own**
training corpus, which is distinct from (and frozen relative to) the model's
pre-training data mix; the latter is configured in `train_config.py`
(`PretrainConfig.phases`) and has since moved to a FineWeb-Edu + NVIDIA Nemotron
(math/code/STEM, gated) + Cosmopedia blend.

### Why 65,536 (and why the on-disk 32,768 is a discrepancy)

The model's declared vocabulary is **65,536 = 2¹⁶** (`vocab_size=65536`,
`src/osrt/presets.py:27`). A power of two keeps the embedding/LM-head GEMMs
hardware-aligned, and 64K is a deliberate **middle-ground** vocabulary:

- Big enough that common words/code idioms are single tokens (good
  *compression* → fewer tokens per document → more text per training step and
  per context window).
- Small enough that the embedding matrix does not dominate the parameter
  budget. This is the **"embedding tax"** lesson (see §7): for a small model,
  every parameter spent on vocabulary is a parameter *not* spent on the
  reasoning blocks.

**Discrepancy — the shipped tokenizer is 32K, not 64K.** The actual
`tokenizer/tokenizer.json` tops out at ID `32767` — its highest three entries
are `"Ġcaching": 32765, "Ġhubs": 32766, "Ġhometown": 32767`
(`tokenizer/tokenizer.json:33002-33004`). That is a **32,768-token** vocabulary.
Consistently, `scripts/train_tokenizer.py` defaults to `--vocab-size 32768`
(`scripts/train_tokenizer.py:432`) and its docstring calls it a "32K BPE
tokenizer." So:

| artifact | declared vocab |
|---|---|
| model preset (`presets.py:27`) | **65,536** |
| `OSRTConfig` default (`config.py:46`) | 32,768 |
| **tokenizer on disk** (`tokenizer.json`) | **32,768** |
| `train_tokenizer.py` default (`:432`) | 32,768 |

The network will happily run with the 32K tokenizer — every ID it emits is
`< 32768 < 65536`, so it indexes a valid embedding row — but **IDs 32768…65535
of the embedding are dead weight** (never produced by the tokenizer, never a
training label). The 64K-vs-32K gap is the natural home for the *missing*
special tokens discussed in §4. The 64K embedding must be retained (the model is
sized for it), but to actually *use* a 64K vocabulary the tokenizer would need to
be retrained at `--vocab-size 65536`.

---

## 3. Special tokens

"Special tokens" are reserved IDs that don't come from BPE merges — they carry
structural meaning (turn boundaries, reasoning markers, padding). They are
stored as `added_tokens` with `special: true`
(`tokenizer/tokenizer.json:5-132`).

### What is *actually* on disk

Inspecting `tokenizer/tokenizer.json`, the `added_tokens` array contains exactly
**14 entries, IDs 0–13** (`tokenizer.json:6-131`). The on-disk order — note that
**`unk` is ID 3**, *before* the FIM tokens, and the order is verified against the
file, not assumed:

| token | id | role | on disk? |
|---|---|---|---|
| `<\|padding\|>` | 0 | PAD — fills batched sequences to equal length; masked in loss | ✓ |
| `<\|begin_of_text\|>` | 1 | BOS — prepended to every sequence (`add_bos_token=true`, `tokenizer_config.json:8`) | ✓ |
| `<\|end_of_text\|>` | 2 | EOS — end of sequence / generation stop | ✓ |
| `<\|unknown\|>` | 3 | unk — fallback (rarely hit; byte-level BPE almost never needs it) | ✓ |
| `<\|fim_prefix\|>` | 4 | FIM prefix marker (fill-in-the-middle, for code infilling) | ✓ |
| `<\|fim_middle\|>` | 5 | FIM middle marker | ✓ |
| `<\|fim_suffix\|>` | 6 | FIM suffix marker | ✓ |
| `<\|think\|>` | 7 | reasoning block **open** | ✓ |
| `<\|/think\|>` | 8 | reasoning block **close** | ✓ |
| `<\|answer\|>` | 9 | answer block **open** | ✓ |
| `<\|/answer\|>` | 10 | answer block **close** | ✓ |
| `<\|user\|>` | 11 | user turn | ✓ |
| `<\|assistant\|>` | 12 | assistant turn | ✓ |
| `<\|system\|>` | 13 | system prompt | ✓ |
| `<\|end_turn\|>` | 14 | turn separator (ChatML-style) | ✗ **not on disk** |
| `<\|tool_call\|>` | 15 | tool invocation open | ✗ **not on disk** |
| `<\|/tool_call\|>` | 16 | tool invocation close | ✗ **not on disk** |
| `<\|tool_result\|>` | 17 | tool result open | ✗ **not on disk** |
| `<\|/tool_result\|>` | 18 | tool result close | ✗ **not on disk** |
| `<\|image\|>` | 19 | reserved for vision retrofit | ✗ **not on disk** |
| `<\|audio\|>` | 20 | reserved for future audio | ✗ **not on disk** |

The first 14 also appear at the top of the `"vocab"` map with the same IDs
(`tokenizer.json:236-250`), and the HF config wires the four scalar roles:
`bos=<\|begin_of_text\|>`, `eos=<\|end_of_text\|>`, `pad=<\|padding\|>`,
`unk=<\|unknown\|>` (`tokenizer_config.json:4-7`), with the rest declared as
`additional_special_tokens` (`special_tokens_map.json:6-17`). This matches the
hard-coded `special_tokens` list the trainer writes
(`scripts/train_tokenizer.py:240-255`).

### What the role tokens do

- **FIM (4–6)** — *fill-in-the-middle*. For code training, a file is rearranged
  as `prefix · suffix · middle` so the model learns to infill a hole given both
  sides, not just left-to-right. The three markers delimit the spans.
- **think / answer (7–10)** — the reasoning contract. The model is trained to
  emit a private chain-of-thought between `<\|think\|>…<\|/think\|>` and the
  user-facing answer between `<\|answer\|>…<\|/answer\|>`. Separating the two as
  *tokens* lets training and decoding treat reasoning and answer differently
  (e.g. score only the answer span).
- **user / assistant / system (11–13)** — conversational role markers.

### The chat template (open-only-tag convention)

This project uses an **open-only-tag** style for role markers — a role tag opens
a turn, and the *next* role tag (or `<\|end_turn\|>`) implicitly closes it. There
are no `<\|/user\|>` / `<\|/assistant\|>` closers. The reasoning/answer blocks,
by contrast, *do* have explicit close tags (`<\|/think\|>`, `<\|/answer\|>`),
because their spans must be unambiguous. A single turn:

```
<|system|>{system_message}
<|user|>{user_question}
<|assistant|><|think|>{reasoning}<|/think|><|answer|>{final_answer}<|/answer|>
```

This is the project's documented convention (`ARCHITECTURE.md:264-271`). The
`<|think|>…<|/think|><|answer|>…<|/answer|>` structure is verified by the
tokenizer's own round-trip self-test, which encodes exactly this string
(`scripts/train_tokenizer.py:398-401`).

> **Discrepancy — ARCHITECTURE.md §3.2 claims "IDs 21-31 reserved, real vocab
> begins at id 32."** That is **false against the file**. On disk, ID 14 is the
> literal character `"!"`, ID 15 is `"\""`, … the printable-ASCII run starts
> *immediately* after the 14 special tokens (`tokenizer.json:251-298`). There is
> **no gap of reserved IDs** at 14–20 or 21–31; ordinary byte/character tokens
> occupy those slots. Trust the file: the real (non-special) vocabulary begins at
> **ID 14**, not 32.

---

## 4. Consequence of the missing tokens (a real gotcha)

IDs 14–20 in the table above — `end_turn`, `tool_call`/`/tool_call`,
`tool_result`/`/tool_result`, `image`, `audio` — are the **v6 contract** the
architecture *intends* to support, but they are **not in `tokenizer.json`**. This
is not a cosmetic gap; it changes behavior:

- A special token is atomic only if the tokenizer knows it. Because the string
  `<|tool_call|>` is **not** a registered token, `tok.encode("<|tool_call|>")`
  does **not** return a single ID — it falls through to byte-level BPE and gets
  **shredded into fragments** (`<`, `|`, `tool`, `_`, `call`, `|`, `>`, or
  similar). The model then sees a meaningless splatter of subwords instead of one
  clean structural marker.
- **Tool-use and multimodal training/inference will silently mis-tokenize**
  until 14–20 are added. "Silently" is the dangerous part: nothing errors; the
  tokens just don't mean what the data pipeline assumes, and the model can't
  learn a crisp tool-call boundary.
- Basic chat (`system`/`user`/`assistant`/`think`/`answer`) is unaffected —
  those tokens (7–13) *are* on disk.

There's a second wrinkle from §2's 32K/64K split. On disk, IDs 14–20 are
**already occupied** by printable-ASCII tokens (`"!"`, `"\""`, …). So the seven
contract tokens **cannot simply be slotted in at 14–20** without renumbering the
entire vocabulary. Their natural home is the **currently-unused 32768–65535
range** — exactly the dead band that exists because the embedding is 64K but the
tokenizer is 32K. Adding them there would consume some of that headroom *and*
keep every existing ID stable. (Whether the 32K/64K gap was left deliberately as
that headroom is not documented in the code; this chapter only states what the
files show.)

**Practical guard:** before any tool-use or vision training, retrain/extend the
tokenizer to register 14–20 (or place them in the 32768+ range) and add a
contract test asserting e.g. `tok("<|end_turn|>")` returns a single ID
(`ARCHITECTURE.md:229-230` recommends exactly this).

---

## 5. The embedding matrix

### Shape and the tied LM head

The embedding is a plain `nn.Embedding`:

```python
# src/osrt/model.py:1237
self.embedding = nn.Embedding(config.vocab_size, config.dim)   # 65536 × 1536
```

So the matrix is **65,536 × 1,536** (`vocab_size × dim`). The forward pass uses
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
all 65,536 rows of the embedding and yields one logit per token — the inverse of
the lookup, using the very same numbers. The class docstring states the intent
plainly: *"LM head is weight-tied to embeddings (via `F.linear` with
`embedding.weight`)"* (`src/osrt/model.py:1569`).

#### Why tie, and the parameter saving

If the input embedding and the output projection were **untied**, you would store
**two** `65,536 × 1,536` matrices ≈ **201 M** parameters. Tying stores **one** and
reuses it, **saving ≈ 100 M parameters** (`65,536 × 1,536 = 100,663,296`). For a
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

`real_vocab_size` is also `65536` in the preset (`presets.py:28`). This slice is
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

- **Token-embedding matrix**: `65,536 × 1,536 = 100,663,296` parameters. This is
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
> reads "65,536 × 1,536 = 100,690,944" is arithmetically wrong; the true product
> is 100,663,296.

### Share of the model — the embedding tax

Against the preset's **~601M physical** parameters
([`00-overview.md`](00-overview.md), from `compute_budget.py`), the token
embedding is **≈ 16.6 %** of the model (`100,663,296 / 601,444,393`). (The
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
  **embedding** is a single `65,536 × 1,536` matrix (`model.py:1237`),
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
