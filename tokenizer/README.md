# OSRT-Ostinato tokenizer

49,184 real tokens · 49,280 embedding rows · single-digit number tokenization

## Lineage — read this before changing anything

Derived from **[SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)**
(49,152 vocab, Apache-2.0), extended with the 32 OSRT chat-contract special
tokens and padded to a multiple of 128 for tensor cores.

**This is not a cosmetic detail.** Base rows `0..49,151` align byte-for-byte
with SmolLM2's, which makes **SmolLM2-1.7B a same-tokenizer teacher** and is
the only reason logit-level KD is available to v7 at all. Anyone reaching for
a teacher must use that family, or the vocabularies will silently disagree.

That alignment survives **adding** tokens — the teacher never emits ours, so
those logits are simply absent. It does **not** survive retraining the merges,
reordering, or removing anything. Extend-only.

## Why this tokenizer (gate G2, roadmap §16)

The v6 65,536 BPE failed in a way that is easy to get backwards: it did not
over-split numbers, it made them **atomic**.

| property | v6 custom | this |
|---|---|---|
| 1 / 2 / 3-digit numbers | 100 / 100 / 96.7% **single-token** | one token per digit |
| context consistency | 75% | **100%** |
| place value | `1234567 → 123·4567` | `1·2·3·4·5·6·7` |
| tied embedding @ dim 1536 | 100.7M | **75.7M** (−25.0M, and it is *active*) |

Atomic numbers meant the model had to memorise ~1000 unrelated symbols rather
than compose digits — and GSM8K arithmetic lives almost entirely in that 1–3
digit range. Cost of the swap: **+6.0% tokens** on the real pretraining mix
(measured, `scripts/probe_tokenizer_fertility.py`), concentrated in the math
slice at +10.4%. A synthetic number-dense sample suggests +36%; that figure is
a worst case and should not be quoted.

## 107 free slots

96 padding rows (allocated to reach 49,280) plus 11 unused `<|reserved_N|>`
placeholders. Filling them costs **zero parameters**. Whether they buy
measurable fertility on this mix is **not yet measured**.

**Freeze before the trunk run.** Vocabulary added after training begins leaves
untrained embedding rows in a checkpoint representing months of compute. v7
trains from scratch, so the window is now — and so is the cost of error.

## Rebuild

    python scripts/build_tokenizer_v7.py --out tokenizer

Deterministic given the same base. Verify with:

    PYTHONPATH=src python scripts/probe_tokenizer_math.py tokenizer
