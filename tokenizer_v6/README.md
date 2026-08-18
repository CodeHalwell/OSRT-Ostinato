# v6 tokenizer (65,536 byte-level BPE) — retained, not current

Kept only so v6 checkpoints can still be loaded — notably roadmap gate **G8**,
which trains a speculative-decoding drafter against the frozen v6 model.

**Do not use it for v7.** Superseded at gate G2 (2026-08-12). Measured defects
(`scripts/probe_tokenizer_math.py`):

| property | v6 (this) | v7 (`../tokenizer`) |
|---|---|---|
| 1–3 digit numbers | **atomic** (100 / 100 / 96.7% single-token) | one token per digit |
| context consistency | **75%** | **100%** |
| place value | frequency-merged: `1234567 -> 123·4567` | aligned: `1·2·3·4·5·6·7` |
| tied embed @1536 | 100.7M | 75.7M |

Atomic numbers were the real problem: the model had to memorise ~1000
unrelated symbols rather than compose digits, and GSM8K arithmetic lives almost
entirely in that 1–3 digit range.
