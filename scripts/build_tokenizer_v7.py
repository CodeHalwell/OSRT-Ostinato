"""Build the v7 tokenizer: SmolLM2 base + the OSRT chat contract.

Why SmolLM2's tokenizer (roadmap gate G2, decided 2026-08-12):

  * Single-digit number tokenization — 100% context consistency and exact
    place-value alignment. The v6 custom 65,536 BPE made 1-3 digit numbers
    ATOMIC (100/100/96.7% single-token), so the model had to memorise ~1000
    unrelated symbols instead of composing digits, and only 75% of numbers
    tokenized identically across contexts.
  * 49,152 base vocab -> ~25M fewer parameters than v6, and those are ACTIVE
    parameters (the head is tied), so it is also a decode win: the LM-head
    matmul is the largest single op in a batch-1 decode step.
  * Apache-2.0, and SmolLM2-1.7B remains available as a same-tokenizer
    teacher should logit-KD be adopted.

Measured with scripts/probe_tokenizer_math.py. Note the fertility cost:
single-digit tokenization is ~36% more tokens on number-dense text. Measure on
the real pretraining mix before locking the token budget.

Usage:
    PYTHONPATH=src uv run python scripts/build_tokenizer_v7.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

BASE = "HuggingFaceTB/SmolLM2-1.7B"
NAME = "OSRT-Ostinato"
PAD_TO_MULTIPLE = 128  # tensor-core friendly embedding rows

CORE = {
    "bos_token": "<|begin_of_text|>",
    "eos_token": "<|end_of_text|>",
    "pad_token": "<|padding|>",
    "unk_token": "<|unknown|>",
}
ADDITIONAL = [
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
    "<|think|>", "<|/think|>", "<|answer|>", "<|/answer|>",
    "<|user|>", "<|assistant|>", "<|system|>", "<|end_turn|>",
    "<|tool_call|>", "<|/tool_call|>", "<|tool_result|>", "<|/tool_result|>",
    "<|image|>", "<|audio|>",
] + [f"<|reserved_{i}|>" for i in range(21, 32)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tokenizer_v7")
    args = ap.parse_args()

    tk = AutoTokenizer.from_pretrained(BASE)
    before = len(tk)

    added = tk.add_special_tokens({
        **CORE,
        "additional_special_tokens": ADDITIONAL,
    })
    after = len(tk)
    real = after
    padded = ((real + PAD_TO_MULTIPLE - 1) // PAD_TO_MULTIPLE) * PAD_TO_MULTIPLE

    out = Path(args.out)
    tk.name_or_path = NAME
    tk.save_pretrained(out)
    (out / "osrt_vocab.json").write_text(json.dumps({
        "name": f"{NAME} tokenizer",
        "base": BASE,
        "base_licence": "Apache-2.0",
        "kd_teacher": BASE,
        "base_vocab": before,
        "special_added": added,
        "real_vocab_size": real,
        "vocab_size": padded,
        "pad_to_multiple": PAD_TO_MULTIPLE,
    }, indent=2) + "\n")

    print(f"base {BASE}: {before:,} -> +{added} special -> {real:,}")
    print(f"  real_vocab_size = {real:,}")
    print(f"  vocab_size      = {padded:,}  (padded to x{PAD_TO_MULTIPLE})")
    print(f"  tied embed @1536 = {padded * 1536:,}  "
          f"({(padded * 1536 - 65536 * 1536) / 1e6:+.1f}M vs v6)")
    print(f"  saved -> {out}/")


if __name__ == "__main__":
    main()
