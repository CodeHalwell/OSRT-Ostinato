"""Measure tokenizer fertility on the REAL pretraining mix (roadmap gate G2).

Fertility decides how much text a fixed token budget buys. Since tokens are
this project's binding constraint (roadmap §3), a tokenizer that represents
arithmetic better but inflates token count is trading one scarce thing for
another, and the exchange rate has to be measured on the actual mix — not on a
synthetic sample, which over-weights numbers and exaggerates the penalty.

Usage:
    uv run python scripts/probe_tokenizer_fertility.py [--docs 200]
"""
from __future__ import annotations

import argparse

from transformers import AutoTokenizer

# The pretrain mix, from train_config.py PretrainConfig "core" phase.
MIX = [
    ("fineweb-edu", "HuggingFaceFW/fineweb-edu", None, 0.40),
    ("nemotron-cc-math", "nvidia/Nemotron-CC-Math-v1", "4plus", 0.25),
    ("nemotron-code", "nvidia/Nemotron-Pretraining-Code-v2",
     "Synthetic-Question-Answering", 0.20),
    ("cosmopedia-web", "HuggingFaceTB/cosmopedia", "web_samples_v2", 0.15),
]
TEXT_KEYS = ("text", "content", "raw_content", "problem", "output")


def text_of(rec):
    for k in TEXT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    for v in rec.values():
        if isinstance(v, str) and len(v) > 200:
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--tokenizers", nargs="*",
                    default=["tokenizer_v6", "tokenizer"])
    args = ap.parse_args()

    from datasets import load_dataset

    tks = {}
    for n in args.tokenizers:
        tks[n] = AutoTokenizer.from_pretrained(n)

    print(f"{'source':<22}{'wt':>6}{'chars':>12}" +
          "".join(f"{n:>20}" for n in args.tokenizers))
    totals = {n: 0.0 for n in args.tokenizers}
    got_weight = 0.0

    for name, hf_id, cfg, weight in MIX:
        try:
            kw = {"split": "train", "streaming": True}
            if cfg:
                ds = load_dataset(hf_id, cfg, **kw)
            else:
                ds = load_dataset(hf_id, **kw)
            texts, chars = [], 0
            for rec in ds:
                t = text_of(rec)
                if not t:
                    continue
                texts.append(t)
                chars += len(t)
                if len(texts) >= args.docs:
                    break
            if not texts:
                print(f"{name:<22}{weight:>6.2f}   no usable text field")
                continue
            counts = {}
            for n, tk in tks.items():
                counts[n] = sum(
                    len(tk.encode(t, add_special_tokens=False)) for t in texts)
                # tokens per 1k chars, weighted into the mix total
                totals[n] += weight * counts[n] / chars * 1000
            got_weight += weight
            print(f"{name:<22}{weight:>6.2f}{chars:>12,}" +
                  "".join(f"{counts[n]/chars*1000:>13.1f}/1k c"
                          for n in args.tokenizers))
        except Exception as e:  # noqa: BLE001
            print(f"{name:<22}{weight:>6.2f}   UNAVAILABLE: "
                  f"{type(e).__name__}: {str(e)[:60]}")

    if got_weight:
        print(f"\nweighted mix fertility (tokens per 1k chars), "
              f"covering {got_weight:.0%} of the mix:")
        base = None
        for n in args.tokenizers:
            v = totals[n] / got_weight
            if base is None:
                base = v
            print(f"  {n:<24}{v:8.1f}   {(v/base - 1)*100:+6.1f}%")


if __name__ == "__main__":
    main()
