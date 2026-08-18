"""Measure a tokenizer's arithmetic fitness.

Three properties decide whether a tokenizer can represent arithmetic cleanly:

1. CONSISTENCY  — the same number must produce the same tokens regardless of
   the characters around it. If "1234" splits differently in "1234" and
   " 1234" and "= 1234", the model must learn each variant separately.
2. PLACE-VALUE ALIGNMENT — when numbers are grouped, grouping must run
   right-to-left so that units/tens/hundreds land in the same slot across
   numbers of different lengths. Left-to-right grouping misaligns them.
3. GRANULARITY — how many tokens a d-digit number costs, by d.

Usage:
    PYTHONPATH=src uv run python scripts/probe_tokenizer_math.py <tok> [<tok> ...]
"""
from __future__ import annotations

import sys

from transformers import AutoTokenizer

CONTEXTS = ["{n}", " {n}", "={n}", "= {n}", "${n}", "({n})", "\n{n}",
            "is {n}.", "x{n}", "{n},", "answer: {n}"]


def toks(tk, s):
    return tk.convert_ids_to_tokens(tk.encode(s, add_special_tokens=False))


def digit_slice(tk, n: str):
    """Tokens covering the digits of n when embedded in a bare context."""
    return toks(tk, n)


def consistency(tk, numbers):
    """Fraction of numbers that tokenize identically across all contexts."""
    stable = 0
    examples = []
    for n in numbers:
        forms = set()
        for c in CONTEXTS:
            t = toks(tk, c.format(n=n))
            # keep only the pieces that contain a digit
            # strip the byte-level space marker: a leading space is real
            # information, not instability. Only the digit grouping matters.
            forms.add(tuple(x.lstrip("\u0120\u2581") for x in t
                            if any(ch.isdigit() for ch in x)))
        if len(forms) == 1:
            stable += 1
        elif len(examples) < 3:
            examples.append((n, sorted(forms)[:3]))
    return stable / len(numbers), examples


def report(name):
    try:
        tk = AutoTokenizer.from_pretrained(name)
    except Exception as e:  # noqa: BLE001
        print(f"\n### {name}\n  UNAVAILABLE: {type(e).__name__}: {e}")
        return
    print(f"\n### {name}   (vocab {tk.vocab_size:,})")

    # granularity by digit length
    print("  tokens per number, by digit length:")
    for d in range(1, 8):
        sample = [str(10**(d-1) + i * 7 % (9 * 10**(d-1))) for i in range(60)]
        sample = [s for s in sample if len(s) == d] or [("1" * d)]
        counts = [len(digit_slice(tk, s)) for s in sample]
        single = sum(c == 1 for c in counts) / len(counts)
        print(f"    {d} digit: mean {sum(counts)/len(counts):5.2f} tok   "
              f"single-token {single*100:5.1f}%   e.g. {sample[0]!r} -> "
              f"{digit_slice(tk, sample[0])}")

    # consistency across contexts
    nums = [str(x) for x in (7, 42, 100, 365, 1234, 2026, 15000, 123456)]
    frac, ex = consistency(tk, nums)
    print(f"  context consistency: {frac*100:.1f}% of numbers tokenize "
          f"identically in all {len(CONTEXTS)} contexts")
    for n, forms in ex:
        print(f"    UNSTABLE {n}: {forms}")

    # place-value alignment: does grouping run right-to-left?
    print("  place-value alignment:")
    for n in ("1234567", "234567", "34567"):
        print(f"    {n:>8} -> {digit_slice(tk, n)}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["tokenizer"]
    for nm in names:
        report(nm)
