#!/usr/bin/env python3
"""Train the custom 65K BPE tokenizer for OSRT-605M (v6 contract).

v6 change vs the v4 32K tokenizer: vocab 32768 → 65536, and the special
set is the full 21-token contract (ARCHITECTURE.md §3.2) — adds
<|end_turn|>, tool_call/result, image, audio (IDs 14-20) that the v4
tokenizer lacked — plus an 11-slot reserved band (21-31) so real BPE
vocab starts at id 32. The model's vocab is frozen into the embedding at
step 0, so this MUST be right before pretraining.


Default uses HuggingFace byte-level BPE. Optionally falls through to
SuperBPE (COLM 2025) if the 'superbpe' package is installed for the
two-stage subwords → superwords curriculum, which produces ~33% fewer
tokens and +4% on benchmarks vs standard BPE.

Training data: sampled from the pre-training mix (FineWeb-Edu +
CodeParrot-clean + Wikipedia) to ensure the tokenizer is optimised for
the exact distribution the model will see during pretraining.

Usage:
    # Full training (~2-4 hours on CPU, ~30 min on GPU)
    uv run python scripts/train_tokenizer.py

    # Quick test (1M chars)
    uv run python scripts/train_tokenizer.py --sample-size 1000000

    # Custom output path
    uv run python scripts/train_tokenizer.py --output ./my-tokenizer
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time

# Try SuperBPE first, fall back to HuggingFace tokenizers
USE_SUPERBPE = importlib.util.find_spec("superbpe") is not None


def sample_training_data(sample_size: int = 2_000_000_000, seed: int = 42) -> str:
    """Sample training data from pre-training mix and save to temp file.

    Samples proportionally:
      - 55% FineWeb-Edu (general text)
      - 30% CodeParrot Clean (code)
      - 15% Wikipedia (factual/STEM)

    For a 32K BPE tokenizer, ~2GB of training data is the sweet spot —
    larger samples hit diminishing returns on merge quality and expose
    us to more HF Hub network issues.

    Resilient to transient HF Hub connection drops: if a source fails
    partway through, we log the partial collection and move on to the
    next source instead of aborting the whole run.

    Args:
        sample_size: Target number of characters to sample (default 2GB).
        seed: Random seed.

    Returns:
        Path to temporary file with training text.
    """
    from datasets import load_dataset

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    )

    # Mix matched to the model's ACTUAL pretraining corpus, not the
    # "multilingual" marketing. The model is math-first (mid-training is
    # ~47% math, GRPO is verifiable-math), so the tokenizer MUST see
    # math-dense text or LaTeX/symbols/numerics tokenize near byte-level
    # — permanently, since the vocab freezes into the embedding. OpenWebMath
    # (same source used in pretrain_extend2) supplies that. Multilingual is
    # deliberately absent because the pretraining corpus is English-only.
    sources = [
        {
            "name": "FineWeb-Edu",
            "hf_id": "HuggingFaceFW/fineweb-edu",
            "fraction": 0.45,
            "text_key": "text",
        },
        {
            "name": "OpenWebMath",
            "hf_id": "open-web-math/open-web-math",
            "fraction": 0.20,
            "text_key": "text",
        },
        {
            "name": "CodeParrot Clean",
            "hf_id": "codeparrot/codeparrot-clean",
            "fraction": 0.20,
            "text_key": "content",
        },
        {
            "name": "Wikipedia",
            "hf_id": "wikimedia/wikipedia",
            "hf_config": "20231101.en",
            "fraction": 0.15,
            "text_key": "text",
        },
    ]

    def _log(msg: str) -> None:
        """Print with explicit flush so Modal log tail stays live."""
        print(msg, flush=True)

    def _stream_with_retry(src: dict, target_chars: int) -> int:
        """Stream from one source, retrying on transient errors.

        Logs progress every ~10MB and on every retry so the run is never
        silent. Returns the number of characters successfully written.
        """
        max_attempts = 3
        collected = 0
        last_error = None
        # Log roughly every 10MB of accumulated text.
        progress_every = 10_000_000
        next_progress_at = progress_every
        src_start = time.time()

        for attempt in range(1, max_attempts + 1):
            if collected >= target_chars:
                break
            try:
                load_kwargs = {"split": "train", "streaming": True}
                if "hf_config" in src:
                    load_kwargs["name"] = src["hf_config"]

                _log(
                    f"    [{src['name']}] attempt {attempt}/{max_attempts}: "
                    f"opening stream..."
                )
                ds = load_dataset(src["hf_id"], **load_kwargs)
                ds = ds.shuffle(buffer_size=10_000, seed=seed + attempt)
                _log(f"    [{src['name']}] stream ready, reading examples...")

                examples_seen = 0
                for example in ds:
                    examples_seen += 1
                    text = example.get(src["text_key"], "")
                    if not text or len(text) < 100:
                        continue
                    tmp.write(text + "\n")
                    collected += len(text)

                    if collected >= next_progress_at:
                        elapsed = time.time() - src_start
                        mb = collected / 1e6
                        pct = collected / target_chars * 100
                        rate = mb / elapsed if elapsed > 0 else 0
                        _log(
                            f"    [{src['name']}] {mb:.0f} MB "
                            f"({pct:.1f}% of target, "
                            f"{examples_seen:,} examples read, "
                            f"{rate:.1f} MB/s, {elapsed:.0f}s elapsed)"
                        )
                        next_progress_at += progress_every

                    if collected >= target_chars:
                        break
                # Clean exit — loop terminated because we hit target or
                # the stream ended naturally.
                elapsed = time.time() - src_start
                _log(
                    f"    [{src['name']}] stream finished at {collected / 1e6:.0f} MB "
                    f"({examples_seen:,} examples, {elapsed:.0f}s)"
                )
                return collected
            except Exception as e:
                last_error = e
                pct = collected / target_chars * 100 if target_chars else 0
                _log(
                    f"    [{src['name']}] attempt {attempt}/{max_attempts} "
                    f"hit {type(e).__name__}: {str(e)[:120]}"
                )
                _log(
                    f"    [{src['name']}] {collected / 1e6:.0f} MB collected so far "
                    f"({pct:.0f}% of target), retrying from fresh stream..."
                )

        if last_error and collected < target_chars:
            pct = collected / target_chars * 100 if target_chars else 0
            _log(
                f"    [{src['name']}] giving up after {max_attempts} attempts "
                f"({collected / 1e6:.0f} MB, {pct:.0f}% of target)"
            )
        return collected

    sample_start = time.time()
    _log(f"  Target: {sample_size:,} chars ({sample_size / 1e9:.1f} GB)")

    total_chars = 0
    for src in sources:
        target_chars = int(sample_size * src["fraction"])
        _log(
            f"  Sampling {src['name']} "
            f"({target_chars:,} chars target, {target_chars / 1e9:.2f} GB)..."
        )
        got = _stream_with_retry(src, target_chars)
        total_chars += got
        _log(f"    Collected {got:,} chars from {src['name']}")

    tmp.close()
    elapsed = time.time() - sample_start
    _log(
        f"  Sampling complete: {total_chars:,} chars "
        f"({total_chars / 1e9:.2f} GB) in {elapsed:.0f}s -> {tmp.name}"
    )

    # Gate scales with what was requested: a small `--sample-size` for
    # quick tests should pass when most of the request lands, while a 2GB
    # default run shouldn't accept a network failure that returned 50MB.
    # Half of requested is the threshold; an absolute floor catches the
    # case where the user asked for almost nothing.
    min_required = max(min(sample_size // 2, 100_000_000), 1_000_000)
    if total_chars < min_required:
        raise RuntimeError(
            f"Sampling produced only {total_chars:,} chars "
            f"(< {min_required:,} required for sample_size={sample_size:,}). "
            "Network to HF Hub is probably broken. Retry later or "
            "lower --sample-size further."
        )

    return tmp.name


def train_with_hf_tokenizers(data_path: str, vocab_size: int, output_dir: str) -> None:
    """Train a BPE tokenizer using HuggingFace tokenizers library.

    Falls back to this when SuperBPE is not installed.
    Still produces a high-quality BPE tokenizer, just without superword merges.
    """
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        pre_tokenizers,
        processors,
        trainers,
    )

    print("\nTraining BPE tokenizer with HuggingFace tokenizers...", flush=True)
    print(f"  Vocab size: {vocab_size:,}", flush=True)

    # BPE model with byte fallback
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenization: byte-level (handles any Unicode)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Special tokens — the full v6 21-token contract (ARCHITECTURE.md §3.2),
    # plus an 11-slot reserved band so real BPE vocab begins at id 32 and
    # future special tokens can be slotted in WITHOUT shifting trained IDs
    # (the v4 tokenizer only had IDs 0-13 and started real vocab at 14,
    # which left no room for tool/vision/audio without a re-embed).
    special_tokens = [
        "<|padding|>",          # 0: pad
        "<|begin_of_text|>",    # 1: bos
        "<|end_of_text|>",      # 2: eos
        "<|unknown|>",          # 3: unk
        "<|fim_prefix|>",       # 4: fill-in-middle (code)
        "<|fim_middle|>",       # 5: fill-in-middle
        "<|fim_suffix|>",       # 6: fill-in-middle
        "<|think|>",            # 7: reasoning open
        "<|/think|>",           # 8: reasoning close
        "<|answer|>",           # 9: answer open
        "<|/answer|>",          # 10: answer close
        "<|user|>",             # 11: user turn
        "<|assistant|>",        # 12: assistant turn
        "<|system|>",           # 13: system prompt
        "<|end_turn|>",         # 14: turn separator (ChatML-style)
        "<|tool_call|>",        # 15: tool invocation open
        "<|/tool_call|>",       # 16: tool invocation close
        "<|tool_result|>",      # 17: tool result open
        "<|/tool_result|>",     # 18: tool result close
        "<|image|>",            # 19: vision retrofit placeholder
        "<|audio|>",            # 20: audio placeholder
        # 21-31: reserved expansion band → real vocab begins at id 32.
        *[f"<|reserved_{i}|>" for i in range(21, 32)],
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Report sample size in MB so the user can estimate training time.
    import os as _os
    sample_mb = _os.path.getsize(data_path) / 1e6
    print(
        f"  Training on {sample_mb:.0f} MB of text "
        f"(this usually takes 2-10 min depending on CPU speed)...",
        flush=True,
    )
    t0 = time.time()
    tokenizer.train([data_path], trainer)
    print(f"  Training done in {time.time() - t0:.0f}s", flush=True)

    # Post-processor: add BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A <|end_of_text|>",
        pair=(
            "<|begin_of_text|> $A <|end_of_text|> "
            "<|begin_of_text|> $B:1 <|end_of_text|>:1"
        ),
        special_tokens=[
            ("<|begin_of_text|>", 1),
            ("<|end_of_text|>", 2),
        ],
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    # Create HF tokenizer config files
    _create_hf_tokenizer_config(output_dir, special_tokens)

    print(f"  Saved to {output_dir}/")

    # Verify
    _verify_tokenizer(output_dir)


def train_with_superbpe(data_path: str, vocab_size: int, output_dir: str) -> None:
    """Train a SuperBPE tokenizer (two-stage: subwords then superwords).

    SuperBPE produces 33% fewer tokens vs standard BPE.
    """
    print("\nTraining SuperBPE tokenizer (two-stage)...")
    print(f"  Vocab size: {vocab_size:,}")

    # Stage 1: Standard BPE with pretokenization (subwords)
    subword_vocab = int(vocab_size * 0.75)  # 75% subwords, 25% superwords
    print(f"  Stage 1: BPE subwords (vocab={subword_vocab:,})...")
    t0 = time.time()

    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "superbpe.train",
         "--input", data_path,
         "--vocab_size", str(subword_vocab),
         "--output", f"{output_dir}/stage1",
         "--pretokenize"],
        check=True,
    )
    print(f"  Stage 1 done in {time.time() - t0:.0f}s")

    # Stage 2: Continue without pretokenization (superwords)
    print(f"  Stage 2: SuperBPE extension to {vocab_size:,}...")
    t0 = time.time()
    subprocess.run(
        [sys.executable, "-m", "superbpe.extend",
         "--input", data_path,
         "--base_tokenizer", f"{output_dir}/stage1",
         "--vocab_size", str(vocab_size),
         "--output", f"{output_dir}/final"],
        check=True,
    )
    print(f"  Stage 2 done in {time.time() - t0:.0f}s")

    # Convert to HF format
    subprocess.run(
        [sys.executable, "-m", "superbpe.construct_hf_tokenizer",
         "--tokenizer_path", f"{output_dir}/final",
         "--output_path", output_dir],
        check=True,
    )

    _verify_tokenizer(output_dir)


def _create_hf_tokenizer_config(output_dir: str, special_tokens: list[str]) -> None:
    """Create HuggingFace tokenizer config files."""
    import json

    # tokenizer_config.json
    config = {
        "model_type": "osrt",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|padding|>",
        "unk_token": "<|unknown|>",
        "add_bos_token": True,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # special_tokens_map.json
    special_map = {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|padding|>",
        "unk_token": "<|unknown|>",
        "additional_special_tokens": special_tokens[4:],  # FIM + think tokens
    }
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_map, f, indent=2)


def _verify_tokenizer(output_dir: str) -> None:
    """Verify the tokenizer works correctly."""
    from transformers import AutoTokenizer

    print("\n  Verifying tokenizer...")
    tok = AutoTokenizer.from_pretrained(output_dir)

    test_cases = [
        "Hello, world!",
        (
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n-1) + fibonacci(n-2)"
        ),
        "The derivative of x^2 is 2x.",
        (
            "<|user|>What is 2+2?<|assistant|><|think|>"
            "2+2 equals 4<|/think|><|answer|>4<|/answer|>"
        ),
        "import torch\nimport torch.nn as nn\n\nclass MyModel(nn.Module):\n    pass",
    ]

    print(f"  Vocab size: {tok.vocab_size:,}")
    print(f"  BOS: {tok.bos_token} (id={tok.bos_token_id})")
    print(f"  EOS: {tok.eos_token} (id={tok.eos_token_id})")
    print(f"  PAD: {tok.pad_token} (id={tok.pad_token_id})")

    for text in test_cases:
        tokens = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(tokens)
        ratio = len(text) / len(tokens) if tokens else 0
        print(f"  [{len(tokens):>4d} tokens, {ratio:.1f} chars/tok] {text[:60]}...")

    # Roundtrip test
    for text in test_cases:
        tokens = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(tokens)
        assert decoded == text, f"Roundtrip failed: {repr(text)} != {repr(decoded)}"
    print("  Roundtrip tests passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Train the v6 65K tokenizer for OSRT-605M"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=65536,
        help="Target vocabulary size (65K default — v6 OSRT-605M contract)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3_000_000_000,
        help="Training text size in chars (~3GB default — more merge data "
             "for the larger 65K vocab than the 2GB used for 32K)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./tokenizer",
        help="Output directory",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Pre-existing training text file (skip download)",
    )
    args = parser.parse_args()

    print("OSRT — Custom Tokenizer Training")
    print("=" * 50)

    # Get training data
    if args.data_path and os.path.exists(args.data_path):
        data_path = args.data_path
        print(f"Using existing data: {data_path}")
    else:
        print(f"Sampling {args.sample_size:,} chars from pre-training mix...")
        data_path = sample_training_data(args.sample_size)

    # Train tokenizer
    if USE_SUPERBPE:
        print("\nUsing SuperBPE (COLM 2025 — 33% fewer tokens)")
        train_with_superbpe(data_path, args.vocab_size, args.output)
    else:
        print("\nSuperBPE not installed, using HuggingFace BPE")
        print("  (install superbpe for better tokenization: pip install superbpe)")
        train_with_hf_tokenizers(data_path, args.vocab_size, args.output)

    print(f"\nTokenizer saved to: {args.output}/")
    print(f"Use with: AutoTokenizer.from_pretrained('{args.output}')")


if __name__ == "__main__":
    main()
