"""Pre-flight every dataset entry of every pretraining phase, CPU-only on Modal
with hf-secret: open the stream and pull rows until three pass the entry's
filter + formatter through the SAME per-row pipeline the trainer runs
(`osrt.data.process_row`). Per source it reports the path taken (format key
or extractor branch), token counts, the rejection-reason histogram and a
200-char repr of the rendered text — so a wrong format (a `role: content`
fallback, a missing `<|end_turn|>`, no markers at all) is visible before
compute is spent. Phases 2 and 3 are otherwise first touched 40 minutes and
~2 days into the trunk.

    MODAL_PROFILE=danielhalwell uv run modal run scripts/preflight_data.py
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .pip_install(
        "torch==2.10.0", extra_options="--index-url https://download.pytorch.org/whl/cpu"
    )
    .pip_install(
        "transformers==5.3.0", "datasets==4.6.1", "tokenizers==0.22.2",
        "huggingface_hub>=0.35", "numpy", "safetensors==0.7.0",
    )
    .add_local_dir(str(ROOT / "src"), "/root/src")
    .add_local_dir(str(ROOT / "tokenizer"), "/root/tokenizer")
)
app = modal.App("osrt-data-preflight", image=image)

MAX_ROWS = 400          # rows to inspect before declaring a source WEAK/FAIL
WANT_PASSED = 3
SAMPLE_CHARS = 200


@app.function(secrets=[modal.Secret.from_name("hf-secret")], timeout=3600, cpu=4)
def preflight(phase_filter: str, sft: bool = False, midtrain: bool = False) -> str:
    import os
    import random
    import sys
    import time
    from collections import Counter

    sys.path.insert(0, "/root/src")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    import osrt.sft_data  # noqa: F401 — registers format="sft"
    from osrt.data import process_row
    from osrt.train_config import MidtrainConfig, PretrainConfig, SFTProbeConfig

    tok = AutoTokenizer.from_pretrained("/root/tokenizer")
    token = os.environ.get("HF_TOKEN")
    if sft:
        sc = SFTProbeConfig()
        phases = {"sft-probe": {"seq_len": sc.seq_len, "datasets": sc.train_datasets()}}
    elif midtrain:
        phases = MidtrainConfig().phases
    else:
        phases = PretrainConfig().phases
    rng = random.Random(0)
    out, bad = [], 0
    for pname, ph in phases.items():
        if phase_filter and pname != phase_filter:
            continue
        out.append(f"== {pname} (seq {ph['seq_len']}, {len(ph['datasets'])} sources)")
        for d in ph["datasets"]:
            t0 = time.time()
            try:
                kw = dict(split=d.get("split", "train"), streaming=True, token=token)
                if d.get("hf_config"):
                    kw["name"] = d["hf_config"]
                ds = load_dataset(d["hf_id"], **kw)
                seen = passed = 0
                reasons: Counter[str] = Counter()
                paths: Counter[str] = Counter()
                lens: list[int] = []
                sample: str | None = None
                first_error: str | None = None
                state: dict = {}          # per-source formatter scratch
                for row in ds:
                    seen += 1
                    res = process_row(d, row, tok, rng, state)
                    paths[res.path] += 1
                    if res.tokens is None:
                        reasons[res.reason] += 1
                        if res.error is not None and first_error is None:
                            first_error = (
                                f"{type(res.error).__name__}: {str(res.error)[:120]} "
                                f"(row keys: {sorted(row)})"
                            )
                        if seen >= MAX_ROWS:
                            break
                        continue
                    passed += 1
                    lens.append(len(res.tokens))
                    if sample is None:
                        sample = res.text
                    if passed >= WANT_PASSED:
                        break
                if passed >= WANT_PASSED:
                    status = "OK "
                else:
                    status = "WEAK" if passed else "FAIL"
                # A chat-shaped rendering must close its assistant turn: the
                # 2026-09-30 review found three coexisting formats, none of
                # which ever emitted <|end_turn|>.
                chatty = sample is not None and "<|user|>" in sample
                if chatty and not sample.endswith("<|end_turn|>"):
                    status = "FMT?"
                if status != "OK ":
                    bad += 1
                out.append(
                    f"  {status} {d['name']:24} rows_seen={seen:4d} passed={passed} "
                    f"tokens/row={lens} path={'/'.join(sorted(paths)) or '-'} "
                    f"rejected={dict(reasons) or '-'} {time.time() - t0:5.1f}s")
                if sample is not None:
                    head = repr(sample[:SAMPLE_CHARS])
                    tail = ""
                    if len(sample) > SAMPLE_CHARS:
                        tail = f" ... {sample[-60:]!r}"
                    out.append(f"       sample: {head}{tail}")
                if first_error:
                    out.append(f"       first error: {first_error}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                out.append(
                    f"  FAIL {d['name']:24} {type(e).__name__}: {str(e)[:160]} "
                    f"{time.time() - t0:5.1f}s")
    out.append(f"== {bad} problem source(s)")
    return "\n".join(out)


@app.local_entrypoint()
def main(phase: str = "", sft: bool = False, midtrain: bool = False):
    print(preflight.remote(phase, sft, midtrain))
