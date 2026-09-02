"""Pre-flight every dataset entry of every pretraining phase, CPU-only on Modal
with hf-secret: open the stream, pull rows until three pass the entry's
filter + formatter, report text length / tokens or FAIL. Phases 2 and 3 are
otherwise first touched 40 minutes and ~2 days into the trunk.

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


@app.function(secrets=[modal.Secret.from_name("hf-secret")], timeout=3600, cpu=4)
def preflight(phase_filter: str) -> str:
    import os
    import random
    import sys
    import time

    sys.path.insert(0, "/root/src")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from osrt.data import FORMAT_FN_PRETRAIN, TokenStream, row_passes
    from osrt.train_config import PretrainConfig

    tok = AutoTokenizer.from_pretrained("/root/tokenizer")
    token = os.environ.get("HF_TOKEN")
    cfg = PretrainConfig()
    ts = TokenStream.__new__(TokenStream)   # only for _extract_text
    rng = random.Random(0)
    out, bad = [], 0
    for pname, ph in cfg.phases.items():
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
                fmt = FORMAT_FN_PRETRAIN.get(d["format"]) if d.get("format") else None
                seen = passed = 0
                lens: list[int] = []
                for row in ds:
                    seen += 1
                    if not row_passes(d, row, rng):
                        if seen >= 400:
                            break
                        continue
                    text = fmt(row) if fmt else ts._extract_text(row, tok)
                    if not text or not text.strip():
                        if seen >= 400:
                            break
                        continue
                    n = len(tok.encode(text, add_special_tokens=False))
                    if d.get("max_tokens") and n > d["max_tokens"]:
                        if seen >= 400:
                            break
                        continue
                    passed += 1
                    lens.append(n)
                    if passed >= 3:
                        break
                status = "OK " if passed >= 3 else "WEAK" if passed else "FAIL"
                if status != "OK ":
                    bad += 1
                out.append(
                    f"  {status} {d['name']:24} rows_seen={seen:4d} passed={passed} "
                    f"tokens/row={lens} {time.time() - t0:5.1f}s")
            except Exception as e:  # noqa: BLE001
                bad += 1
                out.append(
                    f"  FAIL {d['name']:24} {type(e).__name__}: {str(e)[:160]} "
                    f"{time.time() - t0:5.1f}s")
    out.append(f"== {bad} problem source(s)")
    return "\n".join(out)


@app.local_entrypoint()
def main(phase: str = ""):
    print(preflight.remote(phase))
