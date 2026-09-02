"""Measure MTP-head speculative decoding against plain greedy on a ladder
checkpoint, GPU-side (Modal). Reports tok/s, acceptance rate, tokens per
forward, and checks the two paths emit identical tokens.

    MODAL_PROFILE=gradio-winter-hack uv run modal run \\
        scripts/measure_spec_decode.py --arm nohra --step 500
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# Same image as app.py (torch 2.10.0+cu128 + pins), defined inline because the
# container only mounts this file, so `from app import image` cannot resolve.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .pip_install(
        "torch==2.10.0+cu128",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==5.3.0", "datasets==4.6.1", "triton==3.6.0",
        "tokenizers==0.22.2", "safetensors==0.7.0", "huggingface_hub>=0.35", "numpy",
    )
    .add_local_dir(str(Path(__file__).resolve().parent.parent / "src"), "/root/src")
    .add_local_dir(
        str(Path(__file__).resolve().parent.parent / "tokenizer"), "/root/tokenizer")
)
vol = modal.Volume.from_name("osrt-v7-ladder-ckpt")
app = modal.App("osrt-spec-measure", image=image)

PROMPTS = [
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "The derivative of x^3 + 2x is",
    "Question: A train travels 60 miles in 1.5 hours. What is its average speed?\nAnswer:",  # noqa: E501
    "import numpy as np\n\ndef softmax(x):\n",
]


@app.function(gpu="H100", timeout=1800, volumes={"/vol": vol})
def measure(arm: str, step: int, max_new_tokens: int, dtype: str) -> str:
    import time

    import torch

    sys.path.insert(0, "/root/src")
    from transformers import AutoTokenizer

    from osrt.model import OSRTForCausalLM
    from osrt.presets import LADDER_ARMS, build_config

    cfg = build_config(LADDER_ARMS[arm])
    model = OSRTForCausalLM(cfg)
    ck = torch.load(
        f"/vol/ladder_{arm}/osrt_step_{step}.pt", map_location="cpu", weights_only=False
    )
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    model = model.to("cuda", dtype=dt).eval()
    tok = AutoTokenizer.from_pretrained("/root/tokenizer")
    lines = [
        f"[{arm} step {step} {dtype}] mtp_heads={cfg.mtp_heads} missing={len(missing)} unexpected={len(unexpected)}"  # noqa: E501
    ]

    def run(ids, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0, **kw)
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    tot = {
        "greedy_s": 0.0,
        "spec_s": 0.0,
        "tokens": 0,
        "off": 0,
        "acc": 0,
        "fwd": 0,
        "equal": 0,
    }
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        run(ids)  # warm-up (compile/allocator)
        g, tg = run(ids)
        g2, _ = run(ids)                      # greedy repeatability control
        s, ts = run(ids, speculative=True, spec_drafter="mtp")
        st = model.last_spec_stats
        n = g.shape[1] - ids.shape[1]
        eq = torch.equal(g, s)
        rep = torch.equal(g, g2)
        diff = (g[0] != s[0]).nonzero()
        first_div = int(diff[0]) - ids.shape[1] if diff.numel() else -1
        tot["greedy_s"] += tg
        tot["spec_s"] += ts
        tot["tokens"] += n
        tot["off"] += st["drafts_offered"]
        tot["acc"] += st["drafts_accepted"]
        tot["fwd"] += st["forwards"]
        tot["equal"] += int(eq)
        lines.append(
            f"  prompt={p[:28]!r:32} greedy {n / tg:6.1f} tok/s | mtp-spec {n / ts:6.1f} tok/s "  # noqa: E501
            f"| speedup {tg / ts:4.2f}x | accept {st['acceptance_rate']:.2f} "
            f"| tok/fwd {st['tokens_per_forward']:.2f} | identical={eq} greedy_repeatable={rep} first_div@{first_div}"
        )
        lines.append(
            "    greedy: " + repr(tok.decode(g[0, ids.shape[1] : ids.shape[1] + 40]))
        )
    lines.append(
        f"TOTAL greedy {tot['tokens'] / tot['greedy_s']:.1f} tok/s | mtp-spec {tot['tokens'] / tot['spec_s']:.1f} tok/s "  # noqa: E501
        f"| speedup {tot['greedy_s'] / tot['spec_s']:.2f}x | acceptance {tot['acc'] / max(tot['off'], 1):.3f} "  # noqa: E501
        f"| tok/fwd {tot['tokens'] / tot['fwd']:.2f} | identical {tot['equal']}/{len(PROMPTS)}"  # noqa: E501
    )
    return "\n".join(lines)


@app.local_entrypoint()
def main(arm: str = "nohra", step: int = 500, max_new_tokens: int = 128,
         dtype: str = "bf16"):
    print(measure.remote(arm, step, max_new_tokens, dtype))
