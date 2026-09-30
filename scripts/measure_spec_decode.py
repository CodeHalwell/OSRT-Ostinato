"""Measure MTP-head speculative decoding against plain greedy on a ladder
checkpoint, GPU-side (Modal). Reports tok/s (each path over its OWN new
tokens), acceptance rate, tokens per forward, and whether the two paths agree:
`identical` is exact equality; `identical_to_eos` compares the continuations
up to and including the first EOS, which is the correctness bar while the
speculative path may still stop at the same EOS with a different tail length.
After the model fix both should read True.

    MODAL_PROFILE=gradio-winter-hack uv run modal run \\
        scripts/measure_spec_decode.py --arm hra --step 500
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

    eos = model.config.eos_token_id
    if eos is None:
        eos = tok.eos_token_id
    spec_kw = {"speculative": True, "spec_drafter": "mtp"}

    def run(ids, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0, **kw)
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    def to_eos(seq: list[int]) -> list[int]:
        """The continuation up to and including its first EOS."""
        return seq[: seq.index(eos) + 1] if eos in seq else seq

    tot = {
        "greedy_s": 0.0,
        "spec_s": 0.0,
        "greedy_tokens": 0,
        "spec_tokens": 0,
        "off": 0,
        "acc": 0,
        "fwd": 0,
        "identical": 0,
        "identical_to_eos": 0,
    }
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        n_prompt = ids.shape[1]
        # Warm up BOTH paths: each has its own compiled graphs and cache
        # shapes, and whichever ran first used to pay for the other.
        run(ids)
        run(ids, **spec_kw)
        g, tg = run(ids)
        g2, _ = run(ids)                      # greedy repeatability control
        s, ts = run(ids, **spec_kw)
        st = model.last_spec_stats
        g_new = g[0, n_prompt:].tolist()
        s_new = s[0, n_prompt:].tolist()
        n_g, n_s = len(g_new), len(s_new)     # each path's OWN new tokens
        identical = g_new == s_new
        rep = torch.equal(g, g2)
        g_eos, s_eos = to_eos(g_new), to_eos(s_new)
        k = min(len(g_eos), len(s_eos))
        identical_to_eos = k > 0 and g_eos[:k] == s_eos[:k]
        first_div = next(
            (i for i in range(min(n_g, n_s)) if g_new[i] != s_new[i]), -1)
        greedy_tps, spec_tps = n_g / tg, n_s / ts
        tot["greedy_s"] += tg
        tot["spec_s"] += ts
        tot["greedy_tokens"] += n_g
        tot["spec_tokens"] += n_s
        tot["off"] += st["drafts_offered"]
        tot["acc"] += st["drafts_accepted"]
        tot["fwd"] += st["forwards"]
        tot["identical"] += int(identical)
        tot["identical_to_eos"] += int(identical_to_eos)
        lines.append(
            f"  prompt={p[:28]!r:32} greedy {greedy_tps:6.1f} tok/s ({n_g}) "
            f"| mtp-spec {spec_tps:6.1f} tok/s ({n_s}) "
            f"| speedup {spec_tps / greedy_tps:4.2f}x "
            f"| accept {st['acceptance_rate']:.2f} "
            f"| tok/fwd {st['tokens_per_forward']:.2f}"
        )
        lines.append(
            f"    identical={identical} identical_to_eos={identical_to_eos} "
            f"greedy_repeatable={rep} first_div@{first_div} "
            f"eos@greedy={len(g_eos) if eos in g_new else -1} "
            f"eos@spec={len(s_eos) if eos in s_new else -1}"
        )
        lines.append("    greedy: " + repr(tok.decode(g_new[:40])))
    greedy_tps = tot["greedy_tokens"] / tot["greedy_s"]
    spec_tps = tot["spec_tokens"] / tot["spec_s"]
    lines.append(
        f"TOTAL greedy {greedy_tps:.1f} tok/s | mtp-spec {spec_tps:.1f} tok/s "
        f"| speedup {spec_tps / greedy_tps:.2f}x "
        f"| acceptance {tot['acc'] / max(tot['off'], 1):.3f} "
        f"| tok/fwd {tot['spec_tokens'] / max(tot['fwd'], 1):.2f} "
        f"| identical {tot['identical']}/{len(PROMPTS)} "
        f"| identical_to_eos {tot['identical_to_eos']}/{len(PROMPTS)}"
    )
    return "\n".join(lines)


@app.local_entrypoint()
def main(arm: str = "hra", step: int = 500, max_new_tokens: int = 128,
         dtype: str = "bf16"):
    print(measure.remote(arm, step, max_new_tokens, dtype))
