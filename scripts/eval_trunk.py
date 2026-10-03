"""Held-out scoring of the final trunk checkpoints and their soup, GPU-side.

Scores osrt_step_17000 / 17500 / osrt_final (18000) and the fp32 average of
the three on the trunk's held-out slice (fineweb-edu CC-MAIN-2025-26) at the
two context lengths the trunk itself evaluated at, on identical cached
batches. Saves the soup (weights only) to /vol/trunk.

    MODAL_PROFILE=danielhalwell uv run modal run scripts/eval_trunk.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

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
app = modal.App("osrt-eval-trunk", image=image)

CANDIDATES = ["osrt_step_17000.pt", "osrt_step_17500.pt", "osrt_final.pt"]
# (seq_len, batch): the shapes the trunk itself evaluated at (anneal / knowledge)
SHAPES = [(8192, 2), (4096, 6)]


@app.function(gpu="H100", timeout=3600, volumes={"/vol": vol}, memory=65536, cpu=8,
              secrets=[modal.Secret.from_name("hf-secret")])
def score(eval_steps: int, save_soup: bool, candidates: list[str]) -> str:
    import time

    import torch

    sys.path.insert(0, "/root/src")
    from transformers import AutoTokenizer

    from osrt.model import OSRTForCausalLM
    from osrt.presets import OSRT_V7, build_config
    from osrt.train import run_eval

    vol.reload()
    tok = AutoTokenizer.from_pretrained("/root/tokenizer")
    real = len(tok)
    padded = ((real + 127) // 128) * 128
    assert real == OSRT_V7["real_vocab_size"], real
    cfg = build_config(
        vocab_size=padded, real_vocab_size=real,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )
    model = OSRTForCausalLM(cfg)

    cands = candidates or CANDIDATES
    soup_name = "soup_" + "_".join(
        c.removeprefix("osrt_step_").removesuffix(".pt").replace("osrt_final", "18000")
        for c in cands)
    sds, steps = {}, {}
    for n in cands:
        ck = torch.load(f"/vol/trunk/{n}", map_location="cpu", weights_only=False)
        sds[n] = ck["model_state_dict"]
        steps[n] = ck.get("step")
        del ck
    soup = {}
    for k, v in sds[cands[-1]].items():
        if v.is_floating_point():
            soup[k] = sum(sds[n][k].float() for n in cands) / len(cands)
            soup[k] = soup[k].to(v.dtype)
        else:
            soup[k] = v.clone()
    sds[soup_name] = soup

    lines = [f"steps: {steps} | eval_steps={eval_steps} | shapes={SHAPES}"]
    results = {}
    device = torch.device("cuda")
    for name, sd in sds.items():
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        row = {}
        for seq_len, bs in SHAPES:
            t0 = time.time()
            m = run_eval(model, "/root/tokenizer", seq_len, bs, eval_steps, device,
                         cfg.real_vocab_size)
            row[(seq_len, bs)] = (
                m["eval/loss"], m["eval/perplexity"], time.time() - t0)
        results[name] = row
        lines.append(
            f"{name:28s} missing={len(missing)} unexpected={len(unexpected)} | "
            + " | ".join(f"{sl}x{bs}: loss {loss:.4f} ppl {ppl:.1f} ({dt:.0f}s)"
                         for (sl, bs), (loss, ppl, dt) in row.items())
        )
        model.to("cpu")
        torch.cuda.empty_cache()

    if save_soup:
        out = f"/vol/trunk/osrt_{soup_name}.pt"
        torch.save({"step": 18000, "model_state_dict": soup,
                    "soup_of": cands, "heldout": {
                        n: {f"{s}x{b}": r[0] for (s, b), r in row.items()}
                        for n, row in results.items()}}, out)
        vol.commit()
        lines.append(f"saved soup (weights only) -> {out}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(eval_steps: int = 20, save_soup: bool = True, candidates: str = ""):
    cands = [c.strip() for c in candidates.split(",") if c.strip()]
    print(score.remote(eval_steps, save_soup, cands))
