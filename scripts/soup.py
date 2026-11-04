"""Soup midtrain checkpoints from an HF mirror and score each + the soup on the
held-out slice (plain PyTorch, any CUDA card; Colab-friendly).

    PYTHONPATH=src python scripts/soup.py --repo HallD/OSRT-Ostinato-midtrain-v4 \\
        --ckpts osrt_step_12500.pt,osrt_step_13000.pt,osrt_step_13500.pt --push

Saves `osrt_soup_<steps>.pt` (weights only, step = highest) next to the
downloads and, with --push, uploads it to the repo under soup/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from osrt.model import OSRTForCausalLM  # noqa: E402
from osrt.presets import OSRT_V7, build_config  # noqa: E402
from osrt.soup import average_state_dicts  # noqa: E402
from osrt.train import run_eval  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument(
        "--ckpts", required=True, help="comma-separated file names in the repo"
    )
    p.add_argument("--tokenizer", default="./tokenizer")
    p.add_argument("--local-dir", default="./checkpoints/soup")
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--push", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    real = len(tok)
    assert real == OSRT_V7["real_vocab_size"], real
    cfg = build_config(
        vocab_size=((real + 127) // 128) * 128,
        real_vocab_size=real,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
        router_bias_in_gates=True,
    )
    model = OSRTForCausalLM(cfg)
    names = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    sds, steps = {}, []
    for n in names:
        path = hf_hub_download(args.repo, n, local_dir=args.local_dir)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sds[n] = ck["model_state_dict"]
        steps.append(int(ck["step"]))
        del ck
    soup_name = "osrt_soup_" + "_".join(str(s) for s in steps) + ".pt"
    sds[soup_name] = average_state_dicts([sds[n] for n in names])

    device = torch.device("cuda")
    results = {}
    for name, sd in sds.items():
        t0 = time.time()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        m = run_eval(
            model, args.tokenizer, 4096, 6, args.eval_steps, device, cfg.real_vocab_size
        )
        results[name] = m["eval/loss"]
        print(
            f"{name:34s} held-out {m['eval/loss']:.4f} ppl {m['eval/perplexity']:.1f} "
            f"(missing {len(missing)} unexpected {len(unexpected)}, "
            f"{time.time() - t0:.0f}s)",
            flush=True,
        )
        model.to("cpu")
        torch.cuda.empty_cache()

    out = os.path.join(args.local_dir, soup_name)
    torch.save(
        {
            "step": max(steps),
            "model_state_dict": sds[soup_name],
            "soup_of": names,
            "heldout": results,
        },
        out,
    )
    print(f"saved {out}", flush=True)
    if args.push:
        HfApi().upload_file(
            path_or_fileobj=out,
            path_in_repo=f"soup/{soup_name}",
            repo_id=args.repo,
            commit_message=soup_name,
        )
        print(f"pushed {args.repo}/soup/{soup_name}", flush=True)


if __name__ == "__main__":
    main()
