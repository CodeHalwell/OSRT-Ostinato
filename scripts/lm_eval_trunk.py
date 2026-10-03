"""lm-evaluation-harness pass over a trunk checkpoint, GPU-side (Modal).

    MODAL_PROFILE=danielhalwell uv run modal run scripts/lm_eval_trunk.py \\
        --ckpt osrt_final.pt --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande \\
        --limit 0

Results land in /vol/evals/<ckpt>_<tag>.json on the workspace volume and the
summary table is printed. `--limit 0` means the full task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "HF_ALLOW_CODE_EVAL": "1"})
    .pip_install(
        "torch==2.10.0+cu128",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==5.3.0", "datasets==4.6.1", "triton==3.6.0",
        "tokenizers==0.22.2", "safetensors==0.7.0", "huggingface_hub>=0.35", "numpy",
        "lm-eval>=0.4.11", "wandb==0.25.1",
    )
    .add_local_dir(str(Path(__file__).resolve().parent.parent / "src"), "/root/src")
    .add_local_dir(
        str(Path(__file__).resolve().parent.parent / "tokenizer"), "/root/tokenizer")
)
vol = modal.Volume.from_name("osrt-v7-ladder-ckpt")
app = modal.App("osrt-lm-eval", image=image)


@app.function(gpu="H100", timeout=4 * 3600, volumes={"/vol": vol}, memory=32768,
              secrets=[modal.Secret.from_name("hf-secret")])
def evaluate(ckpt: str, tasks: str, limit: int, tag: str, num_fewshot: int | None,
             batch_size: int, max_gen_toks: int, hf_repo: str) -> str:
    import json
    import os
    import time

    sys.path.insert(0, "/root/src")
    from lm_eval import simple_evaluate

    from osrt.lm_eval_wrapper import OSRTLMEval

    vol.reload()
    path = f"/vol/trunk/{ckpt}"
    if not os.path.exists(path) and hf_repo:
        # Not on this workspace's volume: pull the full checkpoint from the
        # private HF mirror (Modal -> HF is fast; the laptop path is not).
        from huggingface_hub import hf_hub_download
        print(f"[lm_eval] {path} absent; downloading trunk/{ckpt} from {hf_repo}",
              flush=True)
        path = hf_hub_download(hf_repo, f"trunk/{ckpt}", local_dir="/root/ckpt")
    assert os.path.exists(path), path
    wrapper = OSRTLMEval(ckpt_path=path, tokenizer_path="/root/tokenizer",
                         batch_size=batch_size, base_model=True,
                         max_gen_toks=max_gen_toks)
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    t0 = time.time()
    res = simple_evaluate(model=wrapper, tasks=task_list,
                          limit=(None if limit == 0 else limit),
                          num_fewshot=num_fewshot, log_samples=False)
    results = res.get("results", {})
    os.makedirs("/vol/evals", exist_ok=True)
    out = f"/vol/evals/{ckpt.removesuffix('.pt')}_{tag}.json"
    with open(out, "w") as f:
        json.dump({"ckpt": ckpt, "tasks": task_list, "limit": limit,
                   "num_fewshot": num_fewshot, "results": results,
                   "secs": time.time() - t0}, f, indent=1, default=str)
    vol.commit()
    lines = [f"{ckpt} | limit={limit or 'full'} | fewshot={num_fewshot} | "
             f"{time.time() - t0:.0f}s | -> {out}"]
    for task, r in results.items():
        keep = {k: v for k, v in r.items()
                if any(k.startswith(m) for m in ("acc", "exact_match", "pass@"))
                and "stderr" not in k}
        cells = [f"{k}={v:.4f}" for k, v in keep.items() if isinstance(v, (int, float))]
        lines.append(f"  {task:22s} " + "  ".join(cells))
    return "\n".join(lines)


@app.local_entrypoint()
def main(ckpt: str = "osrt_final.pt",
         tasks: str = "hellaswag,arc_easy,arc_challenge,piqa,winogrande",
         limit: int = 0, tag: str = "base", num_fewshot: int = -1,
         batch_size: int = 8, max_gen_toks: int = 256,
         hf_repo: str = "HallD/OSRT-Ostinato-trunk"):
    print(evaluate.remote(ckpt, tasks, limit, tag,
                          None if num_fewshot < 0 else num_fewshot,
                          batch_size, max_gen_toks, hf_repo))
