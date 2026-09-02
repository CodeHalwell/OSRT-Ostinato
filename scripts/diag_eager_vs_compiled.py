"""Diagnose a suspicious training-loss drop: load a trunk checkpoint from the
volume and score the SAME knowledge-mix batches under (a) eager train mode,
(b) torch.compile train mode (trainer flags), (c) eager eval mode, plus a
random-token batch (a causal LM must score ~ln(V) = 10.8; far lower = leak).

    MODAL_PROFILE=inference-syn uv run modal run scripts/diag_eager_vs_compiled.py --step 1000
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("torch==2.10.0+cu128",
                 extra_options="--index-url https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.3.0", "datasets==4.6.1", "triton==3.6.0",
                 "tokenizers==0.22.2", "safetensors==0.7.0", "huggingface_hub>=0.35",
                 "numpy")
    .add_local_dir(str(ROOT / "src"), "/root/src")
    .add_local_dir(str(ROOT / "tokenizer"), "/root/tokenizer")
)
vol = modal.Volume.from_name("osrt-v7-ladder-ckpt")
app = modal.App("osrt-diag-eager-compiled", image=image)


@app.function(gpu="H100", timeout=1800, volumes={"/vol": vol},
              secrets=[modal.Secret.from_name("hf-secret")])
def diag(step: int, n_batches: int) -> str:
    import math
    import sys

    import torch

    sys.path.insert(0, "/root/src")
    from osrt.data import make_loader
    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_v7_config
    from osrt.train_config import PretrainConfig

    vol.reload()   # a container can start before the trainer's last commit is visible
    cfg = build_v7_config()
    cfg.fused_cross_entropy_chunks = 8
    model = OSRTForCausalLM(cfg)
    ck = torch.load(f"/vol/trunk/osrt_step_{step}.pt", map_location="cpu", weights_only=False)  # noqa: E501
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model = model.cuda()
    tcfg = PretrainConfig()
    ph = tcfg.phases["knowledge"]
    loader = make_loader(dataset_configs=ph["datasets"], seq_len=4096,
                         tokenizer_name="/root/tokenizer", batch_size=2,
                         step_num=step + 7, num_workers=0)
    it = iter(loader)
    batches = [next(it) for _ in range(n_batches)]
    out = [f"[step {step}] {n_batches} knowledge-mix batches of 2x4096"]

    def score(m, train_mode: bool, tag: str) -> float:
        m.train(train_mode)
        losses = []
        with torch.no_grad():
            for ids, labels in batches:
                ids, labels = ids.cuda(), labels.cuda()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    o = m(ids, labels=labels)
                inner = m._orig_mod if hasattr(m, "_orig_mod") else m
                task = inner.last_task_loss
                losses.append(float(task if task is not None else o.loss))
        mean = sum(losses) / len(losses)
        out.append(f"  {tag:28} task loss mean {mean:.4f}  per-batch {[round(x, 3) for x in losses]}")  # noqa: E501
        return mean

    a = score(model, True, "eager, train mode")
    c = score(model, False, "eager, eval mode")
    # random tokens: causal LM ~ ln(V); a leak scores far lower
    torch.manual_seed(0)
    rnd = torch.randint(0, cfg.real_vocab_size, (2, 4096))
    model.train(True)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        o = model(rnd.cuda(), labels=rnd.cuda())
    rl = float(model.last_task_loss if model.last_task_loss is not None else o.loss)
    out.append(f"  random tokens (train mode)   task loss {rl:.4f}   ln(V)={math.log(cfg.real_vocab_size):.3f}")  # noqa: E501
    import torch._dynamo as _dynamo
    _dynamo.config.capture_scalar_outputs = True
    _dynamo.config.capture_dynamic_output_shape_ops = True
    cm = torch.compile(model)
    b = score(cm, True, "compiled, train mode")
    out.append(f"  eager-vs-compiled gap (train): {a - b:+.4f} nats")
    return "\n".join(out)


@app.local_entrypoint()
def main(step: int = 1000, n_batches: int = 6):
    print(diag.remote(step, n_batches))
