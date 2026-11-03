"""Where does a training step's time go, and what does a token cost per GPU?

Synthetic batches (the GPU sat at 85-100% busy on real data, so the loader is
not the bottleneck), the trunk's model and optimizer (Muon + AdamW), compile
on, bf16 autocast, the midtrain shape by default. Three variants, each 3
warm-up + 5 timed steps, then a torch.profiler pass on the baseline:

  base        the trainer's config: grad checkpointing OFF (trunk peak 109 GB)
  ckpt        grad checkpointing on (a third of the memory, half the speed)
  2x-micro    twice the micro-batch, half the accumulation (same tokens/step)
The baseline step is also split into forward+backward vs optimizer time, and
dynamo graph breaks / recompiles are logged (TORCH_LOGS) so eager islands in
the MoE path are visible.

    MODAL_PROFILE=<ws> uv run modal run --detach scripts/profile_step.py --gpu B200
    (--gpu H200; --gpu H100 --micro-batch-scale 0.5 on an 80 GB card)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

# OSRT_TORCH=2.10.0 selects the old pinned torch (cu128 index, triton 3.6) for
# an A/B against the default 2.15.0 from PyPI; the inductor cache
# is keyed per torch version so the two never share compiled artefacts.
TORCH = os.environ.get("OSRT_TORCH", "2.15.0")
_torch_pins = (
    ["torch==2.10.0+cu128"], ["triton==3.6.0"],
    {"extra_options": "--index-url https://download.pytorch.org/whl/cu128"},
) if TORCH == "2.10.0" else ([f"torch=={TORCH}"], [], {})
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env({"PYTHONUNBUFFERED": "1", "TORCH_LOGS": "graph_breaks,recompiles",
          "TORCHINDUCTOR_CACHE_DIR": f"/vol/inductor_cache_torch{TORCH}"})
    .pip_install(*_torch_pins[0], **_torch_pins[2])
    .pip_install("transformers==5.3.0", *_torch_pins[1], "tokenizers==0.22.2",
                 "safetensors==0.7.0", "numpy")
    .add_local_dir(str(Path(__file__).resolve().parent.parent / "src"), "/root/src")
)
vol = modal.Volume.from_name("osrt-v7-ladder-ckpt", create_if_missing=True)
app = modal.App("osrt-profile-step", image=image)


def _run(gpu: str, batch: int, accum: int, seq: int, steps: int,
         variants: str, prof_rows: int) -> str:
    import time

    import torch
    import torch._dynamo as dynamo

    sys.path.insert(0, "/root/src")
    from osrt.model import OSRTForCausalLM
    from osrt.muon import HybridMuonAdamW, Muon, build_param_groups
    from osrt.presets import build_config

    dynamo.config.capture_scalar_outputs = True
    dynamo.config.capture_dynamic_output_shape_ops = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    cfg = build_config(vocab_size=49280, real_vocab_size=49184, bos_token_id=0,
                       eos_token_id=0, pad_token_id=0, fused_cross_entropy_chunks=8)
    lines = [f"[profile] {gpu} | {torch.cuda.get_device_name()} | "
             f"torch {torch.__version__} | cuda {torch.version.cuda}"]

    def log(s: str) -> None:
        lines.append(s)
        print(s, flush=True)

    def build(grad_ckpt: bool):
        torch.manual_seed(0)
        model = OSRTForCausalLM(cfg).to(dev)
        model.train()
        model.model._osrt_grad_ckpt = grad_ckpt
        model.set_moe_telemetry(False)
        muon_p, adamw_g = build_param_groups(model.named_parameters(), weight_decay=0.1,
                                             per_head_attn=False, head_dim=cfg.head_dim)
        opt = HybridMuonAdamW(
            Muon(muon_p, lr=3e-3, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.1),
            torch.optim.AdamW(adamw_g, lr=1e-4, betas=(0.9, 0.95), eps=1e-8))
        return torch.compile(model), opt

    def step(model, opt, b: int, a: int) -> float:
        """One optimizer step; returns the seconds spent in clip + opt.step()."""
        for _ in range(a):
            ids = torch.randint(0, 49184, (b, seq), device=dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(ids, labels=ids).loss / a
            loss.backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def timed(name: str, grad_ckpt: bool, b: int, a: int):
        try:
            model, opt = build(grad_ckpt)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(3):
                step(model, opt, b, a)
            torch.cuda.synchronize()
            warm = time.perf_counter() - t0
            t0 = time.perf_counter()
            opt_s = sum(step(model, opt, b, a) for _ in range(steps)) / steps
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / steps
            tok = b * a * seq
            peak = torch.cuda.max_memory_allocated() / 2**30
            log(f"  {name:9} {b}x{seq}x{a} | {dt:6.2f} s/step | "
                f"{tok / dt / 1e3:6.1f}K tok/s | peak {peak:5.1f} GB | "
                f"fwd+bwd {dt - opt_s:.2f}s opt {opt_s:.2f}s | warmup {warm:.0f}s")
            return model, opt
        except torch.OutOfMemoryError:
            log(f"  {name:9} {b}x{seq}x{a} | OOM")
            return None, None
        finally:
            torch.cuda.empty_cache()

    want = variants.split(",")
    base = timed("base", False, batch, accum) if "base" in want else (None, None)
    if "ckpt" in want:
        m, o = timed("ckpt", True, batch, accum)
        del m, o
    if "2x-micro" in want:
        m, o = timed("2x-micro", False, batch * 2, max(1, accum // 2))
        del m, o
    model, opt = base
    if model is not None:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step(model, opt, batch, accum)
            torch.cuda.synchronize()
        ka = prof.key_averages()
        cuda_total = sum(getattr(e, "self_device_time_total", 0) for e in ka) / 1e6
        log(f"  profiled step: CUDA kernel time {cuda_total:.2f}s "
            f"(wall from timing above; the gap is host/launch/sync overhead)")
        log(ka.table(sort_by="self_device_time_total", row_limit=prof_rows,
                     max_name_column_width=70))
    out = "\n".join(lines)
    import os
    os.makedirs("/vol/bench", exist_ok=True)
    name = f"/vol/bench/profile_{gpu}_torch{torch.__version__}.txt"
    with open(name, "w") as f:
        f.write(out + "\n")
    vol.commit()
    return out


@app.function(gpu="B200", timeout=3600, volumes={"/vol": vol})
def profile_b200(*a) -> str:
    return _run("B200", *a)


@app.function(gpu="H200", timeout=3600, volumes={"/vol": vol})
def profile_h200(*a) -> str:
    return _run("H200", *a)


@app.function(gpu="H100", timeout=3600, volumes={"/vol": vol})
def profile_h100(*a) -> str:
    return _run("H100", *a)


@app.function(gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def profile_a100(*a) -> str:
    return _run("A100-80GB", *a)


FNS = {"B200": profile_b200, "H200": profile_h200, "H100": profile_h100,
       "A100": profile_a100}


@app.local_entrypoint()
def main(gpu: str = "B200", micro_batch_scale: float = 1.0, seq: int = 4096,
         steps: int = 5, variants: str = "base,2x-micro", prof_rows: int = 25):
    batch = max(1, round(6 * micro_batch_scale))
    accum = round(11 / micro_batch_scale)
    print(FNS[gpu].remote(batch, accum, seq, steps, variants, prof_rows))
