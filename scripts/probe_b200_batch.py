"""Micro-batch sweep at the trunk shape on a B200: for each phase seq_len, step
the micro-batch up until OOM, with the real fp32 params + bf16 autocast + Muon/
AdamW state in place (eager, no compile — conservative on memory). Reports
peak GB and tok/s per point so per-phase batch_size / grad_accum can be set
from a measurement rather than a guess.

    MODAL_PROFILE=danielhalwell uv run modal run scripts/probe_b200_batch.py --gpu B200
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env({"PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})  # noqa: E501
    .pip_install(
        "torch==2.10.0+cu128",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==5.3.0", "triton==3.6.0", "safetensors==0.7.0",
        "huggingface_hub>=0.35", "numpy",
    )
    .add_local_dir(str(ROOT / "src"), "/root/src")
)
app = modal.App("osrt-b200-batch-probe", image=image)

SWEEP = {  # seq_len -> (grad_ckpt, [micro-batches to try, ascending])
    2048: [(False, [8, 16, 24, 32, 48])],
    4096: [(False, [4, 8, 12, 16, 24])],
    8192: [(False, [2, 4, 6, 8]), (True, [4, 8, 12, 16])],
}
CE_CHUNKS = 8   # chunked linear-CE (fused_ce.py): ~1/8 of the (N, vocab) logits live at once  # noqa: E501


def _run(gpu: str) -> str:
    import sys
    import time

    import torch

    sys.path.insert(0, "/root/src")
    from osrt.model import OSRTForCausalLM
    from osrt.muon import HybridMuonAdamW, Muon, build_param_groups
    from osrt.presets import build_v7_config
    from osrt.train_config import PretrainConfig

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9
    cfg = build_v7_config()
    tcfg = PretrainConfig()
    torch.manual_seed(0)
    cfg.fused_cross_entropy_chunks = CE_CHUNKS
    model = OSRTForCausalLM(cfg).to(dev)
    model.train()
    # Compile exactly as train.py does (grouped-GEMM dynamo flags + whole model).
    import torch._dynamo as _dynamo
    _dynamo.config.capture_scalar_outputs = True
    _dynamo.config.capture_dynamic_output_shape_ops = True
    _dynamo.config.cache_size_limit = 64
    cmodel = torch.compile(model)
    muon_params, adamw_groups = build_param_groups(
        model.named_parameters(), weight_decay=tcfg.weight_decay,
        per_head_attn=tcfg.per_head_muon, head_dim=cfg.head_dim)
    muon = Muon(muon_params, lr=getattr(tcfg, "muon_lr", tcfg.peak_lr),
                momentum=getattr(tcfg, "muon_momentum", 0.95), nesterov=True,
                ns_steps=tcfg.muon_ns_steps, ns_stable_steps=tcfg.muon_ns_stable_steps,
                update_rms=tcfg.muon_update_rms, weight_decay=tcfg.weight_decay)
    adamw = torch.optim.AdamW(adamw_groups, lr=tcfg.peak_lr, betas=(0.9, 0.95), eps=1e-8)  # noqa: E501
    opt = HybridMuonAdamW(muon, adamw)
    n_params = sum(p.numel() for p in model.parameters())
    lines = [f"[{props.name}] {total_gb:.0f} GB | params {n_params:,} | torch.compile, fp32 params, bf16 autocast, grouped GEMM={getattr(cfg, 'moe_grouped_gemm', None)}, fused CE chunks={CE_CHUNKS}"]  # noqa: E501
    static_gb = torch.cuda.memory_allocated() / 1e9
    lines.append(f"  static (params+grads-to-be+opt state after 1st step) measured below; params alone {static_gb:.1f} GB")  # noqa: E501

    def step(B: int, S: int) -> float:
        ids = torch.randint(0, cfg.real_vocab_size, (B, S), device=dev)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = cmodel(ids, labels=ids)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for S, variants in SWEEP.items():
        for ckpt, batches in variants:
            model.model._osrt_grad_ckpt = ckpt
            lines.append(f"== seq {S} | grad_ckpt={ckpt}")
            for B in batches:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                try:
                    t_c = step(B, S)                # includes compile/trace for this shape  # noqa: E501
                    step(B, S)                      # allocator warm
                    torch.cuda.reset_peak_memory_stats()
                    ts = [step(B, S) for _ in range(2)]
                    dt = sum(ts) / len(ts)
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    lines.append(
                        f"   B={B:3d}  tokens/microbatch={B * S:8,d}  peak {peak:6.1f} GB "  # noqa: E501
                        f"({peak / total_gb:4.0%})  {dt:6.2f} s/step  {B * S / dt:9,.0f} tok/s  (compile {t_c:5.0f}s)")  # noqa: E501
                except torch.OutOfMemoryError:
                    lines.append(f"   B={B:3d}  OOM")
                    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    break
    return "\n".join(lines)


@app.function(gpu="B200", timeout=3600)
def probe_b200() -> str:
    return _run("B200")


@app.function(gpu="H100", timeout=3600)
def probe_h100() -> str:
    return _run("H100")


@app.local_entrypoint()
def main(gpu: str = "B200"):
    fn = probe_b200 if gpu.upper() == "B200" else probe_h100
    print(fn.remote())
