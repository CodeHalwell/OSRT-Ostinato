"""Modal entry point — v7 ladder runs.

Deliberately narrow. The v6 app.py dispatched a registry of ~20 stages
(pretrain, midtrain x3, SFT v1-v4, GRPO x4, evals); none of those pipelines
survived into v7, so rebuilding that surface would be recreating debt. This
runs the gate that actually blocks the trunk run, and nothing else.

    modal run app.py --arm a          # one ladder arm
    modal run app.py --arm a --spawn  # detached (long runs)

Each arm is a separate invocation on purpose: with N x $30 workspaces the arms
run in parallel, one per workspace, and a arm that dies takes only itself down.

Secrets required: `osrt-secrets` containing HF_TOKEN and WANDB_API_KEY.
"""
from __future__ import annotations

import modal

APP = "osrt-v7-ladder"
GPU = "H100"
TIMEOUT_H = 6

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0", "transformers>=4.57", "datasets>=3.0",
        "huggingface_hub>=0.35", "wandb", "numpy", "safetensors",
    )
    .add_local_dir("src", "/root/src")
    .add_local_dir("tokenizer", "/root/tokenizer")
)

app = modal.App(APP, image=image)
vol = modal.Volume.from_name("osrt-v7-ladder-ckpt", create_if_missing=True)


@app.function(
    gpu=GPU,
    timeout=TIMEOUT_H * 3600,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("osrt-secrets")],
)
def ladder_arm(arm: str, total_steps: int, seq_len: int) -> dict:
    """Run one G3a arm. Active params are identical across arms; only the
    expert count (and hence total) changes, so any loss-per-token difference
    is attributable to total capacity alone."""
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch

    from osrt.presets import LADDER_ARMS, build_config
    from osrt.train import run_training
    from osrt.train_config import PretrainConfig

    if arm not in LADDER_ARMS:
        raise SystemExit(f"unknown arm {arm!r}; expected one of {list(LADDER_ARMS)}")

    cfg = build_config(LADDER_ARMS[arm])
    with torch.device("meta"):
        from osrt.model import OSRTForCausalLM
        n_total = sum(p.numel() for p in OSRTForCausalLM(cfg).parameters())
    print(f"[ladder {arm}] experts={cfg.num_routed_experts} total={n_total:,}",
          flush=True)

    train_cfg = PretrainConfig()
    train_cfg.total_steps = total_steps
    train_cfg.warmup_steps = max(total_steps // 20, 10)
    # WSD's stable phase is the point of the switch, but a ladder arm is a
    # COMPLETE short run, not a chunk of a longer one — so it should anneal.
    train_cfg.lr_schedule = "wsd"
    train_cfg.wsd_decay_frac = 0.3
    train_cfg.dataloader_num_workers = 2
    train_cfg.wandb_run_name = f"osrt-v7-ladder-{arm}"
    for phase in getattr(train_cfg, "phases", {}).values():
        if isinstance(phase, dict) and "seq_len" in phase:
            phase["seq_len"] = seq_len

    ckpt_dir = f"/vol/ladder_{arm}"
    os.makedirs(ckpt_dir, exist_ok=True)

    class _Vol:
        def commit(self) -> None:
            vol.commit()

    run_training(
        model_config=cfg,
        train_cfg=train_cfg,
        vol=_Vol(),
        tokenizer_name="/root/tokenizer",
        ckpt_dir=ckpt_dir,
    )
    vol.commit()
    return {"arm": arm, "experts": cfg.num_routed_experts, "total_params": n_total}


@app.function(
    gpu=GPU,
    timeout=45 * 60,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("osrt-secrets")],
)
def v7_sanity(steps: int = 30) -> dict:
    """The launch gate: a hard-capped run of the REAL committed v7 shape.

    Distinct from a ladder arm — those are ~123M-active proxies, this is the
    968M shape itself. It answers the questions a proxy cannot: does the real
    config build, fit, compile, and step on an H100 with loss going down.

    Capped deliberately. There is NO full-trunk stage in this file, so no
    invocation here can start a paid multi-month run by accident; the trunk is
    launched explicitly, on a box, after G3a reports.
    """
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from transformers import AutoTokenizer

    from osrt.model import OSRTForCausalLM
    from osrt.presets import OSRT_V7, build_config
    from osrt.train import run_training
    from osrt.train_config import PretrainConfig

    steps = min(steps, 100)          # hard ceiling, not a default

    tok = AutoTokenizer.from_pretrained("/root/tokenizer")
    real = len(tok)
    padded = ((real + 127) // 128) * 128
    if real != OSRT_V7["real_vocab_size"]:
        raise SystemExit(
            f"tokenizer has {real} tokens, preset expects "
            f"{OSRT_V7['real_vocab_size']}. Refusing: a shape mismatch here "
            f"means the sanity run does not test the committed model."
        )

    cfg = build_config(
        vocab_size=padded,           # padded rows...
        real_vocab_size=real,        # ...but logits sliced to the true vocab
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
    )
    with torch.device("meta"):
        n = sum(p.numel() for p in OSRTForCausalLM(cfg).parameters())
    print(f"[v7_sanity] {steps} steps | {n:,} params | vocab {real}->{padded} | "
          f"E={cfg.num_routed_experts} top-{cfg.top_k_experts} "
          f"situ_glu={cfg.situ_glu} balance={cfg.router_balance_mode}", flush=True)

    train_cfg = PretrainConfig()
    train_cfg.total_steps = steps
    train_cfg.warmup_steps = max(steps // 5, 2)
    train_cfg.dataloader_num_workers = 2
    train_cfg.wandb_log = False       # a 30-step gate is not a run worth logging
    ckpt_dir = "/vol/v7_sanity"
    os.makedirs(ckpt_dir, exist_ok=True)

    class _Vol:
        def commit(self) -> None:
            vol.commit()

    run_training(
        model_config=cfg, train_cfg=train_cfg, vol=_Vol(),
        tokenizer_name="/root/tokenizer", ckpt_dir=ckpt_dir,
    )
    vol.commit()
    return {"stage": "v7_sanity", "steps": steps, "params": n}


@app.local_entrypoint()
def main(arm: str = "a", total_steps: int = 4000, seq_len: int = 2048,
         spawn: bool = False, sanity: bool = False) -> None:
    """Two stages only: the launch gate, and one ladder arm.

    There is deliberately no trunk stage — the paid multi-month run is not
    reachable from this file.
    """
    if sanity:
        print(v7_sanity.remote(min(total_steps, 100) if total_steps < 4000 else 30))
        return
    if spawn:
        call = ladder_arm.spawn(arm, total_steps, seq_len)
        print(f"spawned ladder arm {arm}: {call.object_id}")
        print("Detached. Watch it in the Modal dashboard or W&B.")
    else:
        print(ladder_arm.remote(arm, total_steps, seq_len))
