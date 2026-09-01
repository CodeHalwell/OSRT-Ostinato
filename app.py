"""Modal entry point — v7 ladder runs.

Deliberately narrow. The v6 app.py dispatched a registry of ~20 stages
(pretrain, midtrain x3, SFT v1-v4, GRPO x4, evals); none of those pipelines
survived into v7, so rebuilding that surface would be recreating debt. This
runs the trunk, its 30-step gate, and the ladder arms.

    modal run --detach app.py --trunk-run        # THE run, detached, resumable
    modal run --detach app.py --trunk-run --hf-repo u/r   # ...also mirrored to HF
    modal run app.py --sanity                    # 30-step gate
    modal run --detach app.py --arm a --spawn    # one ladder arm

    --detach is REQUIRED with --trunk-run / --spawn: without it the CLI stops the
    ephemeral app the moment the entrypoint returns and cancels the spawned call
    ("Stopping app - local entrypoint completed."). Bit the first ladder launch.

Each arm is a separate invocation on purpose: with N x $30 workspaces the arms
run in parallel, one per workspace, and a arm that dies takes only itself down.

Secrets required: `hf-secret` (HF_TOKEN) and `wandb-secret` (WANDB_API_KEY) —
the names every existing workspace already carries from v6.
"""
from __future__ import annotations

import modal

APP = "osrt-v7-ladder"
GPU = "H100"
TIMEOUT_H = 8
TRUNK_TIMEOUT_H = 24   # Modal's ceiling; the trunk chains across invocations

# Image ported from the v6 app.py that trained the whole v6 lineage. Two parts
# are load-bearing, learned the hard way:
#   * torch 2.10.0+cu128 from the PyTorch index. The first ladder launch pinned
#     torch==2.8.0 and every arm died in Inductor's C++ codegen at the first
#     compiled forward ("'zuf3' was not declared in this scope"): 2.8.0 cannot
#     emit unbacked *float* symbols, which the router telemetry's .item() calls
#     become under capture_scalar_outputs (needed for grouped GEMM). v6 ran the
#     same code path on 2.10 without incident.
#   * TOKENIZERS_PARALLELISM=false and expandable_segments — a fork deadlock at
#     "Fetching first batch..." and a fragmentation OOM respectively, both seen
#     on v6 and both fixed by these.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .pip_install(
        "torch==2.10.0+cu128",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==5.3.0",
        "datasets==4.6.1",
        "triton==3.6.0",
        "wandb==0.25.1",
        "tokenizers==0.22.2",
        "safetensors==0.7.0",
        "huggingface_hub>=0.35",
        "numpy",
    )
    .add_local_dir("src", "/root/src")
    .add_local_dir("tokenizer", "/root/tokenizer")
    .add_local_dir("scripts", "/root/scripts")
)

app = modal.App(APP, image=image)
vol = modal.Volume.from_name("osrt-v7-ladder-ckpt", create_if_missing=True)


@app.function(
    gpu=GPU,
    timeout=TIMEOUT_H * 3600,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-secret"),
             modal.Secret.from_name("wandb-secret")],
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
    secrets=[modal.Secret.from_name("hf-secret"),
             modal.Secret.from_name("wandb-secret")],
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


@app.function(
    gpu=GPU,
    timeout=TRUNK_TIMEOUT_H * 3600,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-secret"),
             modal.Secret.from_name("wandb-secret")],
)
def trunk(hf_repo: str = "", total_steps: int | None = None) -> dict:
    """The v7 pretraining run. Committed without gates — see roadmap §19 for
    the bets this embodies and what would falsify each.

    Resumable by design: checkpoints land on the volume, so re-invoking picks
    up from the highest step. Set hf_repo to ALSO mirror them to a private HF
    repo, which is what lets the same run continue from Colab.
    """
    import os
    import sys

    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from transformers import AutoTokenizer

    from osrt.presets import OSRT_V7, build_config
    from osrt.tokenizer_contract import validate_tokenizer_contract
    from osrt.train import run_training
    from osrt.train_config import PretrainConfig

    tok = AutoTokenizer.from_pretrained("/root/tokenizer")
    validate_tokenizer_contract(tok)
    real = len(tok)
    padded = ((real + 127) // 128) * 128
    if real != OSRT_V7["real_vocab_size"]:
        raise SystemExit(f"tokenizer {real} != preset {OSRT_V7['real_vocab_size']}")
    cfg = build_config(
        vocab_size=padded, real_vocab_size=real,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )
    train_cfg = PretrainConfig()
    if total_steps is not None:
        train_cfg.total_steps = total_steps
    train_cfg.dataloader_num_workers = 2
    train_cfg.wandb_run_name = "osrt-v7-trunk"
    print(f"[trunk] {train_cfg.total_steps} steps ≈ "
          f"{train_cfg.total_tokens()/1e9:.2f}B tokens | "
          f"E={cfg.num_routed_experts} top-{cfg.top_k_experts} h{cfg.expert_hidden} | "
          f"vocab {real}->{padded}", flush=True)

    ckpt_dir = "/vol/trunk"
    os.makedirs(ckpt_dir, exist_ok=True)
    sync = None
    if hf_repo:
        from scripts import hf_ckpt_sync as sync
        sync.pull_latest(hf_repo, ckpt_dir, "osrt")
        sync.start_push_daemon(hf_repo, ckpt_dir, "osrt")

    class _Vol:
        def commit(self) -> None:
            vol.commit()

    try:
        run_training(model_config=cfg, train_cfg=train_cfg, vol=_Vol(),
                     tokenizer_name="/root/tokenizer", ckpt_dir=ckpt_dir)
    finally:
        vol.commit()
        if sync is not None:
            sync.flush(hf_repo, ckpt_dir, "osrt")
    return {"stage": "trunk", "steps": train_cfg.total_steps}


@app.local_entrypoint()
def main(arm: str = "a", total_steps: int = 8000, seq_len: int = 2048,
         spawn: bool = False, sanity: bool = False, sanity_steps: int = 30,
         trunk_run: bool = False, hf_repo: str = "") -> None:
    """Stages: --trunk (the run), --sanity (30-step gate), or one ladder --arm.

    total_steps=8000 at seq 2048 is ~1.07B tokens per ladder arm. The trunk
    uses PretrainConfig's own budget (17,500 steps ≈ 5.28B tokens) unless
    --total-steps is given.
    """
    if trunk_run:
        call = trunk.spawn(hf_repo, None if total_steps == 8000 else total_steps)
        print(f"spawned trunk: {call.object_id} — resumes from /vol/trunk on re-invoke")
        return
    if sanity:
        print(v7_sanity.remote(sanity_steps))
        return
    if spawn:
        call = ladder_arm.spawn(arm, total_steps, seq_len)
        print(f"spawned ladder arm {arm}: {call.object_id}")
        print("Detached. Watch it in the Modal dashboard or W&B.")
    else:
        print(ladder_arm.remote(arm, total_steps, seq_len))
