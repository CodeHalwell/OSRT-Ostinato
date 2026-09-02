"""Plain-Python entry point for pretraining outside Modal.

The Modal app (`app.py`) wraps `run_training` in `@app.function`
decorators bound to a Modal Volume for checkpoint persistence. Other
hosts (Lightning AI Studios, on-prem, EC2 spot, anything with a GPU
and persistent disk) don't need any of that — `run_training` is pure
PyTorch and only touches `vol.commit()` after each save, which becomes
a no-op when the underlying disk is already persistent.

Usage (Lightning Studio, EC2, or local with a CUDA GPU):

    # Required env: WANDB_API_KEY, HF_TOKEN
    python -m osrt.train_main \\
        --tokenizer-path ./tokenizer \\
        --ckpt-dir ./checkpoints/v5

Resumes automatically from the highest `osrt_step_N.pt` /
`osrt_rescue_step_N.pt` in `--ckpt-dir`. To start fresh, point at
an empty directory.

For a 1200-step Foundation-matched smoke test (~1h on H100, ~1.6h on
A100 80GB), pass `--total-steps 1200` and a separate ckpt dir to keep
it isolated from the production run.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from osrt.config import OSRTConfig
from osrt.train import run_training
from osrt.train_config import PretrainConfig

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class _LocalVol:
    """Volume stub that satisfies `run_training`'s `vol.commit()` calls
    on hosts where the checkpoint directory is already on persistent
    storage (Lightning Studio's `/teamspace/...`, an EBS volume, a
    local SSD). Modal's `vol.commit()` flushes Modal Volume writes to
    the backing object store; on a host with persistent disk the writes
    are already durable, so `commit` is a no-op.
    """

    def commit(self) -> None:  # noqa: D401 — verbatim shim for Modal API
        return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pretrain OSRT v7 outside Modal (Colab, on-prem, etc.)",
    )
    p.add_argument(
        "--tokenizer-path",
        default=os.environ.get("OSRT_TOKENIZER_PATH", "./tokenizer"),
        help="Local path to the HF tokenizer directory (default: ./tokenizer "
             "or $OSRT_TOKENIZER_PATH).",
    )
    p.add_argument(
        "--ckpt-dir",
        default=os.environ.get("OSRT_CKPT_DIR", "./checkpoints/v7"),
        help="Directory for ckpts. Resumed from the highest step file here.",
    )
    p.add_argument(
        "--hf-repo",
        default=os.environ.get("OSRT_HF_CKPT_REPO", ""),
        help="Private HF repo id for cross-session checkpoint sync (e.g. "
             "'user/osrt-v7-ckpt'). Colab wipes local disk when the VM is "
             "released, so a multi-session run MUST persist off-VM. When set: "
             "pull the newest checkpoint before training, push each new one as "
             "it is written, and flush the tail before exit. Needs HF_TOKEN.",
    )
    p.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override PretrainConfig.total_steps (default 300000). Useful "
             "for sanity / partial-budget runs.",
    )
    p.add_argument(
        "--wandb-run-name",
        default=None,
        help="Override the W&B run name. Defaults to PretrainConfig value.",
    )
    p.add_argument(
        "--wandb-run-id",
        default=None,
        help="Resume an existing W&B run by id (e.g. when resuming after a "
             "credit-driven kill so the dashboard stays one continuous run).",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging (still prints to stdout).",
    )
    p.add_argument(
        "--micro-batch-scale",
        type=float,
        default=1.0,
        help=(
            "Scale every phase's micro-batch (accumulation adjusts to hold "
            "tokens/step). Defaults fit a 192 GB B200; use 0.5 on a 96 GB "
            "RTX PRO 6000, 0.25 on an 80 GB H100."
        ),
    )
    return p.parse_args(argv)


def _build_model_config(tokenizer_path: str) -> OSRTConfig:
    """Load the tokenizer once to seed vocab/special-token IDs into the config.

    Two distinct vocab numbers, and conflating them is a real bug:

      * `real_vocab_size` — the tokenizer's true size. Logits are sliced to it,
        so it must match `len(tok)` exactly or the model can emit ids the
        tokenizer cannot decode.
      * `vocab_size` — the embedding/head row count, padded up to a multiple of
        128 for tensor cores. Padding rows are never valid targets.

    An earlier revision set BOTH to `len(tok)`, silently discarding the
    preset's padding.
    """
    from transformers import AutoTokenizer

    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer directory not found at {tokenizer_path}. Build it with "
            f"`python scripts/build_tokenizer_v7.py --out {tokenizer_path}`.",
        )
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    # Fail closed on the tokenizer BEFORE building a model around it: pins
    # the real vocab size and every structural token id, so a swapped or
    # half-built tokenizer cannot start a run (a checkpoint trained at one
    # vocab cannot load at another).
    from osrt.tokenizer_contract import validate_tokenizer_contract
    validate_tokenizer_contract(tok)
    real = len(tok)
    padded = ((real + 127) // 128) * 128
    print(f"Tokenizer loaded: real_vocab_size={real} -> vocab_size={padded}",
          flush=True)

    from osrt.presets import OSRT_V7, build_config
    if real != OSRT_V7["real_vocab_size"]:
        print(
            f"WARNING: tokenizer has {real} tokens but the OSRT_V7 preset "
            f"expects {OSRT_V7['real_vocab_size']}. The model will follow the "
            f"tokenizer. If this is not a deliberate tokenizer change, stop: "
            f"a checkpoint trained at one vocab cannot load at another.",
            flush=True,
        )
    return build_config(
        vocab_size=padded,
        real_vocab_size=real,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if not torch.cuda.is_available():
        print(
            "ERROR: pretraining requires a CUDA GPU. CPU runs are not "
            "supported (the kernels are torch.compile + bf16 autocast "
            "throughout). Spin up a CUDA-enabled host before retrying.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    model_config = _build_model_config(args.tokenizer_path)

    train_cfg = PretrainConfig()
    if args.total_steps is not None:
        train_cfg.total_steps = args.total_steps
    train_cfg.scale_micro_batches(args.micro_batch_scale)
    if args.wandb_run_name is not None:
        train_cfg.wandb_run_name = args.wandb_run_name
    if args.wandb_run_id is not None:
        train_cfg.wandb_run_id = args.wandb_run_id
    if args.no_wandb:
        train_cfg.wandb_log = False

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Cross-session chaining. run_training's resume-scan globs
    # {ckpt_dir}/osrt_step_*.pt, so pulling the newest file into that directory
    # before the scan is the whole mechanism — no trainer change needed.
    sync = None
    if args.hf_repo:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from scripts import hf_ckpt_sync as sync
        print(f"HF ckpt sync: {args.hf_repo} (prefix 'osrt')", flush=True)
        sync.pull_latest(args.hf_repo, args.ckpt_dir, "osrt")
        sync.start_push_daemon(args.hf_repo, args.ckpt_dir, "osrt")
    print(
        f"Pretrain (Muon hybrid + aux). Tokenizer={args.tokenizer_path}, "
        f"ckpt_dir={args.ckpt_dir}, total_steps={train_cfg.total_steps}",
        flush=True,
    )

    try:
        run_training(
            model_config=model_config,
            train_cfg=train_cfg,
            vol=_LocalVol(),
            tokenizer_name=args.tokenizer_path,
            ckpt_dir=args.ckpt_dir,
        )
    finally:
        # The push daemon polls, and both the rescue path and the end-of-run
        # path exit within one interval of their final save — so the single
        # most important checkpoint is the one most likely to miss its upload
        # window. Flush synchronously, and do it in `finally` so a crash or a
        # Colab pre-emption still persists the tail. (ckpt-sync §2)
        if sync is not None:
            print("flushing checkpoints to HF...", flush=True)
            sync.flush(args.hf_repo, args.ckpt_dir, "osrt")


if __name__ == "__main__":
    main()
