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

Resumes automatically from the highest `osrt_v5_step_N.pt` /
`osrt_v5_rescue_step_N.pt` in `--ckpt-dir`. To start fresh, point at
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
        description="Pretrain OSRT v5 outside Modal (Lightning, on-prem, etc.)",
    )
    p.add_argument(
        "--tokenizer-path",
        default=os.environ.get("OSRT_TOKENIZER_PATH", "./tokenizer"),
        help="Local path to the HF tokenizer directory (default: ./tokenizer "
             "or $OSRT_TOKENIZER_PATH).",
    )
    p.add_argument(
        "--ckpt-dir",
        default=os.environ.get("OSRT_CKPT_DIR", "./checkpoints/v5"),
        help="Directory for ckpts. Resumed from the highest step file here.",
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
    return p.parse_args(argv)


def _build_model_config(tokenizer_path: str) -> OSRTConfig:
    """Load the tokenizer once to seed vocab/special-token IDs into the
    model config. Mirrors the same construction used by `app.py::pretrain`."""
    from transformers import AutoTokenizer

    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer directory not found at {tokenizer_path}. "
            f"Either sync it from Modal "
            f"(`modal volume get osrt-v4-tokenizer / ./tokenizer/`) or "
            f"retrain via `scripts/train_tokenizer.py`.",
        )
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}", flush=True)

    expected_vocab = 65536
    if len(tok) != expected_vocab:
        print(
            f"WARNING: expected vocab {expected_vocab} but got {len(tok)}. "
            f"The model config will follow the tokenizer's actual vocab.",
            flush=True,
        )

    from osrt.presets import build_config
    return build_config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
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
    if args.wandb_run_name is not None:
        train_cfg.wandb_run_name = args.wandb_run_name
    if args.wandb_run_id is not None:
        train_cfg.wandb_run_id = args.wandb_run_id
    if args.no_wandb:
        train_cfg.wandb_log = False

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(
        f"Pretrain (Muon hybrid + aux). Tokenizer={args.tokenizer_path}, "
        f"ckpt_dir={args.ckpt_dir}, total_steps={train_cfg.total_steps}",
        flush=True,
    )

    run_training(
        model_config=model_config,
        train_cfg=train_cfg,
        vol=_LocalVol(),
        tokenizer_name=args.tokenizer_path,
        ckpt_dir=args.ckpt_dir,
    )


if __name__ == "__main__":
    main()
