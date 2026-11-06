"""The SFT probe outside Modal (Colab RTX PRO 6000 etc.): HRA adapters on a
FROZEN base pulled from an HF repo, merged checkpoint (+ bf16 safetensors)
pushed back under sft/. Mirrors app.py::sft_probe.

    PYTHONPATH=src python scripts/run_sft.py \\
        --base-repo HallD/OSRT-Ostinato-midtrain-v4 --base-file soup/osrt_soup_X.pt \\
        --run-name osrt-v7-sft-v4 --push
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-repo", required=True)
    p.add_argument("--base-file", required=True)
    p.add_argument("--tokenizer", default="./tokenizer")
    p.add_argument("--out-dir", default="./checkpoints/sft")
    p.add_argument("--steps", type=int, default=0, help="0 = SFTProbeConfig default")
    p.add_argument("--run-name", default="")
    p.add_argument(
        "--micro-batch-scale",
        type=float,
        default=1.0,
        help="1.0 = B200 shape (8x4096); 0.25 on a 96 GB RTX PRO 6000. "
        "Accumulation adjusts to hold tokens/step.",
    )
    p.add_argument(
        "--push", action="store_true", help="upload merged ckpt to --base-repo"
    )
    args = p.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from osrt.presets import OSRT_V7, build_config
    from osrt.sft_train import run_sft
    from osrt.tokenizer_contract import validate_tokenizer_contract
    from osrt.train_config import SFTProbeConfig

    cfg = SFTProbeConfig()
    cfg.base_hf_repo, cfg.base_file = args.base_repo, args.base_file
    if args.steps:
        cfg.total_steps = args.steps
    if args.run_name:
        cfg.wandb_run_name = args.run_name
    if args.micro_batch_scale != 1.0:
        tokens = cfg.tokens_per_step()
        cfg.batch_size = max(1, round(cfg.batch_size * args.micro_batch_scale))
        cfg.grad_accum_steps = max(1, round(tokens / (cfg.seq_len * cfg.batch_size)))
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    validate_tokenizer_contract(tok)
    real = len(tok)
    if real != OSRT_V7["real_vocab_size"]:
        raise SystemExit(f"tokenizer {real} != preset {OSRT_V7['real_vocab_size']}")
    model_cfg = build_config(
        vocab_size=((real + 127) // 128) * 128,
        real_vocab_size=real,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
        router_bias_in_gates=cfg.legacy_gates,
        mtp_loss_weight=0.0,
    )
    base_path = hf_hub_download(
        cfg.base_hf_repo, cfg.base_file, local_dir="./checkpoints/base"
    )
    out_dir = os.path.join(args.out_dir, cfg.wandb_run_name)
    print(
        f"[sft] {cfg.total_steps} steps x {cfg.tokens_per_step():,} tokens "
        f"({cfg.batch_size}x{cfg.seq_len}x{cfg.grad_accum_steps}) | "
        f"base {base_path}",
        flush=True,
    )
    result = run_sft(model_cfg, cfg, args.tokenizer, base_path, out_dir)
    print(result, flush=True)
    if not args.push:
        return
    api = HfApi()
    merged = result["merged"]
    name = f"sft/{cfg.wandb_run_name}_merged_step_{cfg.total_steps}"
    api.upload_file(
        path_or_fileobj=merged,
        path_in_repo=f"{name}.pt",
        repo_id=cfg.base_hf_repo,
        commit_message=name,
    )
    sd = torch.load(merged, map_location="cpu", weights_only=False)["model_state_dict"]
    bf = {
        k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous()
        for k, v in sd.items()
    }
    st = os.path.join(out_dir, os.path.basename(name) + "_bf16.safetensors")
    save_file(bf, st, metadata={"format": "pt", "sft": cfg.wandb_run_name})
    api.upload_file(
        path_or_fileobj=st,
        path_in_repo=f"{name}_bf16.safetensors",
        repo_id=cfg.base_hf_repo,
        commit_message=f"{name} bf16",
    )
    print(f"pushed {cfg.base_hf_repo}/{name}.pt (+ bf16 safetensors)", flush=True)


if __name__ == "__main__":
    main()
