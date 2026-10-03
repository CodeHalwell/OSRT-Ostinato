"""SFT with HRA adapters on a frozen OSRT base (docs/specs/2026-10-03-sft-probe.md).

    run_sft(model_config, cfg, tokenizer_name, base_ckpt_path, out_dir)

* builds the model exactly as the trunk computes it (`router_bias_in_gates`
  from `cfg.legacy_gates`, MTP loss weight 0), loads the base STRICTLY;
* `inject_hra(freeze_pretrained=True)`: the adapters are the only trainable
  parameters — asserted, not assumed;
* Gumbel tau 0 and balance accumulation off, so routing is the base's routing
  and the Quantile-Balancing bias stays where pretraining left it;
* AdamW on the adapters, linear warmup then cosine to `lr_min`;
* assistant-only loss from `osrt.sft_data.SFTStream`;
* every `ckpt_interval` steps: adapters-only checkpoint; at the end: the
  adapters merged into the base (`merge_hra`) as a full `model_state_dict`
  that a plain `OSRTForCausalLM` loads — the evaluators need nothing new.
"""

from __future__ import annotations

import math
import os
import time

import torch
from torch import nn

from osrt.config import OSRTConfig
from osrt.hra import inject_hra, merge_hra
from osrt.model import OSRTForCausalLM
from osrt.sft_data import IGNORE_INDEX, make_sft_loader
from osrt.train import load_model_state_or_raise, run_eval, set_router_gumbel_tau
from osrt.train_config import SFTProbeConfig


def _lr_at(step: int, cfg: SFTProbeConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    span = max(1, cfg.total_steps - cfg.warmup_steps)
    p = min(1.0, (step - cfg.warmup_steps) / span)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * p))


def _freeze_base_routing(model: nn.Module) -> None:
    """The base's routing, unchanged: no Gumbel, no bias updates."""
    set_router_gumbel_tau(model, 0.0)
    for m in model.modules():
        if hasattr(m, "balance_accum_enabled"):
            m.balance_accum_enabled = False


def build_sft_model(
    model_config: OSRTConfig, cfg: SFTProbeConfig, base_ckpt_path: str,
    device: torch.device,
) -> tuple[OSRTForCausalLM, list[nn.Parameter]]:
    model = OSRTForCausalLM(model_config)
    ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    load_model_state_or_raise(model, ckpt.get("model_state_dict", ckpt),
                              context=f"sft base {base_ckpt_path}")
    del ckpt
    model.to(device)
    hra_params = inject_hra(model, rank=cfg.hra_rank, scale=cfg.hra_scale,
                            freeze_pretrained=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_hra = sum(p.numel() for p in hra_params)
    if n_train != n_hra:
        raise RuntimeError(f"frozen-base violated: {n_train:,} trainable vs "
                           f"{n_hra:,} adapter params")
    _freeze_base_routing(model)
    return model, hra_params


@torch.no_grad()
def _holdout_loss(model, loader_batches, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for input_ids, labels in loader_batches:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids, labels=labels)
        k = int((labels != IGNORE_INDEX).sum().item())
        if k:
            tot += float(out.loss) * k
            n += k
    model.train()
    return tot / max(n, 1)


def run_sft(
    model_config: OSRTConfig,
    cfg: SFTProbeConfig,
    tokenizer_name: str,
    base_ckpt_path: str,
    out_dir: str,
    vol=None,
    wandb_run_id: str = "",
) -> dict:
    device = torch.device("cuda")
    os.makedirs(out_dir, exist_ok=True)
    model, hra_params = build_sft_model(model_config, cfg, base_ckpt_path, device)
    model.train()
    _freeze_base_routing(model)  # .train() must not re-enable anything

    opt = torch.optim.AdamW(hra_params, lr=cfg.lr, betas=cfg.betas,
                            weight_decay=cfg.weight_decay)

    train_loader = make_sft_loader(cfg.train_datasets(), cfg.seq_len, tokenizer_name,
                                   cfg.batch_size, seed=1234)
    hold_loader = make_sft_loader(cfg.holdout_datasets(), cfg.seq_len, tokenizer_name,
                                  cfg.batch_size, seed=4321)
    hold_iter = iter(hold_loader)
    holdout = [next(hold_iter) for _ in range(cfg.eval_steps)]
    del hold_iter, hold_loader

    try:
        import wandb
        wb = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                        id=wandb_run_id or None, resume="allow",
                        config={"sft": cfg.__dict__, "base": base_ckpt_path})
    except Exception as exc:  # noqa: BLE001
        print(f"[sft] wandb disabled: {exc}", flush=True)
        wb = None

    def log(d: dict, step: int) -> None:
        if wb is not None:
            wb.log(d, step=step)

    base_hold = _holdout_loss(model, holdout, device)
    print(f"[sft] step 0 | holdout sft loss {base_hold:.4f} (adapters at zero: "
          f"this is the base)", flush=True)
    log({"sft/holdout_loss": base_hold}, 0)
    fw = run_eval(model, tokenizer_name, 4096, 6, 20, device,
                  model_config.real_vocab_size)
    print(f"[sft] step 0 | fineweb held-out {fw['eval/loss']:.4f}", flush=True)
    log({"sft/fineweb_loss": fw["eval/loss"]}, 0)
    model.train()
    _freeze_base_routing(model)

    it = iter(train_loader)
    t0 = time.time()
    tokens_trained = 0
    for step in range(1, cfg.total_steps + 1):
        lr = _lr_at(step - 1, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc_loss, acc_tok = 0.0, 0
        for _ in range(cfg.grad_accum_steps):
            input_ids, labels = next(it)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids, labels=labels)
                loss = out.loss / cfg.grad_accum_steps
            loss.backward()
            k = int((labels != IGNORE_INDEX).sum().item())
            acc_loss += float(out.loss) * k
            acc_tok += k
        gn = torch.nn.utils.clip_grad_norm_(hra_params, cfg.grad_clip)
        opt.step()
        tokens_trained += cfg.tokens_per_step()
        if step % cfg.log_interval == 0 or step == 1:
            el = time.time() - t0
            mean_loss = acc_loss / max(acc_tok, 1)
            print(f"[sft] step {step}/{cfg.total_steps} | loss {mean_loss:.4f} "
                  f"| sup tok/step {acc_tok} | gn {float(gn):.2f} | lr {lr:.2e} | "
                  f"{tokens_trained / el / 1e3:.1f}K tok/s", flush=True)
            log({"sft/loss": mean_loss, "sft/supervised_tokens": acc_tok,
                 "sft/grad_norm": float(gn), "sft/lr": lr}, step)
        if step % cfg.eval_interval == 0 or step == cfg.total_steps:
            hl = _holdout_loss(model, holdout, device)
            fw = run_eval(model, tokenizer_name, 4096, 6, 20, device,
                          model_config.real_vocab_size)
            model.train()
            _freeze_base_routing(model)
            print(f"[sft] step {step} | holdout sft loss {hl:.4f} | fineweb "
                  f"{fw['eval/loss']:.4f}", flush=True)
            log({"sft/holdout_loss": hl, "sft/fineweb_loss": fw["eval/loss"]}, step)
        if step % cfg.ckpt_interval == 0 or step == cfg.total_steps:
            path = os.path.join(out_dir, f"sft_adapters_step_{step}.pt")
            adapters = {n: p.detach().cpu() for n, p in model.named_parameters()
                        if p.requires_grad}
            torch.save({"step": step, "adapters": adapters,
                        "sft_config": cfg.__dict__, "base": base_ckpt_path}, path)
            if vol is not None:
                vol.commit()
            print(f"[sft] saved {path}", flush=True)

    n_merged = merge_hra(model)
    merged_path = os.path.join(out_dir, f"sft_merged_step_{cfg.total_steps}.pt")
    torch.save({"step": cfg.total_steps, "model_state_dict": model.state_dict(),
                "sft_config": cfg.__dict__, "base": base_ckpt_path,
                "merged_layers": n_merged}, merged_path)
    if vol is not None:
        vol.commit()
    print(f"[sft] merged {n_merged} layers -> {merged_path}", flush=True)
    if wb is not None:
        wb.finish()
    return {"merged": merged_path, "steps": cfg.total_steps,
            "tokens": tokens_trained, "holdout_loss_base": base_hold}
