"""Feature ablation on the copy task — isolate which architectural feature
breaks learning / explodes gradients in the assembled stack.

Runs the same short training loop (grad-clipped, warmup) over a baseline plus
each feature toggled on in isolation, and reports task-CE drop + the worst
pre-clip gradient norm. The culprit shows up as a flat CE and/or a huge gnorm.

    PYTHONPATH=src python scripts/ablate_features.py
"""

from __future__ import annotations

import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM
from osrt.muon import HybridMuonAdamW, Muon, build_param_groups

VOCAB, SEQ, BATCH, STEPS, WARMUP = 128, 48, 24, 120, 20
LR_MUON, LR_ADAMW = 2e-3, 1e-3
_A, _C = 5, 1  # fixed token recurrence next = (A*cur + C) % VOCAB


def make_batch() -> torch.Tensor:
    """Deterministic token recurrence: next = (A*cur + C) % VOCAB. The map
    cur -> next is a fixed function the model can fit from step 1, so loss
    falls smoothly toward 0 — a clean 'does the stack train?' signal (vs the
    induction/copy task, which needs a phase-change too slow for a smoke run)."""
    cur = torch.randint(0, VOCAB, (BATCH, 1))
    cols = [cur]
    for _ in range(SEQ - 1):
        cur = (_A * cur + _C) % VOCAB
        cols.append(cur)
    return torch.cat(cols, dim=1)


BASE = dict(
    dim=256, heads=8, head_dim=32, num_kv_heads=2,
    vocab_size=VOCAB, real_vocab_size=VOCAB, num_blocks=2, recursive_loops=4,
    num_routed_experts=8, top_k_experts=2, expert_hidden=128,
    shared_expert_hidden=128, max_position_embeddings=SEQ,
    aux_loop_loss_weight=0.05, router_balance_bias_enabled=True,
    router_aux_loss_coeff=0.10, router_z_loss_coeff=1e-3,
)

VARIANTS = {
    "baseline (all off)": {},
    "+mHC": dict(use_mhc=True, n_hc=4),
    "+sqrt_softplus": dict(router_affinity="sqrt_softplus"),
    "+attn_sink": dict(attention_sink=True),
    "+swiglu_clamp": dict(swiglu_clamp=10.0),
}


def run(name: str, over: dict) -> None:
    torch.manual_seed(0)
    cfg = OSRTConfig(**{**BASE, **over})
    model = OSRTForCausalLM(cfg)
    model.train()
    mp, ag = build_param_groups(model.named_parameters(), 0.01)
    muon = Muon(mp, lr=LR_MUON)
    adamw = torch.optim.AdamW(ag, lr=LR_ADAMW, betas=(0.9, 0.95))
    opt = HybridMuonAdamW(muon, adamw)

    first = last = None
    max_gnorm = 0.0
    nan_at = None
    for step in range(STEPS):
        scale = min(1.0, (step + 1) / WARMUP)
        for g in muon.param_groups:
            g["lr"] = LR_MUON * scale
        for g in adamw.param_groups:
            g["lr"] = LR_ADAMW * scale
        ids = make_batch()
        opt.zero_grad()
        out = model(ids, labels=ids)
        if not torch.isfinite(out.loss):
            nan_at = step
            break
        out.loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        max_gnorm = max(max_gnorm, gnorm)
        opt.step()
        task = model.last_task_loss.item()
        if step == 0:
            first = task
        last = task
    drop = "n/a" if first is None or last is None else f"{100*(first-last)/first:5.1f}%"
    nan = f" NaN@{nan_at}" if nan_at is not None else ""
    print(f"  {name:22s} taskCE {first:.2f}->{last:.2f}  drop {drop}  "
          f"max_gnorm {max_gnorm:10.1f}{nan}")


def main() -> None:
    print(f"copy task, {STEPS} steps each, grad-clip 1.0, Muon 5e-3 / AdamW 1.5e-3")
    print(f"(uniform task CE = {torch.log(torch.tensor(float(VOCAB-4))):.2f}; "
          f"learning => taskCE falls, healthy => max_gnorm is O(1-100))\n")
    for name, over in VARIANTS.items():
        run(name, over)


if __name__ == "__main__":
    main()
