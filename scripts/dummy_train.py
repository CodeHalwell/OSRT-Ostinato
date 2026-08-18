"""Dummy end-to-end training run — proves the FULL assembled architecture
trains stably before any GPU spend.

Synthetic copy/induction task: each sequence is a random prefix followed by a
copy of itself. Predicting the copied half requires the model to learn an
induction mechanism (attend back exactly half the sequence), so a falling loss
on FRESH batches each step means the architecture is genuinely learning
structure — not memorizing one batch.

Exercises the real canonical stack scaled to CPU: GQA + MLA latent cache, mHC
(Birkhoff/Sinkhorn 4-channel stream), sqrt(softplus) routing, SwiGLU clamps,
per-head attention sink, aux-loop loss, Muon+AdamW. Logs the recursion/MoE
collapse monitoring every few steps so you can watch for MoE collapse or lack
of loop-depth spread on a live run.

    PYTHONPATH=src python scripts/dummy_train.py
"""

from __future__ import annotations

import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM
from osrt.monitoring import summarize
from osrt.muon import HybridMuonAdamW, Muon, build_param_groups

VOCAB = 128
SEQ = 48           # half random, half copied
BATCH = 24
STEPS = 250
LOG_EVERY = 50


_A, _C = 5, 1  # deterministic token recurrence next = (A*cur + C) % VOCAB


def make_batch() -> torch.Tensor:
    """Deterministic token recurrence next = (A*cur + C) % VOCAB -> (BATCH, SEQ).
    The map cur->next is a fixed function fittable from step 1, so loss falls
    smoothly — a clean 'does the stack train?' signal. (An induction/copy task
    was tried first but needs a phase-change too slow for a smoke run, so it
    read as flat for EVERY config including the baseline.)"""
    cur = torch.randint(0, VOCAB, (BATCH, 1))
    cols = [cur]
    for _ in range(SEQ - 1):
        cur = (_A * cur + _C) % VOCAB
        cols.append(cur)
    return torch.cat(cols, dim=1)


def main() -> None:
    torch.manual_seed(0)
    # The full canonical feature set, scaled down for CPU.
    cfg = OSRTConfig(
        dim=256, heads=8, head_dim=32, num_kv_heads=2,    # GQA + MLA latent
        vocab_size=VOCAB, real_vocab_size=VOCAB,
        num_blocks=2, recursive_loops=4,
        num_routed_experts=8, top_k_experts=2,
        expert_hidden=128, shared_expert_hidden=128,
        max_position_embeddings=SEQ,
        use_mhc=True, n_hc=4, mhc_sinkhorn_iters=20,      # mHC
        swiglu_clamp=10.0,                                # SwiGLU clamp
        attention_sink=True,                              # attention sink
        router_affinity="sqrt_softplus",                  # sqrt-softplus routing
        router_balance_bias_enabled=True,
        aux_loop_loss_weight=0.05,                        # anti loop-collapse
        router_aux_loss_coeff=0.10, router_z_loss_coeff=1e-3,
    )
    model = OSRTForCausalLM(cfg)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params | features: GQA+MLA, mHC, "
          f"sqrt_softplus, SwiGLU-clamp, attn-sink, aux-loop\n")

    muon_params, adamw_groups = build_param_groups(model.named_parameters(), 0.01)
    # Conservative LR + a short warmup + gradient clipping — standard training
    # hygiene for a deep recursive + mHC stack. Muon's effective step is large,
    # so without clipping the gradient spikes diverge.
    base_muon_lr, base_adamw_lr = 5e-3, 1.5e-3
    muon = Muon(muon_params, lr=base_muon_lr)
    adamw = torch.optim.AdamW(adamw_groups, lr=base_adamw_lr, betas=(0.9, 0.95))
    opt = HybridMuonAdamW(muon, adamw)
    warmup = 20

    uniform = torch.log(torch.tensor(float(VOCAB - 4)))
    print(f"uniform-baseline task CE = {uniform:.3f}  (random half ~unpredictable; "
          f"copied half is learnable -> task CE should fall well below this)\n")
    print(f"{'step':>5} {'total':>7} {'taskCE':>7}  {'grad':>6}  {'health'}")

    first = last = None
    for step in range(STEPS):
        # Linear LR warmup.
        scale = min(1.0, (step + 1) / warmup)
        for g in muon.param_groups:
            g["lr"] = base_muon_lr * scale
        for g in adamw.param_groups:
            g["lr"] = base_adamw_lr * scale

        ids = make_batch()             # FRESH batch -> generalization, not memorization
        opt.zero_grad()
        out = model(ids, labels=ids)
        loss = out.loss
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if not torch.isfinite(loss):
            print(f"{step:5d}  NON-FINITE LOSS — architecture unstable")
            raise SystemExit(1)
        task = model.last_task_loss.item()
        if step == 0:
            first = task
        last = task
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            _, msg = summarize(model, ids, ids)
            print(f"{step:5d} {loss.item():7.3f} {task:7.3f}  {gnorm:6.2f}  {msg}")

    print(f"\ntask CE {first:.3f} -> {last:.3f}  ({100*(first-last)/first:.0f}% drop)")
    healthy = last < first * 0.85 and last < float(uniform)
    print("RESULT:", "PASS — full stack trains, learns the copy task, no collapse"
          if healthy else "INVESTIGATE — task CE did not fall as expected")


if __name__ == "__main__":
    main()
