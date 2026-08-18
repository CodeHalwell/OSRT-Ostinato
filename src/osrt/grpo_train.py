"""Venue-agnostic GRPO training loop with BATCHED rollouts and log-probs.

Why this module exists
----------------------
The original loop in `app.py::grpo()` is correct but generates and scores one
sequence at a time:

    prompt_batch = prompt_tensor.expand(cfg.group_size, -1)   # batch 16
    ...
    out = model(comp_ids.unsqueeze(0))                        # batch 1, x2

Per optimiser step that is `grad_accum_steps` (32) generate calls at batch 16,
then ~512 policy forwards and ~512 reference forwards at batch 1. Measured on
an H100: **~5.5 minutes per step**, so 900 steps would be ~75 hours.

That is not a small inefficiency, it is the dominant cost, and it is the same
lesson this model has taught three times now: decode here is LAUNCH-BOUND
(18 effective layers of MoE plus 20 sequential Sinkhorn iterations means
thousands of kernel launches per token, a cost that is fixed regardless of
batch size). Measured throughput by batch:

    batch  16-64 : ~620-690 tok/s
    batch    128 : 3,747 tok/s
    batch    256 : 7,504 tok/s
    batch   1024 : 35,631 tok/s

So the fix is not micro-optimisation, it is simply *stop feeding it small
batches*. This module:

  1. Generates ALL `num_prompts x group_size` rollouts in ONE generate() call.
  2. Runs the policy and reference log-prob passes in chunks of many
     sequences rather than one at a time.

Correctness notes
-----------------
* Generation uses LEFT padding (all completions then start at the same index),
  which is what batched sampling requires when prompts differ in length.
* The log-prob passes use RIGHT padding. With causal attention a real token at
  position i only attends to positions <= i, so it never sees the pad tail —
  the logits for real tokens are identical to an unpadded forward. Padded
  positions produce garbage, which we simply never index.
* Advantages are computed PER PROMPT GROUP (`compute_group_advantages` over the
  group's rewards), then attached to each rollout. A prompt whose rollouts all
  succeed or all fail yields zero advantage everywhere and contributes nothing
  — that is expected, and is why prompt difficulty matters so much.
* The loss matches the original formulation exactly: a direct policy gradient
  weighted by the group-normalised advantage, plus Schulman's non-negative KL
  approximation `exp(log_ratio) - log_ratio - 1`. Only the batching differs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from osrt.rewards import compute_group_advantages, compute_reward
from osrt.system_prompts import FEW_SHOT_EXEMPLARS


@dataclass
class Rollout:
    """One sampled completion and everything needed to train on it."""

    ids: Tensor          # full sequence: prompt + completion (1-D, on device)
    prompt_len: int
    advantage: float
    reward: float
    correct: bool
    text: str            # decoded completion only (for printing / rewards)
    gold: str = ""       # ground-truth answer, for offline re-scoring
    breakdown: dict | None = None   # compute_reward's per-term breakdown


def _left_pad(seqs: list[list[int]], pad_id: int, device) -> tuple[Tensor, Tensor, int]:
    """Left-pad for GENERATION so every completion starts at the same index."""
    width = max(len(s) for s in seqs)
    ids = torch.tensor(
        [[pad_id] * (width - len(s)) + s for s in seqs],
        dtype=torch.long, device=device,
    )
    attn = torch.tensor(
        [[0] * (width - len(s)) + [1] * len(s) for s in seqs],
        dtype=torch.long, device=device,
    )
    return ids, attn, width


def _seq_logprobs(
    model: nn.Module,
    batch_ids: Tensor,          # (B, L) right-padded
    prompt_lens: list[int],
    seq_lens: list[int],
    real_vocab_size: int,
    grad: bool,
    temperature: float = 1.0,
) -> list[Tensor]:
    """Per-sequence token log-probs over the COMPLETION span only.

    Right padding is safe here: causal attention means a real token never
    attends to the pad tail, so its logits match an unpadded forward.

    TEMPERATURE MUST MATCH SAMPLING. Rollouts are drawn from pi(a)^(1/T) with
    T=0.4, so the score function the policy gradient needs is grad log pi_T,
    not grad log pi. Computing log-probs at T=1 while sampling at T=0.4 makes
    the gradient a biased, roughly 1/T-attenuated estimate of the wrong
    distribution's score. TRL does exactly this division in its GRPO log-prob
    path (trl/trainer/grpo_trainer.py: "Divide logits by sampling temperature",
    logits = logits / self.temperature) and it is applied to the policy AND the
    reference, so the KL compares like with like.

    Omitting it was measured to matter here: with T=0.4 the policy term was
    ~2.5x under-scaled, stacking with peak_lr 1.5e-6 (base weights moved 0.058%
    in 180 steps) and kl_coeff 0.15 (the KL term averaged 40% of the policy
    term's magnitude, opposing it) to leave GRPO with no detectable effect on
    held-out accuracy across 280 steps.
    """
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(batch_ids).logits[:, :, :real_vocab_size].float()
    if temperature != 1.0:
        logits = logits / temperature
    out: list[Tensor] = []
    for i, (p_len, s_len) in enumerate(zip(prompt_lens, seq_lens)):
        # predict token t from position t-1
        shift_logits = logits[i, p_len - 1:s_len - 1]
        shift_labels = batch_ids[i, p_len:s_len]
        lp = F.log_softmax(shift_logits, dim=-1).gather(
            1, shift_labels.unsqueeze(1)
        ).squeeze(1)
        out.append(lp)
    return out


def generate_rollouts(
    model: nn.Module,
    tok: Any,
    prompts: list[tuple[str, str]],      # (prompt_text, gold_answer)
    cfg: Any,
    device,
    stop_token_ids: list[int] | None = None,
) -> list[list[Rollout]]:
    """Sample `cfg.group_size` rollouts for EVERY prompt in ONE generate call.

    Returns one list of Rollouts per prompt (the group), rewards and advantages
    filled in.
    """
    g = cfg.group_size
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # Exemplar for the regurgitation penalty — empty unless the configured
    # persona actually carries a few-shot demonstration, so unexemplified
    # personas are unaffected.
    #
    # SINGLE-PERSONA ONLY. This is a GLOBAL lookup from cfg.system_persona, so
    # it assumes every prompt in the batch used the same persona. A mixed
    # 0-shot/1-shot batch CANNOT be scored correctly here: unexemplified
    # rollouts would be charged against an exemplar they never saw, and the
    # penalty's measured 0/256 false-positive rate does not cover that case.
    # Mixed-stratum training requires the persona/exemplar to travel WITH each
    # prompt; until it does, cfg.system_persona must name one persona for the
    # whole run.
    exemplar = FEW_SHOT_EXEMPLARS.get(getattr(cfg, "system_persona", "") or "", "")

    # Every prompt repeated group_size times -> a single big batch.
    enc = [tok.encode(p, add_special_tokens=False) for p, _ in prompts]
    flat = [e for e in enc for _ in range(g)]
    ids, attn, width = _left_pad(flat, pad_id, device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            ids, attention_mask=attn,
            max_new_tokens=cfg.max_gen_len,
            temperature=cfg.temperature,
            top_p=getattr(cfg, "top_p", 1.0),
            eos_token_id=tok.eos_token_id,
            stop_token_ids=stop_token_ids,
        )

    groups: list[list[Rollout]] = []
    for pi, (_, gold) in enumerate(prompts):
        p_len = len(enc[pi])
        rollouts: list[Rollout] = []
        rewards: list[float] = []
        for j in range(g):
            row = out[pi * g + j]
            comp_ids = row[width:]
            # drop right padding introduced by shorter siblings finishing early
            if tok.eos_token_id is not None:
                nz = (comp_ids == tok.eos_token_id).nonzero()
                if len(nz):
                    comp_ids = comp_ids[: int(nz[0]) + 1]
            text = tok.decode(comp_ids, skip_special_tokens=False)
            reward, breakdown = compute_reward(
                text, gold,
                correctness_weight=cfg.correctness_reward,
                format_weight=cfg.format_reward,
                length_penalty=cfg.length_penalty,
                think_open=cfg.think_open, think_close=cfg.think_close,
                answer_open=cfg.answer_open, answer_close=cfg.answer_close,
                max_tokens=cfg.max_gen_len,
                completion_tokens=len(comp_ids),
                reasoning_bonus=cfg.reasoning_bonus,
                truncation_penalty=cfg.truncation_penalty,
                empty_think_penalty=cfg.empty_think_penalty,
                few_shot_exemplar=exemplar,
                few_shot_echo_penalty=getattr(
                    cfg, "few_shot_echo_penalty", -3.0),
            )
            rewards.append(reward)
            # full sequence = the UNPADDED prompt + this completion
            full = torch.cat([
                torch.tensor(enc[pi], dtype=torch.long, device=device),
                comp_ids.to(device),
            ])
            rollouts.append(Rollout(
                ids=full[: cfg.seq_len], prompt_len=p_len, advantage=0.0,
                reward=reward, correct=bool(breakdown.get("correct")),
                text=text, gold=str(gold), breakdown=dict(breakdown),
            ))
        advs = compute_group_advantages(rewards)
        for r, a in zip(rollouts, advs):
            r.advantage = float(a)
        groups.append(rollouts)
    return groups


def ema_init(model: nn.Module) -> dict[str, Tensor]:
    """FP32 shadow of the FULL persistent state_dict, for a passive weight EMA.

    Averages EVERY entry, not just parameters. OSRT carries 8 persistent
    float32 buffers that are mutated during training and read at eval time —
    `router_balance_bias` (6 loops x 8 experts, per block, the aux-loss-free
    load balancer) and `gumbel_tau` — plus the constant rope tables. Averaging
    parameters while pairing them with the LATEST router bias would produce a
    hybrid model, not an averaged policy. The model has zero non-floating
    state_dict entries, so covering every floating tensor covers all 229 keys
    and the shadow stays a complete, strictly-loadable state_dict.

    FP32 deliberately: at decay 0.99 each update contributes 1% of a weight,
    which bf16 (8 mantissa bits) cannot accumulate reliably.
    """
    return {k: v.detach().clone().float() for k, v in model.state_dict().items()}


def ema_update(ema: dict[str, Tensor], model: nn.Module, decay: float) -> None:
    """In-place `ema = decay*ema + (1-decay)*live`. Does NOT touch the model.

    Call AFTER a successful optimizer.step(). Purely an observer: it reads the
    live state_dict and writes only into `ema`, so the theta trajectory — and
    therefore the interpretability of the run — is unchanged.
    """
    with torch.no_grad():
        live = model.state_dict()
        missing = set(ema) - set(live)
        if missing:
            raise KeyError(f"EMA shadow has keys absent from the model: "
                           f"{sorted(missing)[:3]}")
        for k, e in ema.items():
            e.mul_(decay).add_(live[k].detach().float(), alpha=1.0 - decay)


def ema_weight_of_init(decay: float, updates: int) -> float:
    """Residual weight on the INITIAL weights after `updates` steps.

    Read this before crediting an early EMA win: at decay 0.99 and 50 updates
    the shadow is still 0.99^50 = 61% the starting checkpoint. When training
    starts from a stronger base (the SFT-v4 soup at 20.0% acc_on), an early EMA
    advantage may only mean "stayed closer to the base", not "averaging helped".
    Compare against the base AND the post-hoc soup, not just against theta.
    """
    return decay ** max(updates, 0)


def dump_rollouts(
    path: str,
    groups: list[list[Rollout]],
    *,
    ckpt: str,
    step: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> int:
    """Append one step's rollouts to a JSONL dump. DEFAULT-OFF in training.

    Stores TOKEN IDS, not just text: `text` round-trips through the tokenizer
    only if encode(decode(ids)) == ids, which is not guaranteed for a byte-level
    BPE with special tokens in the stream. Replaying from ids makes the offline
    A/B consume exactly the sequence the policy produced.

    One line per rollout, with `group` identifying siblings so advantages can be
    recomputed per group offline.
    """
    import json

    n = 0
    with open(path, "a") as f:
        for gi, group in enumerate(groups):
            for r in group:
                ids = r.ids.tolist()
                f.write(json.dumps({
                    "ckpt": ckpt, "step": step, "seed": seed,
                    "temperature": temperature, "top_p": top_p,
                    "group": gi,
                    "prompt_ids": ids[: r.prompt_len],
                    "completion_ids": ids[r.prompt_len:],
                    "prompt_len": r.prompt_len,
                    "gold": r.gold,
                    "text": r.text,
                    "reward": r.reward,
                    "advantage": r.advantage,
                    "correct": bool(r.correct),
                    "breakdown": r.breakdown or {},
                }, ensure_ascii=False) + "\n")
                n += 1
    return n


def load_rollout_dump(path: str, device) -> tuple[list[list[Rollout]], dict]:
    """Rebuild grouped Rollouts from a dump. Inverse of dump_rollouts."""
    import json
    from collections import OrderedDict

    buckets: OrderedDict[tuple[int, int], list[Rollout]] = OrderedDict()
    meta: dict = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            meta = {k: d[k] for k in
                    ("ckpt", "step", "seed", "temperature", "top_p") if k in d}
            ids = torch.tensor(d["prompt_ids"] + d["completion_ids"],
                               dtype=torch.long, device=device)
            buckets.setdefault((d["step"], d["group"]), []).append(Rollout(
                ids=ids, prompt_len=int(d["prompt_len"]),
                advantage=float(d["advantage"]), reward=float(d["reward"]),
                correct=bool(d["correct"]), text=d.get("text", ""),
                gold=str(d.get("gold", "")), breakdown=d.get("breakdown") or {},
            ))
    return list(buckets.values()), meta


def train_on_rollouts(
    model: nn.Module,
    ref_model: nn.Module,
    rollouts: list[Rollout],
    cfg: Any,
    real_vocab_size: int,
    device,
    pad_id: int,
    micro_batch: int = 8,
) -> tuple[float, float]:
    """Batched policy-gradient + KL update over all rollouts of one step.

    Returns (summed loss value, mean approx_kl). Backward is called per
    micro-batch; the caller owns optimizer.step().
    """
    # KL on EVERY usable rollout; policy loss only where the advantage is
    # non-zero. Previously zero-advantage rollouts were dropped before BOTH
    # terms, so they were unanchored — TRL keeps them in the batch precisely so
    # beta*KL still applies. The distinction grows if a correctness clamp is
    # added, since that zeroes many more advantages.
    usable = [r for r in rollouts if len(r.ids) - r.prompt_len > 0]
    if not usable:
        return 0.0, 0.0

    total_loss = 0.0
    total_kl = 0.0
    n = len(usable)
    # Longest-first keeps padding waste down within each micro-batch.
    live = sorted(usable, key=lambda r: len(r.ids), reverse=True)

    for i in range(0, n, micro_batch):
        chunk = live[i:i + micro_batch]
        max_len = max(len(r.ids) for r in chunk)
        batch = torch.full((len(chunk), max_len), pad_id,
                           dtype=torch.long, device=device)
        for k, r in enumerate(chunk):
            batch[k, : len(r.ids)] = r.ids
        p_lens = [r.prompt_len for r in chunk]
        s_lens = [len(r.ids) for r in chunk]

        # Same temperature for BOTH, or the KL compares two different
        # distributions and the penalty becomes meaningless.
        temp = getattr(cfg, "temperature", 1.0) or 1.0
        pol = _seq_logprobs(model, batch, p_lens, s_lens, real_vocab_size, True,
                            temperature=temp)
        ref = _seq_logprobs(ref_model, batch, p_lens, s_lens, real_vocab_size, False,
                            temperature=temp)

        loss = torch.zeros((), device=device)
        for lp, rlp, r in zip(pol, ref, chunk):
            log_ratio = rlp.detach() - lp
            approx_kl = (torch.exp(log_ratio) - log_ratio - 1).mean()
            loss = loss + cfg.kl_coeff * approx_kl      # anchors EVERY rollout
            total_kl += float(approx_kl.detach())
            if abs(r.advantage) > 1e-8:                 # policy term only here
                adv = torch.tensor(r.advantage, device=device,
                                   dtype=torch.float32)
                loss = loss + -(lp * adv).mean()
        loss = loss / n          # mean over ALL live rollouts in the step
        loss.backward()
        total_loss += float(loss.detach())

    return total_loss, total_kl / max(n, 1)


def lr_at_step(step: int, cfg: Any) -> float:
    """Warmup then cosine, honouring lr_anchor_step for re-warmed extensions."""
    anchor = getattr(cfg, "lr_anchor_step", 0)
    eff = max(step - anchor, 0)
    total = max(cfg.total_steps - anchor, 1)
    if eff < cfg.warmup_steps:
        return cfg.peak_lr * eff / max(cfg.warmup_steps, 1)
    prog = (eff - cfg.warmup_steps) / max(total - cfg.warmup_steps, 1)
    return cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
        1 + math.cos(math.pi * min(prog, 1.0))
    )
