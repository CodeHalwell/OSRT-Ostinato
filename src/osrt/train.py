"""Pre-training loop for OSRT (v7 trunk).

One loop: WSD (or cosine) LR applied per param group, phase transitions
(seq_len + dataset swap + micro-batch shape), held-out eval at a fixed
context, atomic checkpoints with fail-closed resume, W&B, torch.compile,
the 23h Modal rescue, annealed Gumbel top-k exploration and the per-expert
balance-bias update after every optimizer step.

Telemetry (all logged to W&B and stdout on logging steps):
  - per_token_entropy — the real router-sharpness signal
  - marginal_entropy — balance proxy (stays high if globally balanced)
  - assignment_entropy — hard f entropy
  - raw_max_prob — pre-renormalisation top-1 confidence
  - top_margin — gap between rank 0 and rank 1 probs
  - drop_rate — capacity drops (identically 0 on the dropless grouped path)
  - loop/update_norm_l*, loop/hidden_norm_ratio — recursion health
  - muon/ortho_err, muon/update_rms_* — optimizer health
  - train/grad_norm, train/nonfinite_steps, train/dead_sources

Health checks: the router-sharpening gate runs once at
`early_stop_check_step`; loop collapse, residual explosion and (after the
gate) the whole router-health set are re-checked on every logging step and
stop the run once they fail `health_check_patience` checks in a row. Failed
runs write `osrt_failed_step_N.pt`, which the resume scan ignores.

`run_training` returns a status string: "complete", "already_complete",
"early_stop", "rescued" (23h boundary, re-invoke to continue) or
"data_dead" (every data source failed permanently; rescue written).
"""

import glob
import hashlib
import json
import math
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
from torch import Tensor

try:
    import wandb
except ImportError:
    wandb = None

from osrt.config import OSRTConfig
from osrt.data import DataSourceDead, make_loader
from osrt.model import OSRTForCausalLM
from osrt.train_config import PretrainConfig


def get_lr(step: int, cfg: PretrainConfig) -> float:
    """Learning rate at `step`, returning the AdamW/Lion peak_lr scale.

    Two schedules, selected by `cfg.lr_schedule`:

    * "wsd" (default, v7) — linear warmup, a flat stable phase, then a linear
      decay to min_lr over the last `wsd_decay_frac` of the run. The stable
      phase is the point: a drip-funded run can stop and resume anywhere in it
      without re-warming or reshaping the curve, and only the branch that
      produces a release checkpoint pays the decay.
    * "cosine" — v6's schedule, kept so historical runs stay reproducible.
    """
    if step < cfg.warmup_steps:
        return cfg.peak_lr * step / cfg.warmup_steps

    if getattr(cfg, "lr_schedule", "cosine") == "wsd":
        decay_steps = max(int(cfg.total_steps * cfg.wsd_decay_frac), 1)
        decay_start = max(cfg.total_steps - decay_steps, cfg.warmup_steps)
        if step < decay_start:
            return cfg.peak_lr                       # stable phase
        frac = min((step - decay_start) / decay_steps, 1.0)
        return cfg.peak_lr + (cfg.min_lr - cfg.peak_lr) * frac

    progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
    return cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
        1 + math.cos(math.pi * progress)
    )


def _set_param_group_lrs(
    optimizer, step: int, cfg: PretrainConfig,
) -> float:
    """Apply `get_lr`'s schedule to every param group, respecting the
    per-group `_peak_lr` / `_min_lr` tags (see `_stamp_schedule_tags`).

    Returns the AdamW/Lion-scale LR for logging. Muon groups carry their own
    peak/floor and follow the SAME schedule shape — warmup from 0, then the
    position between floor and peak that `get_lr` gives the AdamW scale — so
    a WSD run is flat for every optimizer through the stable phase and every
    optimizer reaches its floor together at the end.

    History: until 2026-09-30 this function computed its own cosine and never
    consulted `cfg.lr_schedule`, so `get_lr`'s WSD branch was tested but never
    applied — the trunk decayed from step 400. Routing through `get_lr` keeps
    the two from drifting apart again; tests pin them equal mid-stable-phase.
    """
    base = get_lr(step, cfg)
    if step < cfg.warmup_steps:
        ratio = base / cfg.peak_lr if cfg.peak_lr > 0 else 0.0
        for pg in optimizer.param_groups:
            pg["lr"] = pg.get("_peak_lr", cfg.peak_lr) * ratio
        return base

    span = cfg.peak_lr - cfg.min_lr
    frac = (base - cfg.min_lr) / span if span > 0 else 1.0   # 1 = peak, 0 = floor
    for pg in optimizer.param_groups:
        peak = pg.get("_peak_lr", cfg.peak_lr)
        floor = pg.get("_min_lr", cfg.min_lr)
        pg["lr"] = floor + (peak - floor) * frac
    return base


def _stamp_schedule_tags(optimizer, train_cfg: PretrainConfig) -> None:
    """(Re)write the per-group `_peak_lr` / `_min_lr` tags the schedule reads,
    from the CURRENT config.

    Called at construction and again right after `optimizer.load_state_dict`:
    torch restores every param-group key except `params` from the checkpoint,
    so without the re-stamp a resumed session silently kept the previous
    session's Muon/AdamW peak and floor (verified 2026-09-29: a config change
    from 0.02 to 0.015 came back as 0.02 after load). The strict recipe check
    fails closed on such a change anyway; this makes the applied LR match the
    config that passed the check.
    """
    muon = getattr(optimizer, "muon", None)
    if muon is not None:
        muon_lr = getattr(train_cfg, "muon_lr", train_cfg.peak_lr)
        muon_min = getattr(train_cfg, "muon_min_lr", muon_lr * 0.1)
        for pg in muon.param_groups:
            pg["_peak_lr"], pg["_min_lr"] = muon_lr, muon_min
        groups = optimizer.adamw.param_groups
    else:
        groups = optimizer.param_groups
    for pg in groups:
        pg["_peak_lr"], pg["_min_lr"] = train_cfg.peak_lr, train_cfg.min_lr


def get_router_gumbel_tau(step: int, cfg: PretrainConfig) -> float:
    """Linear Gumbel top-k noise schedule for early router exploration."""
    init = cfg.router_gumbel_tau_init
    final = cfg.router_gumbel_tau_final
    anneal = max(cfg.router_gumbel_anneal_steps, 1)
    progress = min(step / anneal, 1.0)
    return init + (final - init) * progress


def set_router_gumbel_tau(model: nn.Module, tau: float) -> None:
    """Set the per-MoE Gumbel tau buffer on compiled or eager v5 models."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    base = inner.model if hasattr(inner, "model") else inner
    for block in base.blocks:
        block.moe.gumbel_tau.fill_(tau)


def apply_router_balance_updates(model: nn.Module) -> None:
    """Apply once-per-step balance-bias updates on compiled or eager models."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    base = inner.model if hasattr(inner, "model") else inner
    for block in base.blocks:
        block.moe.apply_balance_update()


def get_phase(step: int, cfg: PretrainConfig) -> tuple[str, dict]:
    """Get current phase config for a given step."""
    for name, p in cfg.phases.items():
        if p["start"] <= step < p["end"]:
            return name, p
    last_name = list(cfg.phases.keys())[-1]
    return last_name, cfg.phases[last_name]


# Training semantics that must not silently change across a resume. A v7 trunk
# run is months of drip-funded sessions; if a resumed session quietly uses a
# different schedule, data plan or Muon recipe than the one that produced the
# checkpoint, the loss curve is a splice of two experiments and nothing
# downstream is interpretable. Checked fail-closed on resume.
#
# Deliberately NOT here: the micro-batch shape (`scale_micro_batches` moves a
# run between a 192 GB B200 and a 96 GB RTX PRO 6000 at constant tokens/step —
# the per-phase tokens/step ARE checked, via the phase plan) and operational
# knobs such as dataloader workers or logging intervals.
_STRICT_TRAIN_RECIPE_KEYS = (
    "total_steps", "warmup_steps",
    "lr_schedule", "wsd_decay_frac", "peak_lr", "min_lr",
    "weight_decay", "grad_clip",
    "optimizer_name", "muon_lr", "muon_min_lr", "muon_momentum",
    "per_head_muon", "muon_ns_steps", "muon_ns_stable_steps",
    "muon_update_rms",
)

# Dataset-entry fields that define WHAT is trained on. Order-insensitive keys
# only; a re-ordered but identical list is the same plan.
_PHASE_DATASET_KEYS = (
    "name", "hf_id", "hf_config", "split", "weight", "format", "filter",
    "subsample", "max_tokens", "skip",
)


def _phase_plan_digest(cfg) -> str:
    """Canonical JSON of the phase plan: boundaries, seq_len, tokens/step and
    the dataset entries. Compared as a whole on resume."""
    plan = []
    for name, ph in cfg.phases.items():
        bs = ph.get("batch_size", cfg.batch_size)
        ga = ph.get("grad_accum_steps", cfg.grad_accum_steps)
        plan.append({
            "name": name,
            "start": ph.get("start"),
            "end": ph.get("end"),
            "seq_len": ph["seq_len"],
            "tokens_per_step": bs * ga * ph["seq_len"],
            "datasets": [
                {k: d.get(k) for k in _PHASE_DATASET_KEYS if d.get(k) is not None}
                for d in ph.get("datasets") or []
            ],
        })
    return json.dumps(plan, sort_keys=True, default=str)


def _tokenizer_digest(tokenizer_name: str | None) -> str | None:
    """sha256 of tokenizer.json when `tokenizer_name` is a local directory —
    the vocab and merges a checkpoint's embeddings were trained against.
    A rebuilt tokenizer with the same size and special ids passes the contract
    check; only this catches a changed merge table."""
    if not tokenizer_name:
        return None
    path = os.path.join(tokenizer_name, "tokenizer.json")
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _training_recipe_metadata(cfg, tokenizer_name: str | None = None) -> dict:
    """Serialisable training semantics stamped into every checkpoint."""
    meta = {k: getattr(cfg, k, None) for k in _STRICT_TRAIN_RECIPE_KEYS}
    if getattr(cfg, "phases", None):
        meta["phase_plan"] = _phase_plan_digest(cfg)
    digest = _tokenizer_digest(tokenizer_name)
    if digest is not None:
        meta["tokenizer_sha256"] = digest
    return meta


def _optimizer_lr_tags(opt_state: dict | None) -> dict:
    """Peak-LR tags recorded inside a saved optimizer state. Lets the drift
    check catch a changed Muon/AdamW LR even on checkpoints written before
    `muon_lr` joined the recipe metadata."""
    out: dict = {}
    if not isinstance(opt_state, dict):
        return out
    for key, name in (("muon", "muon_lr"), ("adamw", "peak_lr")):
        groups = (opt_state.get(key) or {}).get("param_groups") or []
        tags = {g.get("_peak_lr") for g in groups if g.get("_peak_lr") is not None}
        if len(tags) == 1:
            out[name] = tags.pop()
    return out


def _model_shape_metadata(cfg) -> dict:
    """Model identity. A checkpoint cannot load into a different vocab or
    expert layout, and v7 changed BOTH against v6 — so record them rather
    than discovering the mismatch as a shape error mid-load. The router gate
    semantics are stamped alongside: not a shape, but weights trained with
    the balance bias inside their gating weights compute a different function
    when loaded with it outside, and a state-dict load cannot tell."""
    return {
        "vocab_size": cfg.vocab_size,
        "real_vocab_size": cfg.real_vocab_size,
        "dim": cfg.dim,
        "num_blocks": cfg.num_blocks,
        "recursive_loops": cfg.recursive_loops,
        "num_routed_experts": cfg.num_routed_experts,
        "top_k_experts": cfg.top_k_experts,
        "expert_hidden": cfg.expert_hidden,
        "router_bias_in_gates": bool(getattr(cfg, "router_bias_in_gates", False)),
    }


def _legacy_gate_semantics_apply(model_config) -> bool:
    """True when a checkpoint written before `router_bias_in_gates` existed
    (2026-09-30) would compute a different function under this config: those
    checkpoints took their gates from the bias-adjusted, Gumbel-noised
    selection distribution (today's `True`), so the difference is real
    whenever the balance bias or Gumbel exploration is in play."""
    if getattr(model_config, "router_bias_in_gates", False):
        return False
    return bool(
        getattr(model_config, "router_balance_bias_enabled", True)
        or getattr(model_config, "router_gumbel_tau_init", 0.0) > 0
    )


def assert_no_resume_drift(
    ckpt: dict,
    *,
    model_config=None,
    train_cfg=None,
    stage: str | None = None,
    path: str = "<checkpoint>",
    tokenizer_name: str | None = None,
) -> None:
    """Fail closed if a checkpoint was produced by a different experiment.

    Silence here is the expensive failure: the run continues, the numbers look
    plausible, and the drift is only discovered when a result cannot be
    reproduced.

    Escape hatch for a DELIBERATE mid-run recipe change (e.g. lowering the
    Muon LR after a documented instability): set OSRT_ALLOW_RECIPE_DRIFT=1 and
    the differences are printed as a warning instead of raising. The new
    values are stamped into the next checkpoint, so the change is on record.
    """
    allow = os.environ.get("OSRT_ALLOW_RECIPE_DRIFT", "") == "1"

    def _fmt(k: str, a, b) -> str:
        if k == "phase_plan":
            return ("    phase_plan: data mix, seq_len, tokens/step or phase "
                    "boundaries differ from the checkpoint")
        if k == "router_bias_in_gates" and a is True and legacy_gates:
            return ("    router_bias_in_gates: checkpoint predates the flag and "
                    "was trained with the balance bias INSIDE its gating "
                    "weights (=True); current=False. Set "
                    "router_bias_in_gates=True to continue that run as trained")
        return f"    {k}: checkpoint={a!r} current={b!r}"

    def _diff(saved: dict | None, current: dict, label: str) -> None:
        if not saved:
            return                       # pre-metadata checkpoint; nothing to check
        bad = {k: (saved.get(k), v) for k, v in current.items()
               if k in saved and saved.get(k) != v}
        if not bad:
            return
        lines = "\n".join(_fmt(k, a, b) for k, (a, b) in bad.items())
        msg = (
            f"{label} drift in {path}:\n{lines}\n"
            f"  Resuming would splice two different experiments. Either point at "
            f"a fresh --ckpt-dir, restore the config that produced this file, or "
            f"set OSRT_ALLOW_RECIPE_DRIFT=1 to continue deliberately."
        )
        if allow:
            print(f"WARNING (OSRT_ALLOW_RECIPE_DRIFT=1): {msg}", flush=True)
            return
        raise RuntimeError(msg)

    legacy_gates = False
    if model_config is not None:
        saved_shape = ckpt.get("model_shape")
        _diff(saved_shape, _model_shape_metadata(model_config), "MODEL SHAPE")
        # A checkpoint from before `router_bias_in_gates` existed carries no
        # such key, so `_diff` cannot see that it was trained under today's
        # `True` while the default is now `False`: the resumed model would
        # weight its selected experts differently mid-run while keeping the
        # optimizer state and the loss curve. Treat the absent key as True.
        if (
            (not saved_shape or "router_bias_in_gates" not in saved_shape)
            and _legacy_gate_semantics_apply(model_config)
        ):
            legacy_gates = True
            _diff({"router_bias_in_gates": True},
                  {"router_bias_in_gates": False}, "ROUTER GATE SEMANTICS")
    if train_cfg is not None:
        _diff(ckpt.get("training_recipe"),
              _training_recipe_metadata(train_cfg, tokenizer_name),
              "TRAINING RECIPE")
        # Checkpoints written before muon_lr/peak_lr joined the recipe keys
        # still carry the LR tags inside the optimizer state.
        _diff(_optimizer_lr_tags(ckpt.get("optimizer_state_dict")),
              {"muon_lr": getattr(train_cfg, "muon_lr", None),
               "peak_lr": train_cfg.peak_lr},
              "OPTIMIZER LR")
    saved_stage = ckpt.get("training_stage")
    if stage is not None and saved_stage is not None and saved_stage != stage:
        raise RuntimeError(
            f"STAGE drift in {path}: checkpoint={saved_stage!r} current={stage!r}."
        )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
    model_config=None,
    train_cfg=None,
    stage: str = "pretrain_v7",
    *,
    data_state: dict | None = None,
    tokenizer_name: str | None = None,
) -> None:
    """Save a training checkpoint.

    Stamps model-shape and training-recipe metadata so a later resume can fail
    closed on drift (`assert_no_resume_drift`) instead of silently splicing two
    experiments together. `data_state` (the streaming loader's position, see
    `osrt.data.TokenStream.state_dict`) rides along so a resumed session
    continues the data stream instead of restarting every source at row 0."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    # Atomic save: serialize to a temp file, then os.replace onto the final
    # name. torch.save writes directly, and a ~4.9GB checkpoint takes several
    # seconds to serialize — so the glob-matched final name would otherwise
    # exist while still truncated, and the HF sync daemon (or a crash mid-write)
    # could capture/resume a partial file. os.replace is atomic, so `path` only
    # ever appears complete. (docs/specs/2026-07-26-ckpt-sync §1)
    tmp_path = f"{path}.tmp"
    torch.save(
        {
            "step": step,
            "model_state_dict": inner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_stage": stage,
            "model_shape": (
                _model_shape_metadata(model_config) if model_config else None),
            "training_recipe": (
                _training_recipe_metadata(train_cfg, tokenizer_name)
                if train_cfg else None),
            "data_state": data_state,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)
    print(f"  -> Checkpoint saved: {path}")


def load_model_state_or_raise(
    model: nn.Module,
    state_dict: dict,
    context: str,
) -> None:
    """Load model weights and fail on any key drift."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    missing, unexpected = inner.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        missing_sample = ", ".join(missing[:8])
        unexpected_sample = ", ".join(unexpected[:8])
        raise RuntimeError(
            f"{context}: checkpoint/model key mismatch. "
            f"missing={len(missing)} [{missing_sample}], "
            f"unexpected={len(unexpected)} [{unexpected_sample}]. "
            "Use an explicit migration or start a fresh checkpoint directory."
        )


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
    *,
    model_config=None,
    train_cfg=None,
    tokenizer_name: str | None = None,
    stage: str | None = None,
) -> tuple[int, dict | None]:
    """Load from checkpoint. Returns (step to resume from, saved data state);
    (0, None) if the path is missing.

    One `torch.load`: the drift check reads the metadata from the same dict
    the weights come from (the old code loaded a multi-GB file twice, once to
    CPU just for two small dicts). Fails closed when the optimizer state does
    not match the configured optimizer — silently "starting fresh" would
    splice two runs — and re-stamps the schedule tags from the current config
    (see `_stamp_schedule_tags`)."""
    if not os.path.exists(path):
        return 0, None
    print(f"Resuming from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    assert_no_resume_drift(
        ckpt, model_config=model_config, train_cfg=train_cfg,
        stage=stage, path=path, tokenizer_name=tokenizer_name,
    )
    load_model_state_or_raise(
        model,
        ckpt["model_state_dict"],
        context=f"pretrain resume from {path}",
    )
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except (KeyError, ValueError, RuntimeError) as e:
        raise RuntimeError(
            f"optimizer state in {path} does not fit the configured optimizer "
            f"({type(e).__name__}: {e}). Resuming with a fresh optimizer would "
            "splice two runs; restore optimizer_name or use a fresh --ckpt-dir."
        ) from e
    if train_cfg is not None:
        _stamp_schedule_tags(optimizer, train_cfg)
    start_step = ckpt["step"] + 1
    data_state = ckpt.get("data_state")
    print(f"  Resumed at step {start_step}"
          f"{' (data position restored)' if data_state else ''}")
    return start_step, data_state


# Eval batches are materialised once per process and replayed on every
# subsequent call, so the held-out set is identical at every eval and the
# stream/tokenise cost is paid once. Keyed on (tokenizer, seq_len, batch,
# steps): the trainer passes the FIXED eval_seq_len/eval_batch_size so the
# set does not change at phase boundaries.
_EVAL_BATCH_CACHE: dict[tuple, list[tuple[Tensor, Tensor]]] = {}


@torch.no_grad()
def run_eval(
    model: nn.Module,
    tokenizer_name: str,
    seq_len: int,
    batch_size: int,
    eval_steps: int,
    device: torch.device,
    real_vocab_size: int | None = None,
) -> dict:
    """Run held-out evaluation on a cached FineWeb-Edu slice.

    FineWeb-Edu has no upstream validation split, so the held-out set is the
    newest dump as its own config (`CC-MAIN-2025-26`, skip 1,000 rows: O(1)
    to open). Training streams the `default` config; its shard order is
    shuffled per session, so a training shard from that dump is possible but
    the overlap is a few shards in thousands — small, not zero.

    The first call materialises `eval_steps` batches and caches them on CPU;
    every subsequent call replays the cache, so the stream/tokenise cost is
    paid once per process and the samples are identical at every eval.
    Callers pass a FIXED seq_len/batch_size (train_config.eval_seq_len /
    eval_batch_size) so the set is also identical across phases.

    Switches model to inference mode (drops off, aux loss excluded).
    `real_vocab_size` is accepted for call-site compatibility and unused: the
    model slices its own logits.
    """
    was_training = model.training
    model.train(False)  # disable capacity drops + dropout-like behaviour

    cache_key = (tokenizer_name, seq_len, batch_size, eval_steps)
    cached = _EVAL_BATCH_CACHE.get(cache_key)
    if cached is None:
        loader = make_loader(
            dataset_configs=[
                {
                    # Held-out = the NEWEST dump as its own config. The
                    # previous `skip: 100_000_000` on `default` iterated 100M
                    # rows through the HF stream: the 2026-09-02 trunk sat
                    # >1 h at step 1000 with no checkpoint (the save came
                    # after the eval). O(1) now.
                    "name": "fineweb-edu-eval",
                    "hf_id": "HuggingFaceFW/fineweb-edu",
                    "hf_config": "CC-MAIN-2025-26",
                    "weight": 1.0,
                    "skip": 1_000,
                },
            ],
            seq_len=seq_len,
            tokenizer_name=tokenizer_name,
            batch_size=batch_size,
            step_num=999999,  # fixed seed, never matches training seeds
            # In-process loading (no worker subprocesses). Eval fetches
            # eval_steps batches exactly ONCE and caches them, so workers buy
            # nothing — and spawning a fresh worker pool on top of the running
            # training workers crashed mid-run on a semaphore/shared-memory
            # failure (DataLoader worker SemLock._rebuild -> FileNotFoundError,
            # "leaked semaphore objects"). num_workers=0 removes that surface.
            num_workers=0,
        )
        data_iter = iter(loader)
        cached = []
        t_mat = time.time()
        for _ in range(eval_steps):
            try:
                cached.append(next(data_iter))
            except StopIteration:
                break
        print(
            f"  [eval] materialised {len(cached)} held-out batches in "
            f"{time.time() - t_mat:.0f}s (cached for the rest of the process)",
            flush=True,
        )
        _EVAL_BATCH_CACHE[cache_key] = cached
        # Drop the iterator before the loader so worker processes are
        # reaped cleanly. `del loader, data_iter` evaluates left-to-right
        # and dropping `loader` first does nothing observable because
        # `data_iter` still holds the loader (and therefore the spawn
        # workers); the actual teardown only fires when `data_iter` is
        # released, racing finalize and producing
        # "Fatal Python error: PyGILState_Release" in one of the workers
        # — a leak per eval call. Force order + GC so workers exit cleanly.
        import gc
        del data_iter
        del loader
        gc.collect()

    total_loss = 0.0
    total_tokens = 0
    # no_grad is load-bearing: without it this forward retains activations for
    # a backward that never comes, on top of the live training allocation. It
    # survived here only because midtrain ran with enough headroom; the same
    # omission OOMed the sft_v4 rollout eval outright. Values are unchanged.
    with torch.no_grad():
        for input_ids, labels in cached:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)
            # In inference mode, outputs.loss is pure task CE (no aux pollution).
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    if was_training:
        model.train(True)

    mean_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(mean_loss, 20.0))
    return {
        "eval/loss": mean_loss,
        "eval/perplexity": perplexity,
        "eval/tokens": total_tokens,
    }


def _collect_moe_metrics(model: nn.Module) -> tuple[dict, dict]:
    """Pull v5 MoE telemetry from each block. Returns (wandb_dict, stdout_summary)."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    base = inner.model if hasattr(inner, "model") else inner

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    per_token_ents: list[float] = []
    marginal_ents: list[float] = []
    assign_ents: list[float] = []
    clean_per_token_ents: list[float] = []
    clean_marginal_ents: list[float] = []
    clean_assign_ents: list[float] = []
    raw_max_probs: list[float] = []
    top_margins: list[float] = []
    clean_raw_max_probs: list[float] = []
    clean_top_margins: list[float] = []
    drop_rates: list[float] = []
    moe_gates: list[float] = []
    expert_maxes: list[float] = []
    expert_mins: list[float] = []
    clean_expert_maxes: list[float] = []
    clean_expert_mins: list[float] = []
    prebias_per_token_ents: list[float] = []
    prebias_marginal_ents: list[float] = []
    prebias_assign_ents: list[float] = []
    prebias_raw_max_probs: list[float] = []
    prebias_top_margins: list[float] = []
    prebias_expert_maxes: list[float] = []
    prebias_expert_mins: list[float] = []
    balance_losses: list[float] = []
    bias_abs_maxes: list[float] = []
    bias_ema_maxes: list[float] = []
    bias_ema_mins: list[float] = []

    metrics: dict = {}
    dead_experts_total = 0
    for bi, blk in enumerate(base.blocks):
        # Log the EFFECTIVE gate (post-softplus) — the raw parameter is
        # an unconstrained pre-image and doesn't reflect the actual
        # routed-branch scaling. softplus(raw) is what multiplies h_routed
        # in RecursiveBlock.forward, so that's what matters for
        # interpreting "is the routed branch contributing".
        mg = blk.effective_moe_gate().item()
        moe_gates.append(mg)
        metrics[f"moe/moe_gate_b{bi}"] = mg

        if blk.moe.balance_loss is not None:
            balance_losses.append(blk.moe.balance_loss.item())
        if hasattr(blk.moe, "router_balance_bias"):
            bias = blk.moe.router_balance_bias           # (num_loops, num_routed)
            bias_abs_max = bias.abs().max().item()
            bias_abs_maxes.append(bias_abs_max)
            metrics[f"moe/bias_abs_max_b{bi}"] = bias_abs_max
            # §17.3: the balance-bias TRAJECTORY per loop, not just its max.
            # Under Quantile Balancing the bias is a one-shot solve, so a
            # per-loop spread that keeps growing means the router's raw
            # affinities are diverging across loops — the weight-tied
            # analogue of residual explosion, visible before the loss is.
            for li in range(bias.shape[0]):
                row = bias[li]
                metrics[f"moe/b{bi}/loop{li}/bias_std"] = row.std().item()
                metrics[f"moe/b{bi}/loop{li}/bias_range"] = (
                    row.max() - row.min()).item()
            metrics[f"moe/b{bi}/bias_loop_spread"] = (
                bias.std(dim=1).max() - bias.std(dim=1).min()).item()
        if hasattr(blk.moe, "expert_ema_fraction"):
            ema = blk.moe.expert_ema_fraction
            ema_max = ema.max().item()
            ema_min = ema.min().item()
            bias_ema_maxes.append(ema_max)
            bias_ema_mins.append(ema_min)
            metrics[f"moe/bias_ema_max_b{bi}"] = ema_max
            metrics[f"moe/bias_ema_min_b{bi}"] = ema_min

        for li, v in enumerate(blk.moe.last_per_token_entropy):
            metrics[f"moe/per_token_entropy_b{bi}_l{li}"] = v
            per_token_ents.append(v)
        for li, v in enumerate(blk.moe.last_marginal_entropy):
            metrics[f"moe/marginal_entropy_b{bi}_l{li}"] = v
            marginal_ents.append(v)
        for li, v in enumerate(blk.moe.last_assignment_entropy):
            metrics[f"moe/assignment_entropy_b{bi}_l{li}"] = v
            assign_ents.append(v)
        for li, v in enumerate(blk.moe.last_clean_per_token_entropy):
            metrics[f"moe/clean_per_token_entropy_b{bi}_l{li}"] = v
            clean_per_token_ents.append(v)
        for li, v in enumerate(blk.moe.last_clean_marginal_entropy):
            metrics[f"moe/clean_marginal_entropy_b{bi}_l{li}"] = v
            clean_marginal_ents.append(v)
        for li, v in enumerate(blk.moe.last_clean_assignment_entropy):
            metrics[f"moe/clean_assignment_entropy_b{bi}_l{li}"] = v
            clean_assign_ents.append(v)
        for li, v in enumerate(blk.moe.last_raw_max_prob):
            metrics[f"moe/raw_max_prob_b{bi}_l{li}"] = v
            raw_max_probs.append(v)
        for li, v in enumerate(blk.moe.last_top_margin):
            metrics[f"moe/top_margin_b{bi}_l{li}"] = v
            top_margins.append(v)
        for li, v in enumerate(blk.moe.last_clean_raw_max_prob):
            metrics[f"moe/clean_raw_max_prob_b{bi}_l{li}"] = v
            clean_raw_max_probs.append(v)
        for li, v in enumerate(blk.moe.last_clean_top_margin):
            metrics[f"moe/clean_top_margin_b{bi}_l{li}"] = v
            clean_top_margins.append(v)
        for li, v in enumerate(blk.moe.last_drop_rate):
            metrics[f"moe/drop_rate_b{bi}_l{li}"] = v
            drop_rates.append(v)
        for li, fracs in enumerate(blk.moe.last_expert_fraction):
            if fracs:
                mx = max(fracs)
                mn = min(fracs)
                metrics[f"moe/expert_max_b{bi}_l{li}"] = mx
                metrics[f"moe/expert_min_b{bi}_l{li}"] = mn
                expert_maxes.append(mx)
                expert_mins.append(mn)
                # Dead experts: load below 10% of the uniform share (1/E).
                # Counts per (block, loop) and accumulates a global total — a
                # rising total is the clearest single collapse signal.
                dead = sum(1 for f in fracs if f < 0.1 / len(fracs))
                metrics[f"moe/dead_experts_b{bi}_l{li}"] = dead
                dead_experts_total += dead
        for li, fracs in enumerate(blk.moe.last_clean_expert_fraction):
            if fracs:
                mx = max(fracs)
                mn = min(fracs)
                metrics[f"moe/clean_expert_max_b{bi}_l{li}"] = mx
                metrics[f"moe/clean_expert_min_b{bi}_l{li}"] = mn
                clean_expert_maxes.append(mx)
                clean_expert_mins.append(mn)

        # Prebias raw-router telemetry — exposed by the model when the
        # bias controller is enabled. Guarded with hasattr so older
        # checkpoints / tests without the attribute don't crash.
        if hasattr(blk.moe, "last_prebias_per_token_entropy"):
            for li, v in enumerate(blk.moe.last_prebias_per_token_entropy):
                metrics[f"moe/prebias_per_token_entropy_b{bi}_l{li}"] = v
                prebias_per_token_ents.append(v)
            for li, v in enumerate(blk.moe.last_prebias_marginal_entropy):
                metrics[f"moe/prebias_marginal_entropy_b{bi}_l{li}"] = v
                prebias_marginal_ents.append(v)
            for li, v in enumerate(blk.moe.last_prebias_assignment_entropy):
                metrics[f"moe/prebias_assignment_entropy_b{bi}_l{li}"] = v
                prebias_assign_ents.append(v)
            for li, v in enumerate(blk.moe.last_prebias_raw_max_prob):
                metrics[f"moe/prebias_raw_max_prob_b{bi}_l{li}"] = v
                prebias_raw_max_probs.append(v)
            for li, v in enumerate(blk.moe.last_prebias_top_margin):
                metrics[f"moe/prebias_top_margin_b{bi}_l{li}"] = v
                prebias_top_margins.append(v)
            for li, fracs in enumerate(blk.moe.last_prebias_expert_fraction):
                if fracs:
                    mx = max(fracs)
                    mn = min(fracs)
                    metrics[f"moe/prebias_expert_max_b{bi}_l{li}"] = mx
                    metrics[f"moe/prebias_expert_min_b{bi}_l{li}"] = mn
                    prebias_expert_maxes.append(mx)
                    prebias_expert_mins.append(mn)

    # Recursive-loop collapse: per-effective-layer residual update ||Δx||/||x||.
    # A deep loop whose update → 0 has collapsed to a no-op; update_norm_last
    # (the deepest effective layer) and update_norm_min are the at-a-glance
    # signals. hidden_norm tracks residual-stream growth/blowup.
    loop_norms = list(getattr(base, "last_loop_update_norm", []) or [])
    for idx, v in enumerate(loop_norms):
        metrics[f"loop/update_norm_l{idx}"] = v
    for idx, v in enumerate(getattr(base, "last_loop_hidden_norm", []) or []):
        metrics[f"loop/hidden_norm_l{idx}"] = v
    if loop_norms:
        metrics["loop/update_norm_mean"] = _mean(loop_norms)
        metrics["loop/update_norm_min"] = min(loop_norms)
        metrics["loop/update_norm_last"] = loop_norms[-1]
    metrics["moe/dead_experts_total"] = dead_experts_total

    metrics["moe/per_token_entropy_mean"] = _mean(per_token_ents)
    metrics["moe/marginal_entropy_mean"] = _mean(marginal_ents)
    metrics["moe/assignment_entropy_mean"] = _mean(assign_ents)
    metrics["moe/clean_per_token_entropy_mean"] = _mean(clean_per_token_ents)
    metrics["moe/clean_marginal_entropy_mean"] = _mean(clean_marginal_ents)
    metrics["moe/clean_assignment_entropy_mean"] = _mean(clean_assign_ents)
    metrics["moe/raw_max_prob_mean"] = _mean(raw_max_probs)
    metrics["moe/top_margin_mean"] = _mean(top_margins)
    metrics["moe/clean_raw_max_prob_mean"] = _mean(clean_raw_max_probs)
    metrics["moe/clean_top_margin_mean"] = _mean(clean_top_margins)
    metrics["moe/drop_rate_mean"] = _mean(drop_rates)
    metrics["moe/moe_gate_mean"] = _mean(moe_gates)
    metrics["moe/expert_max_mean"] = _mean(expert_maxes)
    metrics["moe/expert_min_mean"] = _mean(expert_mins)
    metrics["moe/clean_expert_max_mean"] = _mean(clean_expert_maxes)
    metrics["moe/clean_expert_min_mean"] = _mean(clean_expert_mins)
    metrics["moe/balance_loss_mean"] = _mean(balance_losses)
    metrics["moe/bias_abs_max_mean"] = _mean(bias_abs_maxes)
    metrics["moe/bias_ema_max_mean"] = _mean(bias_ema_maxes)
    metrics["moe/bias_ema_min_mean"] = _mean(bias_ema_mins)
    metrics["moe/prebias_per_token_entropy_mean"] = _mean(prebias_per_token_ents)
    metrics["moe/prebias_marginal_entropy_mean"] = _mean(prebias_marginal_ents)
    metrics["moe/prebias_assignment_entropy_mean"] = _mean(prebias_assign_ents)
    metrics["moe/prebias_raw_max_prob_mean"] = _mean(prebias_raw_max_probs)
    metrics["moe/prebias_top_margin_mean"] = _mean(prebias_top_margins)
    metrics["moe/prebias_expert_max_mean"] = _mean(prebias_expert_maxes)
    metrics["moe/prebias_expert_min_mean"] = _mean(prebias_expert_mins)

    summary = {
        "per_token_H": _mean(per_token_ents),
        "marginal_H": _mean(marginal_ents),
        "assign_H": _mean(assign_ents),
        "clean_per_token_H": _mean(clean_per_token_ents),
        "clean_marginal_H": _mean(clean_marginal_ents),
        "clean_assign_H": _mean(clean_assign_ents),
        "raw_max": _mean(raw_max_probs),
        "top_margin": _mean(top_margins),
        "clean_raw_max": _mean(clean_raw_max_probs),
        "clean_top_margin": _mean(clean_top_margins),
        "drop_rate": _mean(drop_rates),
        "moe_gate": _mean(moe_gates),
        "expert_max": _mean(expert_maxes),
        "expert_min": _mean(expert_mins),
        "clean_expert_max": _mean(clean_expert_maxes),
        "clean_expert_min": _mean(clean_expert_mins),
        "balance_loss": _mean(balance_losses),
        "bias_abs_max": _mean(bias_abs_maxes),
        "bias_ema_max": _mean(bias_ema_maxes),
        "bias_ema_min": _mean(bias_ema_mins),
        "prebias_per_token_H": _mean(prebias_per_token_ents),
        "prebias_marginal_H": _mean(prebias_marginal_ents),
        "prebias_assign_H": _mean(prebias_assign_ents),
        "prebias_raw_max": _mean(prebias_raw_max_probs),
        "prebias_top_margin": _mean(prebias_top_margins),
        "prebias_expert_max": _mean(prebias_expert_maxes),
        "prebias_expert_min": _mean(prebias_expert_mins),
        "loop_update_norm_min": min(loop_norms) if loop_norms else 0.0,
        "loop_update_norm_mean": _mean(loop_norms),
        "loop_update_norm_last": loop_norms[-1] if loop_norms else 0.0,
        "dead_experts_total": dead_experts_total,
    }
    return metrics, summary



def _hidden_norm_ratio(avg: dict) -> float:
    """Residual-stream growth WITHIN the recursion, past the first block.

    Layer 0's hidden norm is the embedding (RMS ~0.02) and norm_loop resets
    the stream to unit RMS at every loop boundary, so "deepest / first" is
    embedding-vs-block-output and reads 1e4 on a healthy run. The ladder's
    W&B norms (2026-09-02) showed the real pathology is inside a loop: on
    the HRA arms each block's output was 50-300x its input, so the stream
    inflated four orders of magnitude across three blocks (g4: 3e9). Measure
    that: the largest hidden norm at any effective layer >= 1, over the
    first block's output. ~1-2 is a residual network; >= 50 means blocks are
    overwriting the stream rather than refining it (FLT §17.3).
    """
    keys = sorted(
        (k for k in avg if k.startswith("loop/hidden_norm_l")),
        key=lambda k: int(k.rsplit("l", 1)[1]),
    )
    if len(keys) < 3 or not avg.get(keys[1]):
        return 1.0
    first_block_out = float(avg[keys[1]])
    return max(float(avg[k]) for k in keys[1:]) / first_block_out


def _average_moe_snapshots(
    snapshots: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Average a list of per-micro-batch MoE snapshots into one dict.

    Each snapshot is a wandb-keyed metrics dict from _collect_moe_metrics.
    Averaging element-wise gives us the per-step MoE state averaged over
    all grad_accum micro-batches — much less noisy than taking only the
    last micro-batch, which matters for the 5k early-stop gate where a
    single-batch outlier on clean_raw_max_prob could false-trip.

    Returns (metrics_avg, summary). summary is rebuilt from the averaged
    *_mean keys so stdout formatting is unchanged.
    """
    if not snapshots:
        return {}, {}
    n = len(snapshots)
    keys = snapshots[0].keys()
    avg: dict[str, float] = {}
    for k in keys:
        total = 0.0
        for snap in snapshots:
            total += snap.get(k, 0.0)
        avg[k] = total / n
    # Residual-stream growth inside the recursion (roadmap §17.3). Put it in
    # the logged metrics too — until 2026-09-30 it existed only in the gate's
    # summary, so RUNBOOK's "watch loop_hidden_norm_ratio" had nothing to watch.
    avg["loop/hidden_norm_ratio"] = _hidden_norm_ratio(avg)

    summary = {
        "per_token_H": avg.get("moe/per_token_entropy_mean", 0.0),
        "marginal_H": avg.get("moe/marginal_entropy_mean", 0.0),
        "assign_H": avg.get("moe/assignment_entropy_mean", 0.0),
        "clean_per_token_H": avg.get("moe/clean_per_token_entropy_mean", 0.0),
        "clean_marginal_H": avg.get("moe/clean_marginal_entropy_mean", 0.0),
        "clean_assign_H": avg.get("moe/clean_assignment_entropy_mean", 0.0),
        "raw_max": avg.get("moe/raw_max_prob_mean", 0.0),
        "top_margin": avg.get("moe/top_margin_mean", 0.0),
        "clean_raw_max": avg.get("moe/clean_raw_max_prob_mean", 0.0),
        "clean_top_margin": avg.get("moe/clean_top_margin_mean", 0.0),
        "drop_rate": avg.get("moe/drop_rate_mean", 0.0),
        "moe_gate": avg.get("moe/moe_gate_mean", 0.0),
        "expert_max": avg.get("moe/expert_max_mean", 0.0),
        "expert_min": avg.get("moe/expert_min_mean", 0.0),
        "clean_expert_max": avg.get("moe/clean_expert_max_mean", 0.0),
        "clean_expert_min": avg.get("moe/clean_expert_min_mean", 0.0),
        "balance_loss": avg.get("moe/balance_loss_mean", 0.0),
        "bias_abs_max": avg.get("moe/bias_abs_max_mean", 0.0),
        "bias_ema_max": avg.get("moe/bias_ema_max_mean", 0.0),
        "bias_ema_min": avg.get("moe/bias_ema_min_mean", 0.0),
        "prebias_per_token_H": avg.get(
            "moe/prebias_per_token_entropy_mean", 0.0,
        ),
        "prebias_marginal_H": avg.get(
            "moe/prebias_marginal_entropy_mean", 0.0,
        ),
        "prebias_assign_H": avg.get(
            "moe/prebias_assignment_entropy_mean", 0.0,
        ),
        "prebias_raw_max": avg.get("moe/prebias_raw_max_prob_mean", 0.0),
        "prebias_top_margin": avg.get("moe/prebias_top_margin_mean", 0.0),
        "prebias_expert_max": avg.get("moe/prebias_expert_max_mean", 0.0),
        "prebias_expert_min": avg.get("moe/prebias_expert_min_mean", 0.0),
        "loop_update_norm_min": avg.get("loop/update_norm_min", 0.0),
        "loop_update_norm_mean": avg.get("loop/update_norm_mean", 0.0),
        "loop_hidden_norm_ratio": avg["loop/hidden_norm_ratio"],
        "loop_update_norm_last": avg.get("loop/update_norm_last", 0.0),
        "dead_experts_total": avg.get("moe/dead_experts_total", 0.0),
    }
    return avg, summary


def _check_early_stop_criteria(
    step: int, summary: dict, cfg: PretrainConfig, model_cfg: OSRTConfig,
    *, scope: str = "full",
) -> list[str]:
    """Return list of failing criteria (empty means all pass).

    scope="full": router sharpening + balance + recursion health (the gate at
    `early_stop_check_step`, and every logging step after it).
    scope="loop": recursion health only (loop collapse, residual explosion) —
    what can be judged before the router has had time to sharpen.
    """
    if scope not in ("full", "loop"):
        raise ValueError(f"scope must be 'full' or 'loop', got {scope!r}")
    failures: list[str] = []
    if scope == "full":
        failures.extend(_router_health_failures(summary, cfg, model_cfg))
    # Recursive-loop health (roadmap §17.3). Both thresholds existed in
    # PretrainConfig and both values were computed into the summary, but
    # nothing compared them — the first ladder ran with these guards inert.
    lu_min = summary.get("loop_update_norm_min")
    if lu_min is not None and lu_min < cfg.min_loop_update_norm:
        failures.append(
            f"loop_update_norm_min {lu_min:.2e} < {cfg.min_loop_update_norm:.2e} "
            "(a loop's residual write has vanished — loop collapse)"
        )
    hn_ratio = summary.get("loop_hidden_norm_ratio")
    if hn_ratio is not None and hn_ratio > cfg.max_loop_hidden_norm_ratio:
        failures.append(
            f"loop_hidden_norm_ratio {hn_ratio:.1f} > "
            f"{cfg.max_loop_hidden_norm_ratio:.1f} "
            "(residual stream inflates across the recursion — FLT §17.3)"
        )
    return failures


def _router_health_failures(
    summary: dict, cfg: PretrainConfig, model_cfg: OSRTConfig,
) -> list[str]:
    """The router-health criteria (v5's four-metric gate plus the pre-bias and
    bias-saturation checks), resolved relative to this model's expert count."""
    failures: list[str] = []
    per_token_h = summary.get("clean_per_token_H", summary["per_token_H"])
    raw_max = summary.get("clean_raw_max", summary["raw_max"])
    top_margin = summary.get("clean_top_margin", summary["top_margin"])
    marginal_h = summary.get("clean_marginal_H", summary["marginal_H"])
    # per_token_entropy starts near ln(num_routed) at init (uniform router)
    # and must drop by at least min_per_token_entropy_drop. This was a
    # hard-coded 2.079 = ln(8) — correct for v6's 8 experts, silently wrong
    # for v7's 28 (ln 28 = 3.332), which would have let an unsharpened router
    # pass as sharpened by ~1.25 nats.
    # Resolve the relative thresholds against THIS model's expert count and
    # top-k (see train_config for why absolutes were wrong at E=28).
    E, K = model_cfg.num_routed_experts, model_cfg.top_k_experts
    ln_e = math.log(E)
    target_pte = ln_e - cfg.per_token_entropy_drop_frac * ln_e
    min_raw_max = cfg.raw_max_prob_frac_of_topk / K
    min_margin = cfg.top_margin_frac_of_topk / K
    min_marginal = cfg.marginal_entropy_frac * ln_e
    min_prebias_marginal = cfg.prebias_marginal_entropy_frac * ln_e
    min_prebias_expert = cfg.prebias_expert_fraction_of_uniform / E

    if per_token_h > target_pte:
        failures.append(
            f"clean_per_token_entropy {per_token_h:.3f} > {target_pte:.3f} "
            f"(router hasn't sharpened; init ln {E} = {ln_e:.3f})"
        )
    if raw_max < min_raw_max:
        failures.append(
            f"clean_raw_max_prob {raw_max:.3f} < {min_raw_max:.3f} "
            f"(no strong primary pick at top-{K})"
        )
    if top_margin < min_margin:
        failures.append(
            f"clean_top_margin {top_margin:.3f} < {min_margin:.3f} "
            "(no gap between rank 0 and rank 1)"
        )
    if marginal_h < min_marginal:
        failures.append(
            f"clean_marginal_entropy {marginal_h:.3f} < {min_marginal:.3f} "
            "(dead/overloaded experts)"
        )
    prebias_marginal_h = summary.get("prebias_marginal_H", marginal_h)
    if prebias_marginal_h < min_prebias_marginal:
        failures.append(
            f"prebias_marginal_entropy {prebias_marginal_h:.3f} < "
            f"{min_prebias_marginal:.3f} "
            "(learned router collapsed before balance bias)"
        )
    prebias_expert_min = summary.get("prebias_expert_min", 1.0)
    if prebias_expert_min < min_prebias_expert:
        failures.append(
            f"prebias_expert_min {prebias_expert_min:.4f} < "
            f"{min_prebias_expert:.4f} "
            f"(an expert is below {cfg.prebias_expert_fraction_of_uniform:.0%} "
            "of its fair share before balance bias)"
        )
    bias_limit = (
        model_cfg.router_balance_bias_max
        * getattr(cfg, "max_bias_saturation_fraction", 1.0)
    )
    if model_cfg.router_balance_bias_enabled and bias_limit > 0:
        bias_abs_max = summary.get("bias_abs_max", 0.0)
        if bias_abs_max > bias_limit:
            failures.append(
                f"router_balance_bias_abs_max {bias_abs_max:.3f} > "
                f"{bias_limit:.3f} "
                "(bias controller is near saturation and may be masking collapse)"
            )
    return failures


def _failure_key(failure: str) -> str:
    """The criterion a failure message belongs to: every message from
    `_check_early_stop_criteria` opens with its metric name."""
    return failure.split(" ", 1)[0]


def _update_health_streaks(streaks: dict[str, int], failures: list[str]) -> list[str]:
    """Advance the per-criterion consecutive-failure counts in `streaks` and
    return the criteria that failed on THIS check, in reported order.

    Patience means "the same criterion keeps failing", not "something failed
    on each of the last N checks": a criterion that is absent from `failures`
    drops back to zero, so three unrelated one-off blips never add up to a
    stop, while one metric that stays bad for `health_check_patience` checks
    does.
    """
    keys = [_failure_key(f) for f in failures]
    for k in list(streaks):
        if k not in keys:
            del streaks[k]
    for k in keys:
        streaks[k] = streaks.get(k, 0) + 1
    return keys


def _health_scope(step: int, cfg: PretrainConfig) -> str | None:
    """Which criteria to evaluate at `step` (None = none).

    * at `early_stop_check_step`: the full set, one-shot semantics (the
      caller stops immediately on failure, as v5's gate always did);
    * after it: the full set, with `health_check_patience`;
    * before it, once warmup is over and `continuous_health_checks` is on:
      recursion health only — the router is not expected to be sharp yet.
    """
    gate = cfg.early_stop_check_step
    if step == gate:
        return "full"
    if not getattr(cfg, "continuous_health_checks", True):
        return None
    if step > gate:
        return "full"
    if step >= cfg.warmup_steps:
        return "loop"
    return None


def _alias_checkpoint(src: str, dst: str) -> None:
    """Expose `src` under a second name (hard link; copy where links are not
    supported, e.g. some network volumes). Used at the end of a run so the
    final checkpoint is also visible as `osrt_step_{total_steps}.pt`: the
    resume scan and the HF sync only look at step-numbered names, so without
    the alias a re-invocation of a FINISHED run resumed from the last interval
    save and re-trained the tail."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        # Copy under a name that neither the resume scan (`{prefix}_step_*.pt`)
        # nor the sync daemon (`_SYNC_RE`, which needs the `.pt` suffix)
        # matches, then publish with an atomic rename — the same discipline
        # save_checkpoint uses — so a multi-GB copy on a link-less volume never
        # exposes a truncated alias for the daemon to upload as complete.
        tmp = f"{dst}.tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)


@torch.no_grad()
def _reset_router_balance_accumulators(model: nn.Module) -> None:
    """Drop the per-step routing statistics after a skipped (non-finite)
    optimizer step, so the next balance-bias update is not solved from a
    forward whose activations may have been garbage."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    base = inner.model if hasattr(inner, "model") else inner
    for block in base.blocks:
        moe = block.moe
        for name in ("balance_count_accum", "balance_total_accum",
                     "qb_hist", "qb_token_count"):
            buf = getattr(moe, name, None)
            if buf is not None:
                buf.zero_()


def run_training(
    model_config: OSRTConfig,
    train_cfg: PretrainConfig,
    vol,
    tokenizer_name: str,
    ckpt_dir: str = "/vol/checkpoints/v7",
) -> str:
    """Execute the v7 pre-training loop.

    Args:
        model_config: model configuration (the committed preset + tokenizer ids).
        train_cfg: training hyperparameters + phase schedule.
        vol: Modal Volume for checkpoints (anything with `.commit()`).
        tokenizer_name: path or HF id of the tokenizer.
        ckpt_dir: directory for checkpoints. Sanity/test runs should pass a
            distinct dir to avoid colliding with real checkpoints.

    Returns a status string: "complete", "already_complete", "early_stop",
    "rescued" or "data_dead" (see the module docstring).
    """
    # Fail closed on an inconsistent recipe before any compute is spent.
    train_cfg.validate()
    device = torch.device("cuda")

    print("=" * 60)
    print("OSRT — Mixtral MoE Pre-training")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Model setup
    model = OSRTForCausalLM(model_config).to(device=device)
    model.train()

    # Gradient (activation) checkpointing — enable EXPLICITLY here, AFTER
    # construction, and print confirmation. The model-config flag alone is
    # unreliable: HF PreTrainedModel.post_init() runs
    # _backward_compatibility_gradient_checkpointing, which (on newer
    # transformers) calls HF's gradient_checkpointing_enable() — our custom
    # model doesn't hook into it — and then RESETS config.gradient_checkpointing
    # to False. So we read the flag from the TRAIN config (which HF never
    # touches) OR the model config, and set model.model.gradient_checkpointing
    # authoritatively here. This is the override that makes the recursive blocks
    # recompute in backward, which is what makes the full batch/seq fit.
    # Driven ONLY by the train config — gradient_checkpointing is never put on
    # the model config (it's an HF-managed name that breaks post_init). Set OUR
    # private gate, which model.py's use_ckpt reads.
    gc_on = bool(getattr(train_cfg, "gradient_checkpointing", False))
    model.model._osrt_grad_ckpt = gc_on
    print(
        f"Gradient checkpointing: {'ENABLED' if gc_on else 'disabled'} "
        f"(model._osrt_grad_ckpt={model.model._osrt_grad_ckpt})"
    )
    fce = getattr(model_config, "fused_cross_entropy_chunks", 0)
    print(f"Fused linear-CE chunks: {fce} ({'on' if fce > 0 else 'off'})")

    total_params = sum(p.numel() for p in model.parameters())
    n_experts = 1 + model_config.num_routed_experts  # +1 shared
    print(f"Physical parameters : {total_params:>12,}")
    print(f"Blocks              : {model_config.num_blocks}")
    print(f"Recursive loops     : {model_config.recursive_loops}")
    print(
        f"Effective layers    : "
        f"{model_config.num_blocks * model_config.recursive_loops}"
    )
    print(
        f"Experts             : {n_experts} "
        f"(1 shared + {model_config.num_routed_experts} routed, "
        f"top-{model_config.top_k_experts})"
    )
    print(f"Hidden dim          : {model_config.dim}")
    print(f"Peak LR             : {train_cfg.peak_lr}")
    print(f"Optimizer           : {train_cfg.optimizer_name}")
    print(f"Total steps         : {train_cfg.total_steps}")
    print(f"Aux loss coeff      : {model_config.router_aux_loss_coeff}")
    print(
        f"Balance bias        : {model_config.router_balance_bias_enabled} "
        f"(rate={model_config.router_balance_bias_update_rate}, "
        f"max={model_config.router_balance_bias_max})"
    )
    print(
        f"Router Gumbel tau   : {train_cfg.router_gumbel_tau_init} -> "
        f"{train_cfg.router_gumbel_tau_final} over "
        f"{train_cfg.router_gumbel_anneal_steps} steps"
    )
    print()

    # compile_enabled opt-out (default True) lets a stage skip torch.compile
    # entirely. Eager mode starts producing step
    # events immediately instead of ~10 min of silent compile tracing, which
    # is what a short smoke test wants.
    if getattr(train_cfg, "compile_enabled", True):
        # B4 grouped-GEMM MoE: torch._grouped_mm with data-dependent offsets
        # needs these two dynamo flags or it graph-breaks on .item()/.tolist()
        # of the per-expert offsets. Harmless when grouped is off; only set
        # when both compile and grouped are on. With them set, the model
        # compiles fullgraph (verified: 0 breaks vs 12 for the loop path).
        if getattr(model_config, "moe_grouped_gemm", False):
            import torch._dynamo as _dynamo
            _dynamo.config.capture_scalar_outputs = True
            _dynamo.config.capture_dynamic_output_shape_ops = True
            print("Grouped-GEMM MoE: enabled dynamo scalar/dynamic-shape capture")
        print("Compiling model with torch.compile...")
        compile_start = time.time()
        model = torch.compile(model)
        print(f"Model compile done in {time.time() - compile_start:.1f}s")
    else:
        print("\nSkipping torch.compile (compile_enabled=False, eager mode).")

    # W&B
    use_wandb = train_cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb_kwargs = {
            "project": train_cfg.wandb_project,
            "name": train_cfg.wandb_run_name,
            "config": {
                "stage": "pretrain",
                "total_params": total_params,
                "architecture": "mixtral_moe",
                "num_blocks": model_config.num_blocks,
                "recursive_loops": model_config.recursive_loops,
                "num_routed_experts": model_config.num_routed_experts,
                "top_k": model_config.top_k_experts,
                "expert_hidden": model_config.expert_hidden,
                "shared_expert_hidden": model_config.shared_expert_hidden,
                "capacity_factor": model_config.router_capacity_factor,
                "aux_loss_coeff": model_config.router_aux_loss_coeff,
                "balance_bias_enabled": (
                    model_config.router_balance_bias_enabled
                ),
                "balance_bias_update_rate": (
                    model_config.router_balance_bias_update_rate
                ),
                "balance_bias_max": model_config.router_balance_bias_max,
                "router_gumbel_tau_init": train_cfg.router_gumbel_tau_init,
                "router_gumbel_tau_final": train_cfg.router_gumbel_tau_final,
                "router_gumbel_anneal_steps": (
                    train_cfg.router_gumbel_anneal_steps
                ),
                "peak_lr": train_cfg.peak_lr,
                "optimizer": train_cfg.optimizer_name,
                "total_steps": train_cfg.total_steps,
            },
        }
        if train_cfg.wandb_run_id:
            wandb_kwargs["id"] = train_cfg.wandb_run_id
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)
        print("W&B logging enabled.")

    # Optimizer — router/loop_embeddings get wd=0 (they're routing-sensitive).
    # Three branches today:
    #   "lion"  — single Lion over all params (router gets a wd=0 group)
    #   "muon"  — hybrid Muon (2D matrix weights) + AdamW (embeddings,
    #             norms, scalars, router/loop_embeddings)
    #   else    — AdamW fallback
    inner_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    if train_cfg.optimizer_name.lower() == "muon":
        from osrt.muon import (
            HybridMuonAdamW,
            Muon,
            build_param_groups,
        )

        muon_params, adamw_groups = build_param_groups(
            inner_model.named_parameters(),
            weight_decay=train_cfg.weight_decay,
            per_head_attn=getattr(train_cfg, "per_head_muon", False),
            head_dim=model_config.head_dim,
        )
        # Muon LR is much smaller-magnitude than Lion/AdamW because the
        # Newton-Schulz update is normalised. Keep it as a separate
        # config knob so users can A/B without nuking the Lion peak_lr.
        muon_lr = getattr(train_cfg, "muon_lr", train_cfg.peak_lr)
        # DeepSeek-V4 Muon recipe (roadmap §14.1 item 1.3): hybrid
        # Newton-Schulz — fast iterations for convergence, then stabilising
        # (2.0, -1.5, 0.5) passes — with the update RMS rescaled rather than
        # left to the shape heuristic.
        muon = Muon(
            muon_params,
            lr=muon_lr,
            momentum=getattr(train_cfg, "muon_momentum", 0.95),
            nesterov=True,
            ns_steps=getattr(train_cfg, "muon_ns_steps", 5),
            ns_stable_steps=getattr(train_cfg, "muon_ns_stable_steps", 0),
            update_rms=getattr(train_cfg, "muon_update_rms", None),
            weight_decay=train_cfg.weight_decay,
        )
        adamw = torch.optim.AdamW(
            adamw_groups,
            lr=train_cfg.peak_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        optimizer = HybridMuonAdamW(muon, adamw)
        # Tag each group with its peak/floor for the schedule (re-stamped
        # after a checkpoint load, which would otherwise restore old tags).
        _stamp_schedule_tags(optimizer, train_cfg)
        n_muon = sum(len(g["params"]) for g in muon.param_groups)
        n_adamw = sum(len(g["params"]) for g in adamw_groups)
        per_head = getattr(train_cfg, "per_head_muon", False)
        rms = getattr(train_cfg, "muon_update_rms", None)
        per_elem = (
            f"{muon_lr * rms:.2e} (lr x update_rms {rms})" if rms
            else f"~{muon_lr / model_config.dim ** 0.5:.2e} (shape heuristic)"
        )
        print(
            f"Using Muon+AdamW hybrid: {n_muon} matrix tensors → Muon "
            f"(lr={muon_lr}{', per-head attn' if per_head else ''}, "
            f"decoupled wd={train_cfg.weight_decay}), "
            f"{n_adamw} other tensors → AdamW (lr={train_cfg.peak_lr}, wd=0: "
            f"embedding, norms, biases, router, loop_embeddings, moe_gate)"
        )
        print(
            f"Muon per-element step at peak LR: {per_elem}; "
            f"AdamW peak_lr={train_cfg.peak_lr}. The two should be on the "
            f"same scale — see train_config.muon_lr."
        )
    else:
        router_params = []
        other_params = []
        for name, param in inner_model.named_parameters():
            if not param.requires_grad:
                continue
            if "router" in name or "loop_embeddings" in name:
                router_params.append(param)
            else:
                other_params.append(param)

        print(
            f"Param groups: {len(other_params)} standard, "
            f"{len(router_params)} router (wd=0)"
        )

        if train_cfg.optimizer_name.lower() == "lion":
            from lion_pytorch import Lion
            optimizer = Lion(
                [
                    {"params": other_params, "weight_decay": train_cfg.weight_decay},
                    {"params": router_params, "weight_decay": 0.0},
                ],
                lr=train_cfg.peak_lr,
            )
            print(f"Using Lion (wd={train_cfg.weight_decay}, router_wd=0.0)")
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": other_params, "weight_decay": train_cfg.weight_decay},
                    {"params": router_params, "weight_decay": 0.0},
                ],
                lr=train_cfg.peak_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
            print(f"Using AdamW (wd={train_cfg.weight_decay}, router_wd=0.0)")
        _stamp_schedule_tags(optimizer, train_cfg)

    # Checkpoint resume.
    # Three kinds of checkpoints with different naming:
    #   osrt_step_N.pt          — normal interval save (resumable)
    #   osrt_rescue_step_N.pt   — 23h timeout rescue    (resumable)
    #   osrt_failed_step_N.pt   — failed early-stop     (NOT resumed; the
    #     run declared itself bad. If you want to investigate or force-resume
    #     anyway, rename the file to osrt_step_N.pt explicitly.)
    # Resume scans the first two patterns and picks the highest step.
    os.makedirs(ckpt_dir, exist_ok=True)

    def _extract_step(path: str) -> int | None:
        """Extract the step number from an osrt_..._step_N.pt path."""
        try:
            return int(path.rsplit("_", 1)[1].split(".")[0])
        except (ValueError, IndexError):
            return None

    best_step = -1
    best_ckpt: str | None = None
    # Scan order matters for tie-breaking: if the 23h rescue fires on a
    # ckpt_interval step, both osrt_step_N.pt and
    # osrt_rescue_step_N.pt exist at step N. The files contain
    # identical optimizer state (same end-of-step save) but we prefer
    # the rescue variant because it's the intentional "resume here"
    # marker — scanning rescue AFTER normal with `>=` makes rescue win
    # on ties. When steps differ, higher step always wins regardless
    # of pattern.
    for pattern in (
        f"{ckpt_dir}/osrt_step_*.pt",
        f"{ckpt_dir}/osrt_rescue_step_*.pt",
    ):
        for f in glob.glob(pattern):
            s = _extract_step(f)
            if s is None:
                continue
            if s > best_step or (s == best_step and "rescue" in f):
                best_step = s
                best_ckpt = f

    # Explicit notice if there's a FAILED checkpoint — user should know.
    failed_paths = sorted(glob.glob(f"{ckpt_dir}/osrt_failed_step_*.pt"))
    if failed_paths:
        print(
            f"WARNING: Found {len(failed_paths)} failed-early-stop "
            f"checkpoint(s): {[os.path.basename(p) for p in failed_paths]}. "
            f"These are NOT resumed automatically. Rename to "
            f"osrt_step_N.pt if you want to force-resume.",
        )

    start_step = 0
    resume_data_state: dict | None = None
    if best_step > 0 and best_ckpt is not None:
        print(f"Found checkpoint at step {best_step}: {best_ckpt}")
        # Fails closed BEFORE the weights are applied: a months-long drip run
        # resumes many times, and a silent config change makes the loss curve
        # a splice of two experiments rather than one result.
        start_step, resume_data_state = load_checkpoint(
            model, optimizer, best_ckpt, device,
            model_config=model_config, train_cfg=train_cfg,
            tokenizer_name=tokenizer_name,
        )
    if start_step >= train_cfg.total_steps:
        print(
            f"Run already complete: checkpoint step {start_step - 1} >= "
            f"total_steps {train_cfg.total_steps}. Nothing to do.",
            flush=True,
        )
        if use_wandb:
            wandb.finish()
        return "already_complete"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    start_time = time.time()
    step = start_step
    current_phase: str | None = None
    current_loader = None
    loader_iter = None
    current_seq_len = 2048
    current_batch_size = train_cfg.batch_size
    grad_accum = train_cfg.grad_accum_steps
    early_stop_triggered = False
    run_status: str | None = None
    health_fail_streaks: dict[str, int] = {}   # criterion -> consecutive fails
    nonfinite_streak = 0
    nonfinite_total = 0
    dead_sources_seen: list[str] = []
    max_nonfinite = max(1, getattr(train_cfg, "max_consecutive_nonfinite_steps", 5))

    def _data_state() -> dict | None:
        """The streaming loader's position, for the checkpoint (None when the
        loader cannot report one, e.g. worker processes)."""
        ds = getattr(current_loader, "dataset", None)
        fn = getattr(ds, "state_dict", None)
        if fn is None:
            return None
        try:
            state = fn()
        except Exception as e:  # noqa: BLE001 — never let this cost a checkpoint
            print(f"  [warn] data position not captured: {type(e).__name__}: {e}",
                  flush=True)
            return None
        return {"phase": current_phase, "state": state} if state is not None else None

    def _save(path: str, at_step: int) -> None:
        save_checkpoint(
            model, optimizer, at_step, path,
            model_config=model_config, train_cfg=train_cfg,
            data_state=_data_state(), tokenizer_name=tokenizer_name,
        )
        vol.commit()

    def _stop_failed(reason: str) -> None:
        """Failed-state checkpoint under a name the resume scan ignores."""
        print(
            f"\n  {reason} Saving failed-state checkpoint (not auto-resumable) "
            "and exiting so compute isn't wasted. Review telemetry before "
            "retrying.",
            flush=True,
        )
        _save(f"{ckpt_dir}/osrt_failed_step_{step}.pt", step)

    while step < train_cfg.total_steps and not early_stop_triggered:
        phase_name, phase_cfg = get_phase(step, train_cfg)

        if phase_name != current_phase:
            current_phase = phase_name
            current_seq_len = phase_cfg["seq_len"]
            grad_accum = phase_cfg.get(
                "grad_accum_steps", train_cfg.grad_accum_steps,
            )
            current_batch_size = phase_cfg.get(
                "batch_size", train_cfg.batch_size,
            )

            print(
                f"\n>>> Phase: {current_phase} | seq_len: {current_seq_len} | "
                f"batch: {current_batch_size} | accum: {grad_accum} | "
                f"Step: {step}"
            )
            print(
                f"    Datasets: {[d['name'] for d in phase_cfg['datasets']]}"
            )
            # Phase-switch memory hygiene. The new (batch, seq) shape recompiles
            # and Triton autotunes the fused-CE backward by CLONING its (N, vocab)
            # logits inputs; with the previous shape's cached allocator blocks
            # still resident that pushed the first trunk launch over 192 GB at
            # this exact point. Drop everything the old shape held first.
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            if current_loader is not None:
                # Tear down the previous phase's loader BEFORE building the
                # new one. The iterator owns the loader, the loader owns
                # the worker processes; if we just rebind both vars in
                # sequence (assigning current_loader first, then
                # loader_iter), the old workers only get reaped when
                # loader_iter is overwritten — which happens *after* the
                # new workers have already spawned, racing teardown
                # against startup. With persistent_workers=True +
                # multiprocessing_context="spawn" + live HF streaming
                # connections (aiohttp/fsspec), that race manifested at
                # the foundation→knowledge transition (step 10000) as
                # "Fatal Python error: PyGILState_Release". Drop both
                # refs and force a GC pass so old workers fully exit
                # before any new ones come up.
                import gc
                loader_iter = None
                current_loader = None
                gc.collect()
            load_t = time.time()
            # Honor the config's worker count (default 0). make_loader
            # defaults to 4, but 4 workers × N streams opens too many
            # concurrent HF connections from one container, which triggers
            # SSL BAD_RECORD_MAC / "Bad file descriptor" / connection-reset
            # storms under any HF flakiness.
            loader_kwargs: dict = {
                "num_workers": getattr(train_cfg, "dataloader_num_workers", 0),
            }
            # Continue the data stream where the checkpoint left it — only
            # for the phase the checkpoint was taken in, and only for the
            # first loader after a resume. Otherwise every source restarts
            # at row 0 of a freshly permuted shard list.
            if (
                resume_data_state
                and resume_data_state.get("phase") == phase_name
                and step == start_step
                and resume_data_state.get("state") is not None
            ):
                loader_kwargs["resume_state"] = resume_data_state["state"]
                print("    Restoring the data stream position from the checkpoint")
            resume_data_state = None
            current_loader = make_loader(
                phase_cfg["datasets"],
                current_seq_len,
                tokenizer_name,
                current_batch_size,
                step,
                **loader_kwargs,
            )
            loader_iter = iter(current_loader)
            print(f"    DataLoader ready in {time.time() - load_t:.1f}s")
        else:
            grad_accum = phase_cfg.get(
                "grad_accum_steps", train_cfg.grad_accum_steps,
            )
            current_batch_size = phase_cfg.get(
                "batch_size", train_cfg.batch_size,
            )

        # Schedule writes per-group LRs honouring _peak_lr / _min_lr
        # tags, returning the AdamW/Lion-scale LR for stdout/W&B logs.
        lr = _set_param_group_lrs(optimizer, step, train_cfg)
        router_gumbel_tau = get_router_gumbel_tau(step, train_cfg)
        set_router_gumbel_tau(model, router_gumbel_tau)

        # Gradient checkpointing: force ON for very long seq (Phase 3,
        # seq 8192) even if the config didn't request it. The threshold is
        # 8192 (not Phase 2's 4096) because H100 80GB only fills ~14 GB at
        # Phase 2 sizes (batch 4 × accum 16 × seq 4096) — checkpointing
        # there would throw away ~50% throughput for headroom we don't
        # need; Phase 3 genuinely needs the memory relief. If activation
        # memory ever crowds the budget at seq 4096 (e.g. bigger batch on
        # H200 141GB), raise batch_size in train_config first; only lower
        # this threshold if that doesn't fit. The model's ONLY gate is
        # _osrt_grad_ckpt (model.py use_ckpt); HF's `gradient_checkpointing`
        # name is deliberately unwired (supports_gradient_checkpointing=
        # False), so write OUR gate — and only ever RAISE it (never clobber
        # a config-forced True down to False at shorter seq).
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        base = inner.model if hasattr(inner, "model") else inner
        # Force recompute-in-backward at 8192 only where it is needed: an 80 GB
        # card. The 2026-09-02 B200 sweep (roadmap §13b) fit 4 x 8192 at 144 GB
        # with checkpointing OFF, and checkpointing did not rescue 8 x 8192 —
        # the memory scales with tokens/micro-batch outside the blocks — so on
        # 96-192 GB cards it would only add the recompute.
        small_card = torch.cuda.is_available() and (
            torch.cuda.get_device_properties(0).total_memory < 100e9)
        if current_seq_len >= 8192 and small_card and not base._osrt_grad_ckpt:
            base._osrt_grad_ckpt = True
            print(
                f"    Gradient checkpointing: ENABLED for seq "
                f"{current_seq_len} (_osrt_grad_ckpt=True)",
                flush=True,
            )

        optimizer.zero_grad(set_to_none=True)
        accum_task_loss = torch.tensor(0.0, device=device)
        accum_balance_norm = torch.tensor(0.0, device=device)
        n_task_terms = 0   # micro-batches that actually contributed a task loss
        # Accumulate MoE telemetry across all grad_accum micro-batches so
        # the per-step metrics (and the 5k gate) average over the full
        # effective batch instead of reading only the last micro-batch.
        moe_snapshots: list[dict[str, float]] = []

        # Hoist the "do we need MoE telemetry this step?" decision out of
        # the micro-batch loop. _collect_moe_metrics does several .item()
        # CPU-GPU syncs per call; skipping it on non-logging steps saves
        # 18 (blocks × loops) × grad_accum syncs per skipped step
        # (review/performance-loop-audit P1).
        should_log_this_step = (
            step % train_cfg.log_interval == 0
            or step == 0
            or (step < 100 and step % 10 == 0)
        )
        is_early_stop_step = step == train_cfg.early_stop_check_step
        collect_moe_this_step = should_log_this_step or is_early_stop_step

        # Tell the model whether to bother computing the .item()/.tolist()
        # MoE telemetry inside its forward. Skipping it saves ~21 syncs ×
        # 18 effective MoE layers per micro-batch on non-logging steps.
        inner.set_moe_telemetry(collect_moe_this_step)

        if step == start_step:
            print("Fetching first batch...")
            batch_t = time.time()

        for micro in range(grad_accum):
            try:
                try:
                    input_ids, labels = next(loader_iter)
                except StopIteration:
                    _, p_cfg = get_phase(step, train_cfg)
                    if current_loader is not None:
                        del current_loader
                    current_loader = make_loader(
                        p_cfg["datasets"],
                        p_cfg["seq_len"],
                        tokenizer_name,
                        p_cfg.get("batch_size", train_cfg.batch_size),
                        step,
                        num_workers=getattr(train_cfg, "dataloader_num_workers", 0),
                    )
                    loader_iter = iter(current_loader)
                    input_ids, labels = next(loader_iter)
            except DataSourceDead as e:
                # Every source has failed permanently (revoked gate, schema
                # change, HF outage). Until 2026-09-30 the stream spun
                # forever here with no step, no checkpoint and no error. Save
                # the last COMPLETED step (this one has no optimizer step yet)
                # and exit with a status the launcher can act on.
                optimizer.zero_grad(set_to_none=True)
                print(f"\n>>> DATA SOURCES DEAD at step {step}: {e}", flush=True)
                if step > start_step:
                    _save(f"{ckpt_dir}/osrt_rescue_step_{step - 1}.pt", step - 1)
                if use_wandb:
                    wandb.finish()
                return "data_dead"

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if step == start_step and micro == 0:
                print(f"First batch fetched in {time.time() - batch_t:.1f}s")
                print("Running first forward pass (torch.compile tracing)...")

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / grad_accum

            loss.backward()
            # Pull separated components from the unwrapped model for clean
            # logging (total loss includes aux; these are the components).
            if inner.last_task_loss is not None:
                accum_task_loss += (
                    inner.last_task_loss.detach() / grad_accum
                )
                n_task_terms += 1
            if inner.last_balance_loss_normalised is not None:
                accum_balance_norm += (
                    inner.last_balance_loss_normalised.detach() / grad_accum
                )

            # Snapshot per-micro-batch MoE telemetry, but ONLY when this
            # step will actually consume it (logging or the early-stop
            # gate). On non-logging steps the snapshot is pure overhead:
            # each call does .item() reads that force CPU-GPU sync.
            if collect_moe_this_step:
                micro_metrics, _ = _collect_moe_metrics(model)
                moe_snapshots.append(micro_metrics)

        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_cfg.grad_clip,
        ))
        # §18.2: Muon's stats (RMS + Newton-Schulz orthogonality error) cost
        # host syncs and a matmul per param, so collect them only on the steps
        # that get logged. (Until 2026-09-30 this was keyed on step+1, so the
        # value was computed on step 49 and discarded before the log at 50.)
        _muon = getattr(optimizer, "muon", None)
        if _muon is not None:
            _muon.collect_ortho_error = should_log_this_step
        if math.isfinite(grad_norm):
            nonfinite_streak = 0
            optimizer.step()
            apply_router_balance_updates(model)
        else:
            # Skip the step: with a NaN/inf norm clip_grad_norm_ has scaled
            # every gradient by NaN and the update would poison the weights
            # (and the next checkpoint). Keep the weights, drop the routing
            # statistics of this forward, and count it.
            nonfinite_streak += 1
            nonfinite_total += 1
            optimizer.zero_grad(set_to_none=True)
            _reset_router_balance_accumulators(model)
            print(
                f"  [warn] non-finite grad norm ({grad_norm}) at step {step}: "
                f"optimizer step skipped ({nonfinite_streak}/{max_nonfinite} "
                f"consecutive, {nonfinite_total} total)",
                flush=True,
            )
            if nonfinite_streak >= max_nonfinite:
                print(
                    f"\n>>> EARLY STOP at step {step}: {nonfinite_streak} "
                    "consecutive non-finite gradient norms.",
                    flush=True,
                )
                _stop_failed("The optimisation has diverged.")
                early_stop_triggered = True
                run_status = "early_stop"
                break

        # Average snapshots once per step. Used for both logging and the
        # early-stop gate so both see the same grad-accum-averaged values.
        # Empty list on non-logging steps → empty dicts; downstream gates
        # never consume moe_metrics/moe_summary on those steps.
        if collect_moe_this_step:
            moe_metrics, moe_summary = _average_moe_snapshots(moe_snapshots)
        else:
            moe_metrics, moe_summary = {}, {}

        # --- Logging ---
        should_log = should_log_this_step
        if should_log:
            # The logged task loss is a sum over micro-batches of
            # last_task_loss / grad_accum, skipping micro-batches where the
            # attribute was unset. Trunk step 1050 (2026-09-02) logged 1.67
            # between 4.13 and 4.41 — the first log after run_eval, during a
            # Dynamo recompile-limit transition — i.e. a fraction of the terms.
            # Rescale to the terms that contributed and make it visible.
            if 0 < n_task_terms < grad_accum:
                accum_task_loss = accum_task_loss * (grad_accum / n_task_terms)
                print(
                    f"  [warn] task loss at step {step} averaged over only "
                    f"{n_task_terms}/{grad_accum} micro-batches (rescaled)",
                    flush=True,
                )
            elapsed = time.time() - start_time
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            eff_batch = current_batch_size * grad_accum
            steps_done = max(step - start_step, 1)
            tok_per_sec = eff_batch * current_seq_len / max(
                elapsed / steps_done, 1e-8,
            )

            # moe_metrics / moe_summary are already computed above as the
            # average over grad_accum micro-batches. No need to re-collect.

            # Sources the streaming loader has given up on (see
            # osrt.data.TokenStream): a change here means the realised mix no
            # longer matches the phase table. Loud, once per change.
            ds_obj = getattr(current_loader, "dataset", None)
            dead_sources = list(getattr(ds_obj, "dead_sources", None) or [])
            if dead_sources != dead_sources_seen:
                print(
                    f"\n>>> DATA SOURCES DROPPED (permanent failures): "
                    f"{dead_sources} — the realised mix now differs from the "
                    f"phase table. Investigate before trusting this phase.",
                    flush=True,
                )
                dead_sources_seen = dead_sources

            print(
                f"step {step:>7d}/{train_cfg.total_steps} | "
                f"task {accum_task_loss.item():.4f} | "
                f"bal {accum_balance_norm.item():.4f} | "
                f"lr {lr:.2e} | gnorm {grad_norm:.3f} | "
                f"gumbel {router_gumbel_tau:.3f} | "
                f"vram {vram_gb:.1f}GB | "
                f"tok/s {tok_per_sec:,.0f} | "
                f"phase {current_phase} | seq_len {current_seq_len}"
                + (f" | nonfinite {nonfinite_total}" if nonfinite_total else "")
                + (f" | dead_sources {len(dead_sources)}" if dead_sources else ""),
                flush=True,
            )
            print(
                f"           moe: "
                f"pte={moe_summary['per_token_H']:.3f} "
                f"marg={moe_summary['marginal_H']:.3f} "
                f"assn={moe_summary['assign_H']:.3f} "
                f"raw_max={moe_summary['raw_max']:.3f} "
                f"margin={moe_summary['top_margin']:.3f} "
                f"drop={moe_summary['drop_rate']:.4f} "
                f"gate={moe_summary['moe_gate']:.3f} "
                f"bias={moe_summary['bias_abs_max']:.3f} "
                f"emax={moe_summary['expert_max']:.3f} "
                f"emin={moe_summary['expert_min']:.3f} "
                f"bal={moe_summary['balance_loss']:.3f}",
                flush=True,
            )
            print(
                f"           clean: "
                f"pte={moe_summary['clean_per_token_H']:.3f} "
                f"marg={moe_summary['clean_marginal_H']:.3f} "
                f"raw_max={moe_summary['clean_raw_max']:.3f} "
                f"margin={moe_summary['clean_top_margin']:.3f} "
                f"emax={moe_summary['clean_expert_max']:.3f} "
                f"emin={moe_summary['clean_expert_min']:.3f}",
                flush=True,
            )
            print(
                f"           prebias: "
                f"pte={moe_summary['prebias_per_token_H']:.3f} "
                f"marg={moe_summary['prebias_marginal_H']:.3f} "
                f"raw_max={moe_summary['prebias_raw_max']:.3f} "
                f"margin={moe_summary['prebias_top_margin']:.3f} "
                f"emax={moe_summary['prebias_expert_max']:.3f} "
                f"emin={moe_summary['prebias_expert_min']:.3f}",
                flush=True,
            )
            # Recursive-loop collapse: per-effective-layer residual update
            # ||Δx||/||x||. A monotone decay to ~0 in the deep loops means they
            # have collapsed to no-ops. 'last' = deepest loop, 'dead' = experts
            # below 10% of uniform share across all blocks/loops.
            n_eff = model_config.num_blocks * model_config.recursive_loops
            loop_str = " ".join(
                f"L{i}={moe_metrics.get(f'loop/update_norm_l{i}', 0.0):.3f}"
                for i in range(n_eff)
            )
            print(
                f"           loop |dx|/|x|: {loop_str}",
                flush=True,
            )
            lu_min = moe_summary["loop_update_norm_min"]
            lu_last = moe_summary["loop_update_norm_last"]
            lu_mean = moe_summary["loop_update_norm_mean"]
            print(
                f"           collapse: loop_upd min={lu_min:.3f} "
                f"last={lu_last:.3f} "
                f"mean={lu_mean:.3f} | "
                f"hidden_norm_ratio={moe_summary['loop_hidden_norm_ratio']:.2f} | "
                f"dead_experts={int(moe_summary['dead_experts_total'])} | "
                f"bias_abs_max={moe_summary['bias_abs_max']:.3f}"
                + (f" | ortho_err={_muon.last_stats['muon/ortho_err']:.4f}"
                   if _muon is not None and "muon/ortho_err" in _muon.last_stats
                   else ""),
                flush=True,
            )

            if use_wandb:
                log_dict = {
                    "train/task_loss": accum_task_loss.item(),
                    "train/balance_loss_normalised": accum_balance_norm.item(),
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/nonfinite_steps": nonfinite_total,
                    "train/dead_sources": len(dead_sources),
                    "moe/gumbel_tau": router_gumbel_tau,
                    "train/vram_gb": vram_gb,
                    "train/tok_per_sec": tok_per_sec,
                    "train/phase": current_phase,
                    "train/seq_len": current_seq_len,
                }
                log_dict.update(moe_metrics)
                if _muon is not None and _muon.last_stats:
                    log_dict.update(_muon.last_stats)
                wandb.log(log_dict, step=step)

        elif step < 100:
            sys.stdout.write(".")
            sys.stdout.flush()
            if step % 25 == 24:
                sys.stdout.write(f" [step {step}]\n")
                sys.stdout.flush()


        # --- Health checks (router + recursion) ---
        # MUST run BEFORE the numbered checkpoint save on the same step.
        # Otherwise a failed run writes osrt_step_N.pt and a later launch
        # would resume past the gate and ignore the failure diagnosis.
        # On failure we save a DIFFERENT filename (osrt_failed_step_N.pt)
        # that the resume scanner explicitly ignores.
        #
        # The grad-accum-averaged summary exists on logging steps and at the
        # gate step; `_health_scope` decides what applies at this step (until
        # 2026-09-30 everything ran exactly once, at step 5,000, and a run
        # that collapsed at 6,000 trained to the end). The gate keeps its
        # one-shot semantics; every other check needs `health_check_patience`
        # consecutive failures so one noisy average cannot end a 45 h run.
        scope = _health_scope(step, train_cfg) if moe_summary else None
        if scope is not None:
            failures = _check_early_stop_criteria(
                step, moe_summary, train_cfg, model_config, scope=scope,
            )
            at_gate = step == train_cfg.early_stop_check_step
            patience = 1 if at_gate else max(
                1, getattr(train_cfg, "health_check_patience", 3))
            if failures:
                keys = _update_health_streaks(health_fail_streaks, failures)
                worst = max(health_fail_streaks[k] for k in keys)
                print(
                    f"\n>>> HEALTH CHECK at step {step} ({scope}): "
                    f"{len(failures)} criteria failing "
                    f"[longest streak {worst}/{patience} consecutive]:",
                    flush=True,
                )
                for f in failures:
                    print(f"      - {f}  "
                          f"[{health_fail_streaks[_failure_key(f)]}/{patience}]")
                if worst >= patience:
                    print(f"\n>>> EARLY STOP at step {step}: health criteria "
                          "failed.", flush=True)
                    _stop_failed(
                        "The architecture bets are not paying off on this run."
                        if at_gate else
                        "The run has drifted into collapse after the gate.")
                    early_stop_triggered = True
                    run_status = "early_stop"
                    break
            else:
                if health_fail_streaks:
                    print(f"\n>>> health check at step {step}: recovered "
                          f"({', '.join(sorted(health_fail_streaks))} no longer "
                          "failing).", flush=True)
                health_fail_streaks.clear()
                if at_gate:
                    print(
                        f"\n>>> Router health gate at step {step}: "
                        f"all criteria PASS. Continuing training "
                        f"(re-checked every {train_cfg.log_interval} steps).",
                        flush=True,
                    )

        # --- Checkpoints (numbered, resumable) ---
        # Runs AFTER the health check so failed runs never produce a
        # step_N.pt that would bypass the gate on resume.
        if step > 0 and step % train_cfg.ckpt_interval == 0:
            _save(f"{ckpt_dir}/osrt_step_{step}.pt", step)

        # --- Eval on held-out FineWeb-Edu --- AFTER the checkpoint save, so a
        # slow eval can never again cost a checkpoint (2026-09-02 trunk).
        # Fixed context/batch so the cached set is the same in every phase.
        if step > 0 and step % train_cfg.eval_interval == 0:
            eval_metrics = run_eval(
                model, tokenizer_name,
                getattr(train_cfg, "eval_seq_len", current_seq_len),
                getattr(train_cfg, "eval_batch_size", current_batch_size),
                train_cfg.eval_steps, device,
            )
            print(
                f"  EVAL step {step} | "
                f"loss {eval_metrics['eval/loss']:.4f} | "
                f"ppl {eval_metrics['eval/perplexity']:.1f}",
                flush=True,
            )
            if use_wandb:
                wandb.log(eval_metrics, step=step)
        # --- 23h Modal safety (rescue checkpoint + clean exit) ---
        # Rescue filename includes the step so resume scanner can rank it
        # against numbered checkpoints.
        if time.time() - start_time > 82_800:
            _save(f"{ckpt_dir}/osrt_rescue_step_{step}.pt", step)
            print(
                f"\n23h boundary reached at step {step}. "
                f"Rescue checkpoint saved; exiting cleanly for resume.",
                flush=True,
            )
            if use_wandb:
                wandb.finish()
            return "rescued"

        step += 1

    # Final checkpoint (full run completed or early stopped)
    if not early_stop_triggered:
        run_status = "complete"
        elapsed_total = time.time() - start_time
        print(
            f"\nPretrain complete. {step:,} steps in "
            f"{elapsed_total / 3600:.1f}h",
            flush=True,
        )
        # Gate the final save: sanity/mem/compile checks set
        # save_final_checkpoint=False so they don't write a throwaway
        # osrt_final.pt that would clobber a real run's final on the volume.
        if getattr(train_cfg, "save_final_checkpoint", True):
            final_path = f"{ckpt_dir}/osrt_final.pt"
            _save(final_path, step)
            # Step-numbered alias so the resume scan / HF sync recognise a
            # finished run instead of re-training its last interval.
            _alias_checkpoint(final_path, f"{ckpt_dir}/osrt_step_{step}.pt")
            vol.commit()
            print(f"Final checkpoint: {final_path} "
                  f"(alias osrt_step_{step}.pt)", flush=True)
        else:
            print(
                "Final checkpoint save skipped (save_final_checkpoint=False).",
                flush=True,
            )
    if use_wandb:
        wandb.finish()
    return run_status or "early_stop"


# ============================================================================
# CONTINUED PRE-TRAINING ("EXTEND" / MID-TRAINING)
# ============================================================================
