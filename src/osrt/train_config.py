"""Training configurations for OSRT.

v7 STATUS — provisional.

This holds only PretrainConfig. The 31 v6 stage classes that used to live here
(PretrainExtend x3, LoopFix x2, MOPD, SystemSFT, Midtrain x6, SFT v1-v4 / Long /
Refresh / Math, GRPO x4) were removed: they encode v6 checkpoints, the v6
tokenizer and a superseded pipeline, and nothing in the tree imported them. They
remain in git history and in CodeHalwell/OSRT-605M-A269M.

The values below are inherited from v6 and are NOT yet re-derived for v7. Three
things must change before a trunk run, and all three are gated:

  * the data mix and token budget, once G3a settles whether the token
    requirement tracks active or total parameters;
  * the schedule — the roadmap adopts WSD trunk-and-branch over cosine (§7.1);
  * batch/seq economics under the new tokenizer, which is ~6% more tokens per
    character on the real mix (§16.4).

Post-training configs are deliberately absent. v7's post-training is redesigned
around a verifier that does not exist yet (roadmap §7.6, §12); the v6 GRPO
reward is on record as having made the model measurably worse.


v5 architecture: Mixtral-style MoE (8 routed × top-2, 1 shared, no dense FFN),
Switch balance loss, orthogonal expert init, eval-time drop-free capacity.

Progressive curriculum:
  Phase 1 (Foundation):  seq_len 2048, FineWeb-Edu + CodeParrot
  Phase 2 (Knowledge):   seq_len 4096, FineWeb-Edu + CodeParrot + Wikipedia
  Phase 3 (Instruction): seq_len 8192, SmolTalk + Evol-Code + OpenHermes

Post-training:
  SFT:  Balanced math + code + STEM + general (native tag format)
  GRPO: Verifiable math rewards


# Optimizer × routing ablation (Foundation-matched cells)
# ──────────────────────────────────────────────────────────────────────

Ran A/B/C to 1200 steps in `--stage ablate`; stopped D at step 600
once its task loss remained tied with C but the raw router had already
collapsed. Headline numbers:

| Cell | Optimizer | Aux  | Best / seen task | prebias emin | bal  |
|------|-----------|-----:|-----------------:|-------------:|-----:|
| A    | Lion      | 0.10 |             ~7.0 |       ~0.002 | ~1.2 |
| B    | Lion      | 0.0  |             ~7.6 |        0.000 | ~3.9 |
| C    | Muon      | 0.10 |         **3.43** |    **0.105** | 1.02 |
| D    | Muon      | 0.0  |   4.66 @ step 600|       ~0.001 | ~2.3 |

Three load-bearing conclusions that drive the v5 defaults:

1. Muon is a ~4-nat task-loss win at this scale, regardless of routing
   scheme. C and D both hit task < 5.0 by step 450; A and B were still
   at ~7.2 there.
2. Gradient aux loss is necessary for router health regardless of
   optimizer — bias controller alone collapses the raw router under
   both Lion and Muon. The DeepSeek-V3 "auxiliary-loss-free" claim
   does not hold at 363M params on this curriculum.
3. C (Muon + aux) is the production recipe: best loss, best balance,
   best emin, best margin. D gets C's loss but with B-style collapse
   on the raw router, which would degrade once task complexity grows
   beyond what 2-3 active experts can fit (Phases 2/3).

Defaults below reflect this. To rerun the ablation, use --stage ablate.


# Optional A/B configurations
# ──────────────────────────────────────────────────────────────────────

DeepSeek-style aux-loss-free routing (research only; failed here)
-----------------------------------------------------------------
v5 ships with both a gradient-driven Switch balance aux loss
(coefficient `router_aux_loss_coeff`, default 0.10) and a
non-gradient per-loop bias controller (`router_balance_bias_*`).
DeepSeek-V3 reports that the bias controller alone is sufficient on
their 671B model, and that removing the gradient aux loss eliminates
the well-known specialisation-vs-balance tradeoff (the gradient term
forces uniformity even when token-context says it shouldn't).

The v5 ablation rejected that recipe at 363M params: both bias-only
cells collapsed the raw pre-bias router. Cell D proved why clean
metrics are not enough: Muon kept task loss near Cell C through step
600, but prebias emin fell to ~0.001 and the model was effectively
using only a small expert subset. Treat aux-loss-free runs as research
or failure-reproduction only unless the routing algorithm itself
changes.

To reproduce aux-loss-free routing on this codebase, override the model
config when constructing it:

    model_config = OSRTConfig(
        router_aux_loss_coeff=0.0,        # disable gradient aux
        router_balance_bias_enabled=True, # keep bias controller
        router_balance_bias_update_rate=0.10,
        ...,
    )

The clean four-metric Phase-1 health gate is not sufficient for this
experiment because the bias controller can hide collapse. Watch the
raw metrics instead:
`moe/prebias_marginal_entropy_mean`,
`moe/prebias_expert_min_mean`,
`moe/prebias_raw_max_prob_mean`, and
`moe/prebias_top_margin_mean`.
If `moe/prebias_expert_min_mean` falls near zero while task loss still
looks good, the recipe has failed even if clean balance appears healthy.

Sweep template (drop into app.py near the existing `sweep` stage)::

    sweep_configs = [
        {"name": "aux_only",  "aux": 0.10, "bias": True},   # current default
        {"name": "bias_only", "aux": 0.0,  "bias": True},   # DeepSeek-style
        {"name": "both_low",  "aux": 0.03, "bias": True},   # belt + braces
    ]
"""


class PretrainConfig:
    """Pre-training hyperparameters for v5."""

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 8
    # Cosine horizon sized to the base-pretrain budget (~$100 on Modal H100,
    # $3.95/hr ≈ 25 H100-hr ≈ ~3,500 steps at ~5k tok/s, seq-2048 foundation
    # @ 131K tok/step ≈ ~455M tokens). total_steps is the LR-anneal target, and
    # the step counter persists across resumes, so chunked runs toward a FIXED
    # total_steps are one continuous cosine (no re-warm between chunks). The
    # cosine fully decays peak→min_lr by step 3,500, so the run self-terminates
    # at the budget with a clean, annealed base. Long-context (4096/8192) and
    # math specialisation happen in the SEPARATE mid-training/extend stages,
    # which re-warm from this checkpoint — so annealing the base to min here is
    # correct. To train a longer base, raise this BEFORE the first chunk and
    # keep it fixed across resumes (changing it mid-run reshapes the cosine).
    # LR schedule. "cosine" is v6's; "wsd" is warmup-stable-decay
    # (trunk-and-branch), adopted for v7 by roadmap item 0.2 and now the
    # unanimous 2026 practice (Nemotron two-phase WSD, Kimi Linear).
    #
    # Why it matters here specifically: this project trains in ~$120/month
    # drip chunks. Under cosine, every extension either re-warms (paying the
    # tax again) or reshapes the curve mid-run — §2.2 records the token-budget
    # arithmetic as the plan's weakest link, and re-warm waste comes straight
    # off it. WSD holds LR flat through the trunk so a run can be stopped and
    # resumed at no cost, and decays only on the branch that produces a
    # release checkpoint.
    # DataLoader workers for the streaming corpus. 0 is correct on Colab:
    # HF streaming + BPE inside forked workers leaks semaphores and dies with
    # "Bad file descriptor" once workers x streams gets large, and a dead
    # worker on a session-capped runtime costs the whole session. On a
    # dedicated box raise it.
    dataloader_num_workers: int = 0
    lr_schedule: str = "wsd"
    # Fraction of total_steps spent in the final decay ramp. 0.2 is the
    # common choice; the stable phase is everything between warmup and it.
    wsd_decay_frac: float = 0.2
    total_steps: int = 3_500
    warmup_steps: int = 400          # ~11% — spins up Muon + the MoE balance bias
    peak_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.3
    grad_clip: float = 1.0
    log_interval: int = 50
    eval_interval: int = 1_000
    eval_steps: int = 20           # number of batches per eval
    # Frequent ckpts protect against budget-driven Modal kills: with a
    # capped credit pool, the function dies hard (no clean shutdown,
    # no rescue ckpt) when the wallet hits zero. 500-step intervals
    # bound progress loss to ~30 min on H100 at this throughput.
    ckpt_interval: int = 500
    # Write the final checkpoint at the end of a completed run. Real runs leave
    # this True; sanity/mem/compile checks set it False so they don't clobber
    # a real run's final checkpoint on the shared volume with throwaway weights.
    save_final_checkpoint: bool = True
    # Default optimizer is Muon hybrid (Muon for 2D matrix weights,
    # AdamW for embeddings/norms/scalars/router/loop_embeddings). The
    # 1200-step ablation (A/B/C to completion; D stopped at step 600)
    # showed Muon+aux delivers a
    # ~4-nat task-loss improvement over Lion+aux at step 1200 (cell C
    # task ~3.4 vs cell A task ~7.4) AND keeps the learned pre-bias
    # routed-expert population balanced (prebias emin > 0.10 vs cell A's
    # late-warmup collapse to emin < 0.01). Lion is still available
    # via optimizer_name="lion" for comparison runs. AdamW is the
    # fallback when optimizer_name is anything else.
    optimizer_name: str = "muon"
    # Muon LR (used only when optimizer_name == "muon"). The Newton-Schulz
    # update is normalised, so Muon's effective step size is much smaller
    # per parameter than Lion/AdamW. The 1200-step Cell C run held
    # task-loss steady at lr=0.02 through 23 % of warmup with no fatal
    # divergence. If a full Phase 1 (10k steps, peak at step 3000)
    # destabilises, drop to 0.015 first — that's the next thing to
    # try before deeper changes. AdamW (the other half of the hybrid)
    # keeps using peak_lr / min_lr.
    muon_lr: float = 0.02
    muon_min_lr: float = 2e-3
    # Per-head Muon (Kimi K3 §2.5): orthogonalise each attention head's block
    # of q_proj / kv_down / v_from_k separately instead of the full matrix, so
    # no single head dominates the shared update. Adds no params and the update
    # magnitude is held constant, so it's a clean A/B toggle. Default off.
    per_head_muon: bool = False

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "osrt"
    # Suffix the optimizer in the W&B name so dashboard runs from
    # different optimizer configs don't visually pile up on top of the
    # historical Lion runs. Override per-run if you want a custom label.
    wandb_run_name: str = "osrt-v7-pretrain"
    wandb_run_id: str = ""

    # --- Success criteria for Phase 1 (Foundation) ---
    # v4's router was never alive. v5 uses a four-metric check that would
    # have correctly diagnosed v4's failure (batch-marginal entropy stayed
    # high while per-token entropy also stayed high). All four must hold by
    # step `early_stop_check_step` or training is considered failed.
    early_stop_check_step: int = 5_000
    min_per_token_entropy_drop: float = 0.55   # init 2.08 → 1.53 (ln 8 = 2.08)
    min_raw_max_prob: float = 0.30             # well above uniform 1/8 = 0.125
    min_top_margin: float = 0.10               # clear gap between rank 0 and 1
    min_marginal_entropy: float = 1.80         # balanced across experts
    # The four checks above use clean deployed routing (router + balance bias).
    # These guardrails make sure the learned pre-bias router is not secretly
    # collapsed while the non-gradient bias controller hides it.
    min_prebias_marginal_entropy: float = 1.55
    min_prebias_expert_fraction: float = 0.01
    max_bias_saturation_fraction: float = 0.85

    # --- Router exploration ---
    # Sanity runs showed experts can die during the first 20 optimizer steps:
    # once an expert falls out of top-k it receives no task gradient. Add
    # noisy top-k exploration early. Anneal after LR warmup peaks so the
    # router does not face a rising task-loss gradient with no exploration,
    # but finish 1k steps before the 5k clean-router health gate.
    router_gumbel_tau_init: float = 0.5
    router_gumbel_tau_final: float = 0.0
    router_gumbel_anneal_steps: int = 4_000

    # Progressive seq_len curriculum
    # Tokens per step per phase:
    #   Phase 1 (foundation, 10K steps):  8 × 8  × 2048 = 131K tok/step → ~1.3B
    #   Phase 2 (knowledge, 240K steps):  4 × 16 × 4096 = 262K tok/step → ~63B
    #   Phase 3 (instruction, 50K steps): 2 × 32 × 8192 = 524K tok/step → ~26B
    # Total budget: ~90B tokens if the full 300K schedule completes.
    phases: dict = {  # noqa: RUF012
        "foundation": {
            "start": 0,
            "end": 9_500,
            "seq_len": 2048,
            "grad_accum_steps": 8,
            "datasets": [
                {
                    "name": "fineweb-edu",
                    "hf_id": "HuggingFaceFW/fineweb-edu",
                    "weight": 0.40,
                },
                {
                    "name": "nemotron-cc-math-4plus",
                    "hf_id": "nvidia/Nemotron-CC-Math-v1",
                    "hf_config": "4plus",
                    "weight": 0.25,
                },
                {
                    "name": "nemotron-code-syn-qa",
                    "hf_id": "nvidia/Nemotron-Pretraining-Code-v2",
                    "hf_config": "Synthetic-Question-Answering",
                    "weight": 0.20,
                },
                {
                    "name": "cosmopedia-web",
                    "hf_id": "HuggingFaceTB/cosmopedia",
                    "hf_config": "web_samples_v2",
                    "weight": 0.15,
                },
            ],
        },
        "knowledge": {
            "start": 9_500,
            "end": 250_000,
            "seq_len": 4096,
            # Bumped from batch_size=4, grad_accum_steps=16 once the
            # grad-checkpointing threshold was raised (see train.py
            # commit 57513a9). At seq_len 4096 with no checkpointing,
            # H100 80GB had ~31 GB unused at batch 4 (49 GB total).
            # Batch 8 OOMed at 76.7 GB (activations scale super-linearly
            # at this sequence length); batch 6 sits at ~59 GB, leaving
            # ~20 GB headroom for fragmentation and the optimizer step's
            # transient buffers. Effective batch 6*11=66 sequences,
            # close to the prior 4*16=64. If a future GPU has tighter
            # VRAM (3090, A100 40GB), drop batch_size back to 4 with
            # grad_accum_steps=16.
            "batch_size": 6,
            "grad_accum_steps": 11,
            "datasets": [
                {
                    "name": "nemotron-cc-math-4plus",
                    "hf_id": "nvidia/Nemotron-CC-Math-v1",
                    "hf_config": "4plus",
                    "weight": 0.25,
                },
                {
                    "name": "nemotron-stem",
                    "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                    "hf_config": "Nemotron-Pretraining-STEM-SFT",
                    "weight": 0.15,
                },
                {
                    "name": "nemotron-math-textbooks",
                    "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                    "hf_config": "Nemotron-Pretraining-Math-Textbooks",
                    "weight": 0.15,
                },
                {
                    "name": "nemotron-reasoning",
                    "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                    "hf_config": "Nemotron-Pretraining-InfiniByte-Reasoning",
                    "weight": 0.10,
                },
                {
                    "name": "fineweb-edu",
                    "hf_id": "HuggingFaceFW/fineweb-edu",
                    "weight": 0.15,
                },
                {
                    "name": "nemotron-code-syn-qa",
                    "hf_id": "nvidia/Nemotron-Pretraining-Code-v2",
                    "hf_config": "Synthetic-Question-Answering",
                    "weight": 0.10,
                },
                {
                    "name": "cosmopedia-openstax",
                    "hf_id": "HuggingFaceTB/cosmopedia",
                    "hf_config": "openstax",
                    "weight": 0.10,
                },
            ],
        },
        "instruction": {
            "start": 250_000,
            "end": 300_000,
            "seq_len": 8192,
            "batch_size": 2,
            "grad_accum_steps": 32,
            "datasets": [
                {
                    "name": "smoltalk",
                    "hf_id": "HuggingFaceTB/smoltalk",
                    "hf_config": "all",
                    "weight": 0.30,
                },
                {
                    "name": "evol-instruct-code",
                    "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
                    "weight": 0.20,
                },
                {
                    "name": "openhermes",
                    "hf_id": "teknium/OpenHermes-2.5",
                    "weight": 0.10,
                },
                {
                    "name": "nemotron-post-training-math",
                    "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
                    "split": "math",
                    "weight": 0.20,
                },
                {
                    "name": "nemotron-post-training-stem",
                    "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
                    "split": "stem",
                    "weight": 0.20,
                },
            ],
        },
    }

    # Budget note: the schedule is aspirational — the user runs in chunks as
    # Modal credits allow. Checkpoints every 1K steps keep stop/resume cheap.
    # Any early stopping still leaves a usable model for SFT.
