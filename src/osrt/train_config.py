"""Training configurations for OSRT.

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
    # Write osrt_v5_final.pt at the end of a completed run. Real runs leave
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
    wandb_run_name: str = "osrt-pretrain-muon"
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


class PretrainExtendConfig(PretrainConfig):
    """Continued pre-training ("mid-training") on top of an SFT checkpoint.

    Goal: fill the pretrain data-mix gaps identified after SFT —
    specifically math (Nemotron-CC-Math), better code (The Stack v2),
    and scientific text (RedPajama-arxiv). The original pretrain ran
    on FineWeb-Edu + CodeParrot + Wikipedia only, with effectively no
    math content; the result was a model that emits structurally
    correct chat output but can't do double-digit multiplication.

    Resume strategy
    ───────────────
    Loads from `osrt_v5_sft_ultralong_final.pt` rather than the pure
    pretrain ckpt (osrt_v5_step_17000.pt). This keeps the SFT
    investment intact so we don't have to redo SFT from scratch
    afterward. The risks (chat-format erosion, HRA drift, full-token
    loss vs prompt-masked) are mitigated by:

      1. Conservative LR: peak 1.5e-5 (2.5 % of original 6e-4).
      2. SFT-formatted rehearsal data (25 % of mix) — Nemotron rows
         wrapped in <|user|>...<|assistant|><|think|>...<|/think|>
         <|answer|>...<|/answer|> so the model keeps seeing chat
         tags during ostensibly "raw" pretraining.
      3. HRA frozen (`hra_frozen=True`) — the 86 M HRA delta layer
         stays as the SFT-trained delta; only base weights absorb new
         pretrain knowledge. Cleanly separates concerns and gives a
         small ~5–8 % throughput win from skipping HRA backward pass.

    Token budget
    ────────────
    $30 of H100 time (~7.6 hr) at Phase-2-ish throughput
    (~15 sec/step at seq 4096) ≈ 1,800 steps × 270 k tok/step ≈
    485 M new tokens. About 15 % of our prior 3.27 B pretrain budget,
    concentrated in the underrepresented categories.

    Lineage
    ───────
    Output checkpoint: osrt_v5_extend_step_N.pt and
    osrt_v5_extend_final.pt (distinct prefix so resume scans don't
    collide with base pretrain checkpoints). Subsequent SFT
    "refresh" pass (200 steps, ~$4) loads from the extend-final to
    re-anchor chat format if it has degraded.
    """

    # ── Schedule ─────────────────────────────────────────────────────
    # Extended from 1,800 → 2,800 mid-run (post-step 200) to use more
    # of the $30 workspace budget on actual training (Liquid AI / Phi
    # philosophy: small models benefit from heavy overtraining past
    # Chinchilla-optimal). At our 8.4 sec/step throughput, 2,800
    # steps fits ~$26 of compute, leaving ~$4 for a limit-100 eval
    # pass to measure the lift. The cosine schedule recalibrates
    # automatically — at step 200 with new total=2800, LR is at
    # ~99.7 % of peak (vs ~99 % under the old total=1800), so the
    # transition is a tiny upward LR bump (0.5 %), then cosine cools
    # more gently over the longer horizon.
    total_steps: int = 2_800
    warmup_steps: int = 50          # 3 % of original — kept fixed (re-warmup
                                    # already done in steps 0-50)
    peak_lr: float = 1.5e-5         # 2.5 % of original 6e-4
    min_lr: float = 1.5e-6          # cosine to 10 % of peak
    weight_decay: float = 0.1       # softer wd than pretrain (0.3)
    grad_clip: float = 1.0
    log_interval: int = 25
    eval_interval: int = 9_999_999  # skip in-run eval (heartbeat risk; see extend2)
    eval_steps: int = 20
    ckpt_interval: int = 200        # ~14 ckpts over the 2,800-step run

    # Optimizer reuses the Muon hybrid from pretrain. The lower
    # peak_lr also propagates down to Muon via the same _peak_lr
    # tagging in train.py::_set_param_group_lrs. Override muon_lr
    # explicitly so we don't reuse the pretrain-tuned 0.02 (which
    # would shock SFT-flavored weights at this stage).
    optimizer_name: str = "muon"
    muon_lr: float = 5e-3           # 25 % of pretrain's 0.02
    muon_min_lr: float = 5e-4

    # ── Routing exploration ─────────────────────────────────────────
    # Disable Gumbel exploration entirely. The router has been trained
    # for 17k pretrain + 2.7k SFT steps and is well-formed; reintroducing
    # noise would hurt rather than help.
    router_gumbel_tau_init: float = 0.0
    router_gumbel_tau_final: float = 0.0
    router_gumbel_anneal_steps: int = 1   # avoid div-by-zero

    # ── Early-stop gate ────────────────────────────────────────────
    # Push past the budget so the gate never trips — the four-metric
    # gate was tuned for cold-start pretraining where the router needs
    # to specialise from scratch. At extend time the router is already
    # healthy (clean_per_token_H ~1.40, assn ~2.07 in last SFT) and the
    # gate's thresholds (designed for the "is this run salvageable?"
    # question) don't apply.
    early_stop_check_step: int = 9_999_999

    # ── HRA ────────────────────────────────────────────────────────
    # SFT-ultralong ckpt has HRA params (rank 256, +86.1M) in its
    # state_dict — must inject before load. `hra_frozen=True` is a
    # new flag (see train.py::run_pretrain_extend) that sets
    # requires_grad=False on adapters_a/adapters_b after load so the
    # frozen SFT-trained delta layer stays as-is while base absorbs
    # new pretrain content.
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_scale: float = 1.0
    hra_before_load: bool = True
    hra_frozen: bool = True
    pretrained_checkpoint: str = (
        "/vol/checkpoints/v5/osrt_v5_sft_ultralong_final.pt"
    )

    # Distinct ckpt prefix so this stage's checkpoints
    # (osrt_v5_extend_step_N.pt) don't collide with base pretrain
    # ckpts (osrt_v5_step_N.pt) under the resume scan.
    stage_prefix: str = "extend"

    # W&B labels
    wandb_run_name: str = "osrt-pretrain-extend"
    wandb_run_id: str = ""

    # ── Data mix (single phase, seq 4096) ──────────────────────────
    # Three new datasets + two existing for general-capability
    # maintenance + two SFT-formatted rehearsal streams.
    #
    # Token-weighted sampling (debt-based, see TokenStream._pick_stream)
    # produces actual token mixes matching these weights regardless of
    # per-stream example length. Weights need not sum to exactly 1; the
    # sampler normalises.
    phases: dict = {  # noqa: RUF012
        "extend": {
            "start": 0,
            "end": 1_800,
            "seq_len": 4096,
            # Phase 2 sizing — known to fit comfortably on H100 80GB
            # at ~60 GB VRAM with the bump from 4×16 to 6×11.
            "batch_size": 6,
            "grad_accum_steps": 11,
            "datasets": [
                # ── Math (35 %) — biggest gap, biggest expected lift
                {
                    "name": "nemotron-cc-math",
                    "hf_id": "nvidia/Nemotron-CC-Math-v1",
                    # `4plus` subset (FineMath-classifier ≥4) is the
                    # higher-quality 52B-token variant. Quality > quantity
                    # for our 243M-token math budget. Available subsets
                    # are: '3' (133B, broader), '4plus' (52B, higher
                    # quality), '4plus_MIND' (most curated). REQUIRES
                    # gated-access approval at
                    # https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1
                    "hf_config": "4plus",
                    "weight": 0.35,
                    "format": "nemotron_math",
                },
                # ── Math/scientific web text (12 %)
                # OpenWebMath replaces the original RedPajama-arxiv plan
                # because RedPajama-Data-1T uses a deprecated Python
                # loader script that modern HF datasets no longer
                # supports ("Dataset scripts are no longer supported").
                # OpenWebMath is the well-known 14.7B-token math web
                # corpus that Nemotron-CC-Math itself is positioned
                # against — provides math/science diversity beyond
                # Nemotron's curation pipeline.
                {
                    "name": "open-web-math",
                    "hf_id": "open-web-math/open-web-math",
                    "weight": 0.12,
                    "format": "arxiv",  # same `text` field shape
                },
                # ── Code (12 %) — CodeParrot
                # Originally planned bigcode/the-stack-smol but it is
                # gated. CodeParrot-Clean is the same dataset already
                # used in original pretrain so we know it streams
                # reliably; the goal here is to maintain code
                # capability under the new mix, not introduce a
                # different code distribution.
                {
                    "name": "codeparrot-clean",
                    "hf_id": "codeparrot/codeparrot-clean",
                    "weight": 0.12,
                    # Default extractor handles the `content` field
                    # natively (see TokenStream._extract_text).
                },
                # ── General-capability maintenance (16 %)
                {
                    "name": "fineweb-edu",
                    "hf_id": "HuggingFaceFW/fineweb-edu",
                    "weight": 0.08,
                },
                {
                    "name": "wikipedia",
                    "hf_id": "wikimedia/wikipedia",
                    "hf_config": "20231101.en",
                    "weight": 0.08,
                },
                # ── SFT-formatted rehearsal (25 %) — anti-forgetting
                # wraps Nemotron rows in <|user|>...<|/answer|> chat
                # schema before tokenisation. Pretrain loss is full-
                # token (no masking) so the model trains on every
                # token in the formatted string including chat tags.
                {
                    "name": "nemotron-math-rehearsal",
                    "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
                    "split": "math",
                    "weight": 0.15,
                    "format": "nemotron_sft",
                },
                {
                    "name": "nemotron-stem-rehearsal",
                    "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
                    "split": "stem",
                    "weight": 0.10,
                    "format": "nemotron_sft",
                },
            ],
        },
    }


class PretrainExtend2Config(PretrainExtendConfig):
    """Second mid-training pass — broadened reasoning + code + science.

    Built after the GRPO step-700 probe showed the model handles
    single-digit arithmetic but fails on two-digit subtraction,
    multiplication, and order-of-ops. The diagnosis: GRPO can only
    optimize capabilities the base model already has. The fix is
    more pretraining on reasoning-dense data, NOT more RL.

    Strategy
    ────────
    Follows the DeepSeek-R1 cold-start playbook: inject high-density
    `<think>` traces from real R1 distillations (OpenR1-Math-220k,
    OpenMathReasoning, OpenThoughts-114k) alongside evolved code
    (Magicoder), pure logic (BBH), and general-capability anchors
    (UltraChat, cosmopedia-v2, fineweb-edu).

    Lineage
    ───────
    Resume from `osrt_v5_grpo_final.pt` (canonical step-700 GRPO ckpt)
    with HRA frozen. The 86M HRA delta holds the GRPO-tuned answer
    format + chat tags learned through SFT + RL; only the 363M base
    weights absorb new pretrain knowledge. After this stage, a short
    sft_refresh (200 steps) re-anchors chat format if it has drifted.

    Mix (matches 30/40/15/15 target per user spec):
      Code 30%       — Magicoder-Evol-Instruct (20) + starcoderdata (10)
      Math/Sci 40%   — OpenR1-Math (15) + OpenMathReasoning (15) +
                       open-web-math (10)
      Reasoning 15%  — OpenThoughts (10) + BBH (5)
      General 15%    — UltraChat (8) + cosmopedia-v2 (4) + fineweb-edu (3)

    Cost
    ────
    ~3,000 steps × 8.4 sec/step × $4/hr H100 ≈ $28. Yields ~100M
    new training tokens at the configured batch/seq, concentrated
    in reasoning + code categories where the probe identified gaps.

    Tag rewrite
    ───────────
    R1-style sources emit `<think>...</think>` (HTML) while our
    tokenizer's special tokens are `<|think|>...<|/think|>` (pipe).
    Each cold-start format function in data.py rewrites these tags
    so the model trains on its own format consistently — without the
    rewrite, the inner reasoning blocks would tokenise as raw BPE
    and create a parallel non-canonical reasoning format.
    """

    # ── Schedule ─────────────────────────────────────────────────────
    # 3,000 steps at seq 2048. Shorter seq than extend1's 4096 because
    # most R1 reasoning traces fit comfortably in 1,500-2,000 tokens
    # (problem + think + answer); seq 4096 would waste compute on
    # padding. peak_lr 1e-5 is one-third below extend1's 1.5e-5 — we
    # are starting from a GRPO-tuned policy that has converged tighter
    # than the SFT-ultralong base extend1 resumed from, so the
    # gradient step size needs proportionally smaller to avoid
    # overshooting the carefully-found GRPO optimum.
    # Phase progression (lr_anchor_step makes each "phase" a fresh
    # cosine from re-warm → peak → cool, layered onto the same model):
    #   Phase 1: steps    0 → 3000  (peak 1e-5, sft_math base)
    #   Phase 2: steps 3000 → 5600  (peak 1e-5 re-warm; cut short at 5600)
    #   Phase 3: steps 5600 → 8100  (peak 7e-6, tight consolidation cool)
    # Phase 3 finishes mid-training with a fresh cosine: lower peak
    # (7e-6 vs phase 1/2's 1e-5) for consolidation rather than
    # exploration, deeper cool (min_lr 7e-7 vs 1e-6) for a tight
    # final state. ~2500 steps × ~1.5 sec compiled = ~62 min, ~$4.
    total_steps: int = 8_100
    lr_anchor_step: int = 5_600     # resume point — phase 3 anchors here
    warmup_steps: int = 60          # re-warm length over steps 5600-5660
    peak_lr: float = 7e-6           # cooler peak for consolidation
    min_lr: float = 7e-7            # 10% of peak — tight final cool

    # Muon hybrid mirrors extend1; lower peak_lr propagates via the
    # same _peak_lr tagging in train.py.
    muon_lr: float = 3e-3
    muon_min_lr: float = 3e-4

    # log_interval=25 — once .spawn() is the launch mechanism (the
    # cancellation we saw on .remote() was caller-disconnect related,
    # not heartbeat related), we don't need step-by-step logging.
    # 25-step interval is sane for a 3000-step run (~120 step events
    # total instead of 3000), keeps wandb/console output manageable.
    log_interval: int = 25
    ckpt_interval: int = 200        # ~15 ckpts over 3,000 steps
    eval_interval: int = 9_999_999  # skip in-run eval (heartbeat risk)
    eval_steps: int = 20

    # ── Resume ───────────────────────────────────────────────────────
    # Points at the extend2 step_5600 ckpt — the resume target for
    # phase 3 (consolidation cosine). run_pretrain_extend's startup
    # check requires `pretrained_checkpoint` to exist on the volume
    # regardless of whether a step-named ckpt also exists; pointing
    # it at step_5600 keeps the check happy AND matches what the
    # resume scan would load. Migrated from gradio-winter-hack.
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_extend2_step_5600.pt"
    hra_frozen: bool = True
    hra_before_load: bool = True

    # Distinct ckpt prefix — `extend2` keeps resume scans from
    # crossing with extend1's `extend_step_N.pt`.
    stage_prefix: str = "extend2"
    wandb_run_name: str = "osrt-pretrain-extend2"
    wandb_run_id: str = ""

    # DataLoader sizing: lowered from 4 → 1 because extend2's 9-stream
    # mix at 4 workers triggered "Connection reset by peer" + "Bad file
    # descriptor" cascades from HF Hub. 4 × 9 = 36 simultaneous HF
    # streaming connections is over what HF's load balancer tolerates
    # for a single client. 1 × 9 = 9 connections is safely under.
    dataloader_num_workers: int = 1
    dataloader_prefetch_factor: int = 2

    # torch.compile re-enabled: earlier disable was based on a wrong
    # hypothesis — the actual cause of the ~2-min cancellations was
    # .remote() inside a local_entrypoint losing the caller. .spawn()
    # fixed it (see feedback_modal_spawn_for_long_tasks memory).
    # With .spawn() handling the caller-disconnect issue, compile's
    # silent first-forward JIT is fine. 2-3x speedup → finishes in
    # ~3hr instead of ~8hr, saves ~$15.
    compile_enabled: bool = True

    # ── Data mix (single phase, seq 2048) ──────────────────────────
    # 11 streams across 4 categories. Format functions live in
    # data.py::FORMAT_FN_PRETRAIN; streams without a `format` key
    # use the generic _extract_text path (text/content/messages
    # auto-detected).
    # Phase key is hardcoded as "extend" in train.py::run_pretrain_extend
    # (lines 1376, 1488). Keeping the key name shared with extend1 lets
    # the same training loop drive both stages; the `stage_prefix` field
    # is what distinguishes their checkpoint filenames.
    phases: dict = {  # noqa: RUF012
        "extend": {
            "start": 0,
            "end": 3_000,
            "seq_len": 2048,
            # Seq-2048 sizing — same as base pretrain phase 1, leaves
            # headroom for the larger batch needed by reasoning data.
            "batch_size": 8,
            "grad_accum_steps": 8,
            # 9-stream working mix isolated via sanity v9-v25 bisection.
            # Dropped (caused C++ terminate/SIGABRT on first forward
            # pass — root cause unidentified, but reproducibly bad):
            #   * ise-uiuc/Magicoder-Evol-Instruct-110K
            #   * ise-uiuc/Magicoder-OSS-Instruct-75K
            #   * HuggingFaceTB/cosmopedia-v2 python-edu subset
            #   * lukaemon/bbh (not cleanly isolated; skipped)
            # Note: pattern correlates with "datasets containing code
            # content alongside text" — suspect the BPE pre-tokenizer
            # hits an edge case on a specific token sequence that
            # tips the model into a non-finite forward pass.
            "datasets": [
                # ─── Math/Science (45%) ──────────────────────────────
                {
                    "name": "openr1-math-220k",
                    "hf_id": "open-r1/OpenR1-Math-220k",
                    "hf_config": "default",
                    "weight": 0.18,
                    "format": "openr1_math",
                },
                {
                    "name": "open-math-reasoning",
                    "hf_id": "nvidia/OpenMathReasoning",
                    "split": "cot",
                    "weight": 0.15,
                    "format": "openmath_reasoning",
                },
                {
                    "name": "open-web-math",
                    "hf_id": "open-web-math/open-web-math",
                    "weight": 0.07,
                    "format": "arxiv",  # same `text` field shape
                },
                # ─── Reasoning (20%) ────────────────────────────────
                {
                    "name": "open-thoughts-114k",
                    "hf_id": "open-thoughts/OpenThoughts-114k",
                    "hf_config": "default",
                    "weight": 0.15,
                    "format": "openthoughts",
                },
                {
                    "name": "dolmino-flan",
                    "hf_id": "allenai/dolmino-mix-1124",
                    "hf_config": "flan",
                    "weight": 0.08,
                    # Generic _extract_text handles `text` field.
                },
                # ─── Science / academic (10%) ───────────────────────
                {
                    "name": "dolmino-pes2o",
                    "hf_id": "allenai/dolmino-mix-1124",
                    "hf_config": "pes2o",
                    "weight": 0.10,
                    # Generic _extract_text handles `text` field.
                },
                # ─── General-capability anchor (25%) ────────────────
                {
                    "name": "ultrachat-200k",
                    "hf_id": "HuggingFaceH4/ultrachat_200k",
                    "split": "train_sft",
                    "weight": 0.10,
                    # Generic _extract_text handles `messages` field.
                },
                {
                    "name": "cosmopedia-v2",
                    "hf_id": "HuggingFaceTB/cosmopedia-v2",
                    "hf_config": "cosmopedia-v2",
                    "weight": 0.10,
                    # Generic _extract_text handles `text` field.
                },
                {
                    "name": "fineweb-edu",
                    "hf_id": "HuggingFaceFW/fineweb-edu",
                    "weight": 0.07,
                    # Generic _extract_text handles `text` field.
                },
            ],
        },
    }


class LoopFixConfig(PretrainExtend2Config):
    """Architecture-fix continuation: per-loop auxiliary LM-head losses.

    Motivation
    ──────────
    The recursive-loop probe (probe_recursion.py, 2026-06-05) showed
    that loop 5 (the final loop) was doing ~6.0 points of CE loss
    reduction by itself, while loops 1-4 contributed a total of ~0.75
    points across all of them. Effective depth of the 18-effective-
    layer model was closer to ~6 layers (3 blocks × 2 functional loops:
    initial projection + final answer projection). This is a classic
    "loop collapse" — gradient signal flowed entirely through the
    final loop because only its output fed the LM head.

    Fix
    ───
    Attach the SAME weight-tied LM head to the hidden state at the
    end of each non-final loop (after norm_out for path consistency).
    Compute CE loss against the same next-token labels. Add to main
    loss with `aux_loop_loss_weight`. This forces the intermediate
    loops to learn predictive representations rather than just
    setting up the final loop.

    Same data mix + same checkpoint resume as PretrainExtend2 phase 3.
    Short run (1500 steps, ~$5) to validate the architecture fix
    before committing the rest of the post-training budget. Re-run
    probe_recursion.py against the resulting ckpt to confirm loops
    1-4 are now contributing more.

    Schedule
    ────────
    Fresh cosine: re-warm 60 steps → peak 5e-6 → cosine to 5e-7 by
    step 1500. Lower peak than extend2 phase 3 (7e-6) because the
    aux losses add ~0.5× extra gradient pressure and we don't want
    to destabilise the model.
    """

    # Fresh stage with its own step counter (stage_prefix=loopfix has
    # no existing step ckpts so resume scan starts at step 0). Don't
    # use lr_anchor_step here — that pattern is for resuming WITHIN
    # the same stage_prefix. New prefix = fresh counter.
    total_steps: int = 1_500
    lr_anchor_step: int = 0
    warmup_steps: int = 60
    peak_lr: float = 5e-6
    min_lr: float = 5e-7

    # The headline fix. With 6 loops, 5 non-final loops contribute aux
    # losses. At weight 0.1 each, total aux = ~0.5× main loss when
    # all intermediate losses ≈ main loss. Big enough to drive
    # learning, small enough to keep main task dominant.
    aux_loop_loss_weight: float = 0.1

    log_interval: int = 25
    ckpt_interval: int = 200

    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_extend2_final.pt"
    stage_prefix: str = "loopfix"
    wandb_run_name: str = "osrt-loopfix"
    wandb_run_id: str = ""


class LoopFixV2Config(LoopFixConfig):
    """Stacked architecture fixes on top of base loop_fix.

    Layered on the aux LM-head loss from LoopFixConfig:

      1. Loop dropout (stochastic depth) — with prob=0.2, truncate
         the recursive chain to a random length in [3, 6]. Forces
         each loop to be standalone-useful for next-token prediction
         in some fraction of batches.

      2. Aux-weight curriculum — ramp aux_loop_loss_weight from
         0.02 → 0.10 over the first 200 steps. Avoids the initial
         loss shock seen in sanity (main task loss climbed 1.57 →
         2.03 in first 25 steps before recovering).

      3. Per-loop aux weights — bias toward earlier loops, which
         start with less prediction-relevant info and need more
         pressure to become useful. Weights [2.0, 1.5, 1.0, 0.7, 0.5]
         apply the most pressure to loop 0 (the laggard in our probe).

    Resumes from loop_fix's final ckpt (or extend2_final.pt if loop_fix
    hasn't run yet). 1500 steps, same compile/spawn machinery.
    """

    # Loop dropout settings (passed to the model via OSRTConfig).
    loop_dropout_prob: float = 0.2
    loop_dropout_min_loops: int = 3

    # Aux-weight curriculum (read by train.py training loop).
    aux_loop_curriculum_steps: int = 200
    aux_loop_weight_start: float = 0.02
    # The "final" aux_loop_loss_weight is inherited from LoopFixConfig (0.1).

    # Per-loop aux weights bias toward earlier loops (also passed via
    # model config). Length must equal recursive_loops - 1 = 5.
    per_loop_aux_weights: list[float] = [2.0, 1.5, 1.0, 0.7, 0.5]  # noqa: RUF012

    stage_prefix: str = "loopfixv2"
    wandb_run_name: str = "osrt-loopfixv2"


class PretrainExtend3Config(PretrainExtend2Config):
    """Third mid-training pass — first run with WORKING recursive depth.

    Motivation
    ──────────
    All prior pretrain/SFT/GRPO/extend1/extend2 (~30k+ steps) ran with
    a depth-collapsed model effectively using ~6 layers instead of 18
    (probe_recursion 2026-06-05). loop_fix + loop_fix_v2 fixed the
    architecture. The first 300 steps of v2 already showed the
    capability ceiling rising — task CE dropped 1.80 → 1.54 in 300
    steps with the fix on, on the same extend2 data the model had
    seen 8100 steps of. That's not polishing, that's the model
    finally absorbing data it couldn't encode at shallow effective
    depth.

    This stage exploits that: more mid-training on the proven 9-stream
    mix, with the architectural fix permanently in the loss path, so
    the model can actually use all 18 effective layers to absorb the
    data. Expected: another 0.2-0.5 CE drop + meaningful capability
    gain on multi-step tasks (where depth matters most).

    Schedule
    ────────
    Fresh stage_prefix → fresh step counter. 3000 steps at peak_lr
    3e-6 (lower than v2's 5e-6 because we're past the rewiring
    phase and refining capacity). Cosine warmup 60 → 3e-6 → cool to
    3e-7 by step 3000. Cost ~$18.

    Fix knobs
    ─────────
    - aux_loop_loss_weight = 0.05 (half of v2's 0.10 — model has
      already learned to use depth, we just need to keep the signal)
    - loop_dropout_prob = 0.10 (half of v2's 0.20 — same reasoning)
    - per_loop_aux_weights uniform (None — the bias toward early loops
      was a one-time correction; loops are now balanced)
    - aux_loop_curriculum_steps = 100 (short ramp to avoid initial
      shock from changing aux weight)
    - aux_loop_weight_start = 0.02
    """

    total_steps: int = 3_000
    lr_anchor_step: int = 0
    warmup_steps: int = 60
    peak_lr: float = 3e-6
    min_lr: float = 3e-7

    # Fix knobs — softer than v2 since the architecture is already
    # rewired. We're keeping the gradient signal flowing to prevent
    # regression to the collapsed state, not driving major change.
    aux_loop_loss_weight: float = 0.05
    loop_dropout_prob: float = 0.10
    loop_dropout_min_loops: int = 3
    per_loop_aux_weights: None = None  # uniform
    aux_loop_curriculum_steps: int = 100
    aux_loop_weight_start: float = 0.02

    log_interval: int = 50
    ckpt_interval: int = 300

    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_loopfixv2_merged.pt"
    stage_prefix: str = "extend3"
    wandb_run_name: str = "osrt-extend3"
    wandb_run_id: str = ""


class MOPDConfig(PretrainExtend3Config):
    """Multi-teacher On-Policy Distillation (MOPD) from Gemini rollouts.

    Trains on a local JSONL of teacher rollouts (see
    scripts/collect_rollouts.py) using the same training loop as
    pretrain_extend, but with the rollout_dataset_path override that
    swaps in make_rollout_loader instead of the streaming HF loader.

    Per-record format the loader produces:
        <|user|>{prompt}<|assistant|><|think|>{thinking}<|/think|>
        <|answer|>{response}<|/answer|>
    Labels are -100 on the user prefix so cross-entropy fires ONLY on
    the assistant turn (thinking + answer). This is what makes it
    distillation rather than full-sequence LM training.

    The architecture-fix knobs (aux loop loss, loop dropout) remain ON
    at the same low values as extend3 so the recursive depth keeps
    being exercised during distillation — drift back to depth collapse
    would defeat the point of having built that capability.

    Schedule
    ────────
    Lower peak LR (1.5e-6) than mid-training because this is alignment
    on a small, high-quality dataset — we want gentle nudges, not big
    weight moves. 1000 steps at batch=4/accum=16 = ~64K examples seen
    (≈ 14 epochs over 4.4K rollouts). Cosine to 1.5e-7. ~$5-7 Modal.

    Resume from extend3_final.pt (or extend3_merged.pt if we run the
    merge first).
    """

    total_steps: int = 1_000
    lr_anchor_step: int = 0
    warmup_steps: int = 50
    peak_lr: float = 1.5e-6
    min_lr: float = 1.5e-7

    # Keep the architecture fix active during distillation so the model
    # doesn't drift back toward loop collapse on a different objective.
    # Same values as extend3.
    aux_loop_loss_weight: float = 0.05
    loop_dropout_prob: float = 0.10
    loop_dropout_min_loops: int = 3
    per_loop_aux_weights: None = None
    aux_loop_curriculum_steps: int = 50
    aux_loop_weight_start: float = 0.02

    # The crucial override — when set, run_pretrain_extend swaps in
    # make_rollout_loader (data.py) for make_loader. Path is inside the
    # Modal container, so it's mounted via volume below.
    rollout_dataset_path: str = "/vol/rollouts/mopd_v1.jsonl"

    log_interval: int = 25
    ckpt_interval: int = 200

    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_extend3_merged.pt"
    stage_prefix: str = "mopd"
    wandb_run_name: str = "osrt-mopd"
    wandb_run_id: str = ""


class SystemSFTConfig(MOPDConfig):
    """System-prompt SFT — teaches the model to attend to <|system|> blocks.

    Resumes from grpo_v2_step_50.pt (or mopd_final.pt) and trains on
    rollouts where each record has a `system` field. RolloutDataset
    formats them as:
        <|system|>{sys}<|user|>{prompt}<|assistant|>{response}

    Loss is computed only on the assistant turn (the system+user
    prefix is masked with -100), so the model learns to:
      1. USE the system prompt as context for generation
      2. NOT regenerate the system prompt verbatim (that would mean
         training the model to predict tokens that are in the
         loss-ignored prefix — there's no gradient teaching it to
         echo, only to attend)
    The regurgitation penalty added later in GRPO is a
    backstop for residual echoing this SFT doesn't fully eliminate.

    Schedule:
        500 steps from grpo_v2_step_50.pt at peak_lr 1e-6 → cosine
        to 1e-7. Same architecture-fix knobs as MOPD (aux=0.05,
        loop_dropout=0.10) so depth utilisation stays preserved.
        ~$5-7 Modal at batch=4/accum=8 over ~10K rollouts.
    """

    total_steps: int = 500
    lr_anchor_step: int = 0
    warmup_steps: int = 25
    peak_lr: float = 1e-6  # lower than MOPD — small alignment tweak
    min_lr: float = 1e-7
    grad_accum_steps: int = 8

    aux_loop_loss_weight: float = 0.05
    loop_dropout_prob: float = 0.10
    loop_dropout_min_loops: int = 3
    aux_loop_curriculum_steps: int = 0  # already curriculum'd
    aux_loop_weight_start: float = 0.05

    rollout_dataset_path: str = "/vol/rollouts/system_prompt_sft.jsonl"

    log_interval: int = 10
    ckpt_interval: int = 100  # 5 ckpts

    # Resume from the best v2 ckpt we have (grpo_v2 step_50 was the
    # only v2 ckpt actually saved before we stopped). Falls back to
    # mopd_final if grpo_v2_step_50.pt isn't on the volume.
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_grpo_v2_step_50.pt"
    stage_prefix: str = "sys_sft"
    wandb_run_name: str = "osrt-sys-sft"
    wandb_run_id: str = ""


class MidtrainConfig(PretrainConfig):
    """v6 mid-training: continued PRETRAINING on the foundation base.

    Resumes from the annealed v6 foundation checkpoint (step 3500),
    re-warms a fresh cosine at a real continued-pretraining LR (2e-4),
    doubles context to seq 4096, and trains the math-heavy knowledge mix.

    Unlike the v5 PretrainExtend* stages this does NOT resume from an
    SFT/GRPO checkpoint, so there is no chat-format investment to protect:
    HRA stays TRAINABLE and the LR is ~33% of foundation peak, not the
    2.5% the v5 stages used.

    HRA is NATIVE here (built inline from the preset's adapter_rank=256
    and already present in the foundation checkpoint), so hra_native=True
    skips inject_hra — see run_pretrain_extend.

    See docs/superpowers/specs/2026-06-09-v6-midtraining-design.md.
    """

    # ── Schedule (fresh re-warm cosine) ──────────────────────────────
    # total_steps drives the cosine: with lr_anchor_step=0 the LR anneals
    # peak_lr → min_lr over the full total_steps (see _set_param_group_lrs:
    # eff_total = total_steps - anchor). REDUCED 8000 → 5500 mid-run after
    # the wallet check ($147 left, 8000 needed ~$213): at ~27.6s/step,
    # 5500 lands ~$140 total with margin, and the cosine reshapes on
    # resume so the checkpoint is properly ANNEALED to min_lr at 5500
    # rather than dying mid-cool at a budget kill. ~1.49B tokens.
    total_steps: int = 5_500
    warmup_steps: int = 150          # re-warm from the annealed base
    lr_anchor_step: int = 0          # fresh cosine (foundation already cooled)
    peak_lr: float = 2e-4            # ~33% of foundation's 6e-4
    min_lr: float = 2e-5             # cosine floor at step 5500
    weight_decay: float = 0.1        # softer than foundation's 0.3
    grad_clip: float = 1.0

    optimizer_name: str = "muon"
    muon_lr: float = 6.6e-3          # proportional: (2e-4/6e-4) * 0.02
    muon_min_lr: float = 6.6e-4

    log_interval: int = 50
    ckpt_interval: int = 500         # 16 ckpts over 8000; bounds Modal-kill loss
    eval_interval: int = 500         # held-out eval every 500 (16 evals)
    eval_steps: int = 20

    # ── Router exploration: off (router is well-formed) ──────────────
    router_gumbel_tau_init: float = 0.0
    router_gumbel_tau_final: float = 0.0
    router_gumbel_anneal_steps: int = 1

    # ── Early-stop gate: disabled (cold-start gate doesn't apply) ────
    early_stop_check_step: int = 9_999_999

    # ── HRA: native + trainable ──────────────────────────────────────
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_scale: float = 1.0
    hra_native: bool = True          # skip inject_hra (run_pretrain_extend)
    hra_frozen: bool = False         # trainable

    # ── Resume / lineage ─────────────────────────────────────────────
    # Foundation final ckpt (run_training writes osrt_v5_final.pt; the
    # 500-step interval also leaves osrt_v5_step_3500.pt). If a run was
    # killed before the final save, repoint at osrt_v5_step_3500.pt.
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_final.pt"
    # Checkpointing ON — REQUIRED at seq 4096. We tested the throughput bet
    # (checkpointing OFF buys ~25-30% by skipping activation recompute on the
    # 18 effective layers) with the midtrain_sanity probe: it OOM'd at
    # 78.36GB/79.18GB on the H100 (storing all 18 loops' activations at seq
    # 4096 / batch 6 doesn't fit 80GB). So the bet lost — back to True. The
    # foundation run also used checkpointing (app.py:419) at seq 2048/39.5GB.
    # run_pretrain_extend reads this via getattr and sets the real
    # _osrt_grad_ckpt gate (model.py use_ckpt).
    gradient_checkpointing: bool = True

    # Distinct prefix — osrt_v5_midtrain_step_*.pt, no collision with
    # foundation's osrt_v5_step_*.pt resume scan.
    stage_prefix: str = "midtrain"

    wandb_run_name: str = "osrt-v6-midtrain"
    wandb_run_id: str = ""

    # ── Data mix: the knowledge phase (seq 4096, math-heavy) ─────────
    # Single phase keyed "extend" (run_pretrain_extend reads
    # phases["extend"]). Content mirrors PretrainConfig.phases["knowledge"].
    phases: dict = {  # noqa: RUF012
        "extend": {
            "start": 0,
            "end": 5_500,
            "seq_len": 4096,
            "batch_size": 6,         # knowledge-phase sizing; sanity-gated
            "grad_accum_steps": 11,
            "datasets": [
                {"name": "nemotron-cc-math-4plus",
                 "hf_id": "nvidia/Nemotron-CC-Math-v1",
                 "hf_config": "4plus", "weight": 0.25},
                {"name": "nemotron-stem",
                 "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                 "hf_config": "Nemotron-Pretraining-STEM-SFT", "weight": 0.15},
                {"name": "nemotron-math-textbooks",
                 "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                 "hf_config": "Nemotron-Pretraining-Math-Textbooks",
                 "weight": 0.15},
                {"name": "nemotron-reasoning",
                 "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                 "hf_config": "Nemotron-Pretraining-InfiniByte-Reasoning",
                 "weight": 0.10},
                {"name": "fineweb-edu",
                 "hf_id": "HuggingFaceFW/fineweb-edu", "weight": 0.15},
                {"name": "nemotron-code-syn-qa",
                 "hf_id": "nvidia/Nemotron-Pretraining-Code-v2",
                 "hf_config": "Synthetic-Question-Answering", "weight": 0.10},
                {"name": "cosmopedia-openstax",
                 "hf_id": "HuggingFaceTB/cosmopedia",
                 "hf_config": "openstax", "weight": 0.10},
            ],
        },
    }

    # DataLoader: ONE worker. Each worker independently opens ALL 7 streams
    # and fills a 5k-example shuffle buffer per stream before yielding, so
    # cold-start cost scales as num_workers × n_streams. Probe #3 set this to
    # 2 (=14 stream-opens) and never assembled the first batch in ~6 min on a
    # cold container; the W&B GPU monitor logged zero rows (stuck in data
    # assembly, not OOM/compute). Foundation reached step 0 fine at
    # num_workers=1 / 4 streams, and PretrainExtend2Config sets =1 with the
    # same "many concurrent HF streams cascade" rationale. One worker = 7
    # stream-opens, comparable to foundation. Throughput cost is negligible
    # vs the gradient-checkpointed seq-4096 step time.
    dataloader_num_workers: int = 1
    dataloader_prefetch_factor: int = 2
    compile_enabled: bool = True


class MidtrainSanityConfig(MidtrainConfig):
    """30-step VRAM/throughput probe at the REAL seq/batch before the
    $150 launch. Writes no final checkpoint, distinct prefix, eager mode
    so step events appear immediately."""

    total_steps: int = 30
    warmup_steps: int = 5
    ckpt_interval: int = 9_999_999
    eval_interval: int = 9_999_999
    save_final_checkpoint: bool = False
    stage_prefix: str = "midtrain-sanity"
    wandb_run_name: str = "osrt-v6-midtrain-sanity"
    compile_enabled: bool = False


class SFTConfig:
    """Balanced SFT config for v5 — math + code + STEM + general."""

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 8
    total_steps: int = 5_000
    warmup_steps: int = 250
    peak_lr: float = 1.5e-5
    min_lr: float = 1.5e-6
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 25
    ckpt_interval: int = 500
    optimizer_name: str = "adamw"
    seq_len: int = 2048              # short seq_len + packing (see v4_sft_data)

    # HRA
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_scale: float = 1.0
    hra_lr: float = 7.5e-5
    hra_freeze_pretrained: bool = False
    hra_before_load: bool = False
    stage_prefix: str = "sft"

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "osrt"
    wandb_run_name: str = "osrt-sft"
    wandb_run_id: str = ""

    # Pretrained checkpoint. Points to the explicit step file rather
    # than `osrt_v5_final.pt` because pretraining was stopped early at
    # step 17000 once the eval-loss curve flatlined (eval 3.48 / ppl
    # 32.4 — Chinchilla-knee on 192M active params). Path is set
    # explicitly so SFT loads the actual snapshot, not a stale
    # filename convention.
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_step_17000.pt"

    # Chat format (native single-token tags)
    user_tag: str = "<|user|>"
    assistant_tag: str = "<|assistant|>"
    think_open: str = "<|think|>"
    think_close: str = "<|/think|>"
    answer_open: str = "<|answer|>"
    answer_close: str = "<|/answer|>"

    # Balanced dataset mixture (same as v4)
    datasets: list = [  # noqa: RUF012
        # Math (25%)
        {
            "name": "gsm8k",
            "hf_id": "openai/gsm8k",
            "hf_config": "main",
            "split": "train",
            "weight": 0.10,
            "format": "gsm8k",
        },
        {
            "name": "numina-math-cot",
            "hf_id": "AI-MO/NuminaMath-CoT",
            "split": "train",
            "weight": 0.15,
            "format": "numina_math",
        },
        # Code (25%)
        {
            "name": "evol-instruct-code",
            "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
            "split": "train",
            "weight": 0.15,
            "format": "evol_code",
        },
        {
            "name": "code-instructions-122k",
            "hf_id": "TokenBender/code_instructions_122k_alpaca_style",
            "split": "train",
            "weight": 0.10,
            "format": "alpaca_code",
        },
        # STEM (20%)
        {
            "name": "orca-math",
            "hf_id": "microsoft/orca-math-word-problems-200k",
            "split": "train",
            "weight": 0.10,
            "format": "orca_math",
        },
        {
            "name": "math-instruct",
            "hf_id": "TIGER-Lab/MathInstruct",
            "split": "train",
            "weight": 0.10,
            "format": "math_instruct",
        },
        # General (20%)
        {
            "name": "alpaca-cleaned",
            "hf_id": "yahma/alpaca-cleaned",
            "split": "train",
            "weight": 0.10,
            "format": "alpaca",
        },
        {
            "name": "openhermes",
            "hf_id": "teknium/OpenHermes-2.5",
            "split": "train",
            "weight": 0.10,
            "format": "openhermes",
        },
        # Instruction following (10%)
        {
            "name": "ifeval-like",
            "hf_id": "argilla/ifeval-like-data",
            "hf_config": "filtered",
            "split": "train",
            "weight": 0.05,
            "format": "ifeval",
        },
        {
            "name": "longform",
            "hf_id": "akoksal/LongForm",
            "split": "train",
            "weight": 0.05,
            "format": "longform",
        },
    ]


class MidtrainExtendConfig(MidtrainConfig):
    """v6 midtrain phase 2 — extended continued-pretraining to push the badly
    undertrained base toward Chinchilla.

    The base has seen ~1.7B tokens (foundation ~0.46B + midtrain ~1.22B) — only
    ~0.3x Chinchilla-optimal for 278M active params. v1's incoherent reasoning
    is likely undertraining-dominant, not just short-SFT-data. So before any
    more post-training we add tokens with a fresh re-warm cosine on a
    REASONING/INSTRUCTION-heavy mix (the modern annealing/decay phase): the
    knowledge mix already contains SFT-style text (Nemotron STEM-SFT +
    InfiniByte-Reasoning + math textbooks), so we just reweight toward it — no
    loader change. Full-sequence LM (every token gets gradient, unlike masked
    SFT) = maximal capability per dollar on an undertrained base.

    Budget: ~4000 steps @ seq 4096 ≈ ~1.1B tokens ≈ ~$110 on Lightning, taking
    the base to ~2.8B (~0.5x Chinchilla). A real lift, not "good" (that needs
    ~$2.5K of tokens) — but the right use of a tight budget vs SFT'ing an
    undertrained base. Resume from midtrain_final; reassess capability after.
    """

    # 2000 steps ≈ ~$61 on Modal H100 — spans TWO $30 workspaces by design:
    # workspace 1 dies around step ~1000 (ckpt_interval 250 keeps step_1000),
    # then chain: `modal volume get` step_1000 + re-upload to workspace 2 with
    # midtrain_final, and the resume scan continues the SAME 2000-step cosine
    # to the fully-annealed end. First verdict at the step-250 eval; if ppl is
    # flat at 250/500 the base is saturated → kill early, save the credit.
    # --total-steps on the Lightning entry overrides this for other runs.
    total_steps: int = 2_000
    warmup_steps: int = 50
    lr_anchor_step: int = 0           # fresh re-warm cosine over the full run
    peak_lr: float = 3e-5             # GENTLE re-warm — the 1e-4 was too hot for
                                      # an annealed base (ppl rose 30→34, flat)
    min_lr: float = 1e-5
    muon_lr: float = 9.9e-4           # proportional to the AdamW peak (×33)
    muon_min_lr: float = 3.3e-4

    eval_interval: int = 250          # frequent verdict points on a capped run
    ckpt_interval: int = 250          # a credit-death then loses ≤250 steps
    log_interval: int = 50
    dataloader_num_workers: int = 1   # avoid the cold-stream connection storm

    # Reasoning/instruction-heavy reweight (reasoning+STEM-SFT+math = 0.75,
    # was 0.65). Same sources/seq as midtrain — just emphasises the SFT-style
    # Nemotron splits the user wants more of. Weights sum to 1.0.
    phases: dict = {  # noqa: RUF012
        "extend": {
            "seq_len": 4096,
            "batch_size": 6,
            "grad_accum_steps": 11,
            "datasets": [
                {"name": "nemotron-cc-math-4plus",
                 "hf_id": "nvidia/Nemotron-CC-Math-v1",
                 "hf_config": "4plus", "weight": 0.20},
                {"name": "nemotron-stem",
                 "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                 "hf_config": "Nemotron-Pretraining-STEM-SFT", "weight": 0.20},
                {"name": "nemotron-math-textbooks",
                 "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                 "hf_config": "Nemotron-Pretraining-Math-Textbooks",
                 "weight": 0.15},
                {"name": "nemotron-reasoning",
                 "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                 "hf_config": "Nemotron-Pretraining-InfiniByte-Reasoning",
                 "weight": 0.20},
                {"name": "fineweb-edu",
                 "hf_id": "HuggingFaceFW/fineweb-edu", "weight": 0.10},
                {"name": "nemotron-code-syn-qa",
                 "hf_id": "nvidia/Nemotron-Pretraining-Code-v2",
                 "hf_config": "Synthetic-Question-Answering", "weight": 0.075},
                {"name": "cosmopedia-openstax",
                 "hf_id": "HuggingFaceTB/cosmopedia",
                 "hf_config": "openstax", "weight": 0.075},
            ],
        },
    }

    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_midtrain_final.pt"
    stage_prefix: str = "midtrain2"
    wandb_run_name: str = "osrt-v6-midtrain2"
    wandb_run_id: str = ""


class MidtrainExtendSanityConfig(MidtrainExtendConfig):
    """30-step probe: native-HRA loads clean from midtrain_final, the reweighted
    mix streams, VRAM fits at seq 4096 — before the ~$110 extend run."""

    total_steps: int = 30
    warmup_steps: int = 5
    ckpt_interval: int = 9_999_999
    eval_interval: int = 9_999_999
    save_final_checkpoint: bool = False
    compile_enabled: bool = False
    stage_prefix: str = "midtrain2-sanity"
    wandb_run_name: str = "osrt-v6-midtrain2-sanity"


class MidtrainExtend3Config(MidtrainExtendConfig):
    """v6 midtrain phase 3 — the LONG capability-building continued-pretrain,
    chained across monthly $30 Modal workspaces.

    SFT v2 confirmed the base is UNDERTRAINED, not capacity-capped: format_ok
    hit 1.0 (clean <|think|>/<|answer|>) but GSM8K stayed ~0.05 (=SFT v1), and
    generations are fluent-but-wrong ("50/80 = 6.25%", "32 inches in an inch").
    That's the signature of a base that learned the SHAPE of reasoning but has
    too few tokens for the substance. Models this size that reason (Qwen2.5-0.5B
    ~40% GSM8K) saw trillions of tokens; we've seen ~2.2B (~0.4x Chinchilla on
    278M active). The fix is more PRETRAINING, not more SFT/RL.

    Target: +3.4B tokens → ~5.6B total = 1x Chinchilla, the first point GSM8K
    should lift off the floor. At eff-batch 66 x 4096 = 270k tok/step that's
    ~12,600 steps. One ~$30 workspace ≈ 1000 steps (~270M tok), so this is a
    ~13-workspace, multi-month drip: each month resume from the highest
    midtrain3 checkpoint (the resume-scan chains it) and continue the SAME long
    cosine. Re-run SFT v2 (unchanged — it already produces clean form) once the
    base is stronger.

    LR: peak 5e-5 — above midtrain2's gentle 3e-5 probe (this is a LONG
    capability push, not a short re-warm, so it can afford more learning per
    token) but well under the 1e-4 that displaced the base over 1000 steps. The
    long cosine barely moves in the first 1000 steps, so it's effectively a
    sustained 5e-5 that anneals to 1e-5 only near the 12,600-step end. Base =
    midtrain2_step_1750 (the best intact midtrain2 artifact, ppl 28.2).
    """

    pretrained_checkpoint: str = (
        "/vol/checkpoints/v5/osrt_v5_midtrain2_step_1750.pt"
    )
    stage_prefix: str = "midtrain3"
    wandb_run_name: str = "osrt-v6-midtrain3"

    total_steps: int = 12_600         # +3.4B tokens → ~1x Chinchilla (multi-month)
    warmup_steps: int = 100
    peak_lr: float = 5e-5             # sustained capability LR over a long run
    min_lr: float = 1e-5
    muon_lr: float = 1.65e-3          # ×33 of the AdamW peak
    muon_min_lr: float = 3.3e-4

    # In-loop eval DISABLED: the held-out skip=100M build stalls the GPU for
    # 20-30min (single-threaded with num_workers=0), and Colab reclaims
    # idle-GPU VMs — it killed a run at step 500 right before its checkpoint.
    # We eval checkpoints OFFLINE anyway (run_sft_eval / probes), so the
    # in-loop eval is pure liability here.
    eval_interval: int = 9_999_999
    # Checkpoint every 100 (not 500): given repeated Colab VM reclaims, bank
    # progress to HF fast — first checkpoint ~1hr in, a reclaim costs ≤100
    # steps. hf_ckpt_sync prunes remote to the newest few so HF stays small.
    ckpt_interval: int = 100
    dataloader_num_workers: int = 1


class MidtrainExtend3SanityConfig(MidtrainExtend3Config):
    """30-step probe for midtrain3 (clean load of step_1750 + mix streams)."""

    total_steps: int = 30
    warmup_steps: int = 5
    ckpt_interval: int = 9_999_999
    eval_interval: int = 9_999_999
    save_final_checkpoint: bool = False
    compile_enabled: bool = False
    stage_prefix: str = "midtrain3-sanity"
    wandb_run_name: str = "osrt-v6-midtrain3-sanity"


class SFTv1Config(SFTConfig):
    """v6 SFT v1 — system-prompt instruction tuning on the midtrain base.

    Roadmap Stage 1: make the mid-trained base FOLLOW a <|system|> prompt and
    emit the <|think|>/<|answer|> format. Each example is prefixed with a
    <|system|>{persona} turn (set system_tag), the persona sampled per-example
    from the reasoning-on/off pool chosen by each dataset's reasoning_mode —
    so "follow the system prompt" is literally true in both directions AND the
    reasoning-on/off toggle (the project's north-star metric) is built in.
    See docs/superpowers/specs/2026-06-11-sft-v1-design.md.

    NOT the long-reasoning stage. Moderate length (200-800 tok), seq 2048,
    HRA trainable, on the fully-annealed midtrain_final base.
    """

    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_midtrain_final.pt"
    stage_prefix: str = "sft_v1"
    seq_len: int = 2048
    # The v6 model (601M, mHC, MTP, 8 experts) OOMs at batch8/seq2048
    # un-checkpointed (the SFTConfig defaults were sized for the v5 363M).
    # Enable gradient checkpointing (run_sft sets _osrt_grad_ckpt) AND split
    # the effective-64 batch as 4x16 so the per-step activation peak is halved.
    gradient_checkpointing: bool = True
    batch_size: int = 4
    grad_accum_steps: int = 16        # eff batch 64 (unchanged), lower peak mem
    total_steps: int = 2_000          # 4 x 16 x 2048 = 131K tok/step ≈ 260M tok
    # peak_lr 1.5e-5 → min 1.5e-6, warmup 250, AdamW — inherited from SFTConfig.
    eval_interval: int = 500
    ckpt_interval: int = 500

    # System turn + length floor (consumed by SFTStream/make_sft_loader).
    system_tag: str = "<|system|>"
    min_response_tokens: int = 150    # "not too short"

    # HRA stays trainable but is NATIVE (built from config, already in the
    # midtrain ckpt) — set hra_native so run_sft skips inject_hra (the v5
    # HRALinear graft would break the load). With hra_native, hra_params is
    # empty → run_sft trains ALL params (incl. native HRA) at the single SFT
    # LR (no differential-LR split needed).
    hra_native: bool = True

    wandb_run_name: str = "osrt-v6-sft-v1"
    wandb_run_id: str = ""

    # Data mix (§5 of the spec). reasoning_mode selects the persona pool:
    # 'on'  → math (real CoT), 'off' → general/chat/code (answer-direct).
    datasets: list = [  # noqa: RUF012
        {
            "name": "tulu3-sft",
            "hf_id": "allenai/tulu-3-sft-mixture",
            "split": "train",
            "weight": 0.30,
            "format": "tulu",
            "reasoning_mode": "off",
        },
        {
            "name": "openhermes",
            "hf_id": "teknium/OpenHermes-2.5",
            "split": "train",
            "weight": 0.25,
            "format": "openhermes",
            "reasoning_mode": "off",
        },
        {
            "name": "gsm8k",
            "hf_id": "openai/gsm8k",
            "hf_config": "main",
            "split": "train",
            "weight": 0.20,
            "format": "gsm8k",
            "reasoning_mode": "on",
        },
        {
            "name": "numina_math",
            "hf_id": "AI-MO/NuminaMath-CoT",
            "split": "train",
            "weight": 0.15,
            "format": "numina_math",
            "reasoning_mode": "on",
        },
        {
            "name": "evol_code",
            "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
            "split": "train",
            "weight": 0.10,
            "format": "evol_code",
            "reasoning_mode": "off",
        },
    ]


class SFTv1SanityConfig(SFTv1Config):
    """30-step SFT-v1 probe: verifies the <|system|> turn builds, native-HRA
    loads clean, masking is right, and VRAM fits — before the paid run.
    Writes no final, distinct prefix, skips the (slow) reasoning eval."""

    total_steps: int = 30
    warmup_steps: int = 5
    ckpt_interval: int = 9_999_999
    eval_interval: int = 9_999_999     # skip the generation eval in the probe
    save_final_checkpoint: bool = False
    stage_prefix: str = "sft_v1-sanity"
    wandb_run_name: str = "osrt-v6-sft-v1-sanity"


class SFTv2Config(MidtrainConfig):
    """v6 SFT v2 — reasoning distillation from the clean midtrain base.

    The v6 line went pretrain → midtrain → SFT v1 (system-prompt instruction
    tuning), which proved the format works but left reasoning incoherent. The
    v5 line had a reasoning-distillation stage (MOPD) the v6 line skipped; this
    closes that gap by training the midtrain base on the long, coherent teacher
    CoT in rollouts/sft_v2.jsonl (built by scripts/build_sft_v2_data.py from
    mopd_v1 + system_prompt_sft, with reasoning-on/off personas baked into each
    record's `system` field — 65/20/15 ON/OFF/CHAT, all ≤ 4096 tokens).

    Mechanism: reuses run_pretrain_extend (the MOPD rollout-loader path) with
    MidtrainConfig's proven v6 plumbing — native HRA, seq 4096, gradient
    checkpointing — but with a gentle SFT schedule (peak 1e-5, not midtrain's
    continued-pretrain 2e-4) since this is supervised alignment on a small,
    high-quality set. RolloutDataset reads system/prompt/thinking/response and
    masks the prefix, so loss fires only on the assistant turn (think+answer).

    Eval: in-loop perplexity eval is DISABLED (it would stream the knowledge
    mix, not the rollouts — confusing). The north-star reasoning-on/off GSM8K
    eval runs OFFLINE on the periodic checkpoints via scripts/local_sft_eval.py
    (fast now that generation is batched). ckpt_interval=300 → 5 checkpoints.
    """

    # ── Schedule: gentle SFT (not continued-pretrain) ────────────────
    # 1000 steps: the sanity gate measured steady-state ~33s/step (8k tok/s at
    # batch 4, seq 4096, grad-ckpt) — SLOWER than the ~25s first assumed, so
    # 1200 would run ~11hr ≈ $43 and blow one $40 workspace mid-anneal. 1000
    # completes cleanly (~9.2hr ≈ $37, fully annealed) and is still ~1.2 epochs
    # over the 53k-row verified corpus — SFT converges well within this (v1
    # plateaued by ~1000). ckpt every 200 → the $40 wall can't lose progress.
    total_steps: int = 1_000
    warmup_steps: int = 100
    lr_anchor_step: int = 0          # fresh cosine over the full run
    peak_lr: float = 1e-5            # SFT-scale (cf. SFT v1 1.5e-5); NOT 2e-4
    min_lr: float = 1e-6
    # Muon group proportional to the AdamW peak (midtrain ratio ~33×).
    muon_lr: float = 3.3e-4
    muon_min_lr: float = 3.3e-5

    # Keep recursive depth exercised during SFT (same rationale as MOPD).
    aux_loop_loss_weight: float = 0.05
    loop_dropout_prob: float = 0.10
    loop_dropout_min_loops: int = 3

    # ── Rollout-loader override (the MOPD mechanism) ─────────────────
    # run_pretrain_extend swaps make_rollout_loader in when this is set. The
    # Lightning/Modal entrypoints repoint it at the local path as needed.
    rollout_dataset_path: str = "/vol/rollouts/sft_v2.jsonl"

    # ── Phase sizing for the rollout path (seq/batch/accum read here) ─
    # seq 4096 (long CoT genuinely needs it); eff batch 4×16=64. The datasets
    # list is unused on the rollout path (only printed) — placeholder name.
    phases: dict = {  # noqa: RUF012
        "extend": {
            "seq_len": 4096,
            "batch_size": 4,
            "grad_accum_steps": 16,
            "datasets": [{"name": "sft_v2_rollout", "weight": 1.0}],
        },
    }

    # In-loop perplexity eval off (see docstring); checkpoint often — 6 ckpts
    # over 1200 steps = soup candidates + free local reasoning-evals per ckpt.
    eval_interval: int = 9_999_999
    ckpt_interval: int = 200
    log_interval: int = 25

    # ── Lineage ──────────────────────────────────────────────────────
    # Base = midtrain2 step_1750 (ppl 28.2), the best INTACT midtrain2
    # artifact: the step-2000 final save was interrupted mid-write on the
    # volume (truncated at 2.4/4.9GB, unloadable — verified by two
    # independent downloads). The lost 1750→2000 tail at lr ≤1.3e-5 was
    # worth ~0.1 ppl; SFT re-warms its own schedule anyway.
    pretrained_checkpoint: str = (
        "/vol/checkpoints/v5/osrt_v5_midtrain2_step_1750.pt"
    )
    stage_prefix: str = "sft_v2"
    wandb_run_name: str = "osrt-v6-sft-v2"
    wandb_run_id: str = ""


class SFTv2SanityConfig(SFTv2Config):
    """30-step SFT-v2 probe: confirms the rollout loader builds the v6
    system/think/answer sequence, native-HRA loads clean from midtrain_final,
    and VRAM fits at seq 4096 — before the paid run. No final, distinct prefix."""

    total_steps: int = 30
    warmup_steps: int = 5
    ckpt_interval: int = 9_999_999
    save_final_checkpoint: bool = False
    compile_enabled: bool = False
    stage_prefix: str = "sft_v2-sanity"
    wandb_run_name: str = "osrt-v6-sft-v2-sanity"


class SFTv3Config(SFTv2Config):
    """v6 SFT v3 — reasoning distillation on the midtrain3 (Chinchilla) base.

    Corpus: rollouts/sft_v3.jsonl (scripts/build_sft_v3_data.py): 42,215 rows
    = verified v2 anchor (mopd gold-checked + openr1/stratos/chat subsamples)
    + Nemotron-PT math/science thinking-ON + math/chat OFF + smoltalk2
    instruct/chat. 54/30/15 ON/OFF/CHAT; 8-gram decontamination vs GSM8K-test
    + MATH-500; cross-source problem dedup; all rows ≤4096 assembled tokens.
    Report: rollouts/sft_v3_report.md. Also mirrored on HF
    HallD/osrt-v6-ckpt @ data/sft_v3.jsonl (pulled by the sft_v3_prep stage).

    Sizing: RolloutDataset is one example per seq_len row (NO packing), so
    42,215 rows / eff-batch 64 ≈ 660 steps/epoch. 800 steps ≈ 1.2 epochs ≈
    7.3h ≈ $29 at v2's measured ~33s/step (identical shape: H100, batch 4,
    seq 4096, grad-ckpt) — fits a $40 workspace with margin; v1 and v2 both
    plateaued by ~1000 steps. Same gentle LR as v2 (peak 1e-5); the base is
    stronger but the mechanism unchanged.
    """

    total_steps: int = 800
    warmup_steps: int = 80
    rollout_dataset_path: str = "/vol/rollouts/sft_v3.jsonl"

    # Base = midtrain3_final (step 12,600, ~1x Chinchilla; math ppl 2.97 /
    # fineweb 26.30). Pulled from HF by `--stage sft_v3_prep` on fresh
    # workspaces — the v6 line's ONLY intact post-midtrain3 base artifact.
    pretrained_checkpoint: str = (
        "/vol/checkpoints/v5/osrt_v5_midtrain3_final.pt"
    )
    ckpt_interval: int = 200
    stage_prefix: str = "sft_v3"
    wandb_run_name: str = "osrt-v6-sft-v3"


class SFTv4Config(SFTv3Config):
    """v6 SFT v4 — length-matched reasoning, broadened mix, LONGER schedule.

    Why v4 (see scripts/build_sft_v4_data.py for the full measurement): v3's
    reasoning data was unusable at this scale — 4-8k-char R1 traces the model
    cannot execute (it emits 250-440c) plus 456c of mopd meta-narration, with
    essentially nothing in between. It learned to narrate. GSM8K 0/20, and ON
    vs OFF produced the SAME wrong answers. v4 teaches brief correct
    derivation from GSM8K-train's own human solutions (median 249c) plus
    orca-math, caps ON thinking at 2,000c, rejects ON rows whose think is
    shorter than their answer, cuts math/science from 64% to ~48%, and adds
    general reasoning, rewriting/summarising, and tool calling.

    Schedule: 1,200 steps ~ 1.7 epochs over ~43k rows at eff-batch 64 — 50%
    more than v3's 1.2 epochs. Sized against MEASURED throughput (9,582 tok/s
    at the end of the v3 run => 27.4s/step): 9.1h ~ $39, which is the most
    that fits ONE $40 workspace with the cosine fully annealed. Going longer
    inside one workspace risks the budget wall stopping the run mid-anneal and
    leaving an un-annealed checkpoint (the midtrain2 LR-displacement lesson);
    if held-out loss is still falling at 1,200 the right move is a separate
    extend run with a fresh short cosine, not a partial anneal.

    Stopping signal: `rollout_eval_path` + `rollout_eval_interval` log
    held-out SFT loss every 100 steps. v3 was sized off a TRAINING-loss
    plateau, which midtrain3 had already shown to be unreliable here (train
    loss flat while held-out ppl kept improving). Never size the next SFT run
    off train loss again.
    """

    # seq_len 2048, NOT 4096. v4's corpus is 2.4x shorter than v3's (median
    # 388 assembled tokens vs 925) because 6k-char R1 traces were replaced
    # with 250-500c derivations — at 4096 that is 88% PADDING, i.e. we would
    # burn ~half the budget computing pad positions. Measured on the built
    # corpus: only 1.0% of rows exceed 2048, and those were DROPPED at build
    # time rather than left to truncate mid-derivation (an unfinished chain is
    # exactly what we must not teach). Halving seq_len halves step time, so
    # the same ~$39 buys 2,400 steps ~ 3.0 epochs instead of 1.5.
    # Safe for inference at longer contexts: 2048 < the 4096 the base was
    # pretrained and midtrained at, so no RoPE position is left untrained.
    # 2,200 not 2,400: measured eager throughput on the sanity was 8,175 tok/s
    # => 16.0s/step; v3 gained ~22% from torch.compile, so expect ~13.1s/step
    # => 2,200 steps ~ 8.0h ~ $34, leaving margin under the $40 workspace wall
    # even if compile underdelivers. 2,400 would land at $37-41 — i.e. it could
    # be killed mid-anneal, which is the one outcome to avoid. Still ~2.7
    # epochs vs v3's 1.2.
    total_steps: int = 2_200
    warmup_steps: int = 150
    rollout_dataset_path: str = "/vol/rollouts/sft_v4.jsonl"
    # batch 8 x accum 8 = eff batch 64, unchanged from v2/v3 (so the
    # optimisation trajectory stays comparable) but with twice the micro-batch.
    # VRAM should FALL vs v3's 44.8GB despite the bigger batch: MLP/norm
    # activations scale with batch*seq (16,384 tokens either way, identical),
    # while attention scales with batch*seq^2 — 8*2048^2 is HALF of 4*4096^2.
    # Bigger micro-batches also use the tensor cores better, so step time may
    # beat the 2x that halving seq_len alone predicts. The sanity run measures
    # both before the paid run commits.
    phases: dict = {  # noqa: RUF012
        "extend": {
            "seq_len": 2048,
            "batch_size": 8,
            "grad_accum_steps": 8,
            "datasets": [{"name": "sft_v4_rollout", "weight": 1.0}],
        },
    }

    # Held-out SFT eval — the signal that says when to stop.
    rollout_eval_path: str = "/vol/rollouts/sft_v4_val.jsonl"
    rollout_eval_interval: int = 100
    rollout_eval_steps: int = 32     # 32 x batch 4 = 128 held-out examples

    ckpt_interval: int = 200         # 6 checkpoints; pick the best by eval
    stage_prefix: str = "sft_v4"
    wandb_run_name: str = "osrt-v6-sft-v4"


class SFTv4SanityConfig(SFTv4Config):
    """30-step SFT-v4 probe: corpus + val split parse, midtrain3_final loads
    clean, held-out eval path works, VRAM fits at seq 4096."""

    total_steps: int = 30
    warmup_steps: int = 5
    rollout_eval_interval: int = 10   # exercise the eval path in the probe
    rollout_eval_steps: int = 4
    ckpt_interval: int = 9_999_999
    save_final_checkpoint: bool = False
    compile_enabled: bool = False
    stage_prefix: str = "sft_v4-sanity"
    wandb_run_name: str = "osrt-v6-sft-v4-sanity"


class SFTv4Batch16SanityConfig(SFTv4SanityConfig):
    """Paired batch-16 probe against SFTv4SanityConfig's batch 8.

    Question: does doubling the micro-batch again buy throughput, and does it
    fit with enough margin for a 9-hour run? Estimate says activations roughly
    double (both MLP and attention scale with batch here), so ~38GB -> ~66GB
    against an 80GB card — and VRAM CREPT 38.3 -> 44.8GB across the v3 run
    from compile buffers and allocator fragmentation, so a tight fit at step 0
    is not a safe fit at step 2,000. Measure instead of guessing: same GPU,
    same corpus, same 30 steps, only batch/accum differ. eff batch stays 64.
    """

    phases: dict = {  # noqa: RUF012
        "extend": {
            "seq_len": 2048,
            "batch_size": 16,
            "grad_accum_steps": 4,
            "datasets": [{"name": "sft_v4_rollout", "weight": 1.0}],
        },
    }
    stage_prefix: str = "sft_v4-sanity-b16"
    wandb_run_name: str = "osrt-v6-sft-v4-sanity-b16"


class SFTv3SanityConfig(SFTv3Config):
    """30-step SFT-v3 probe (run on every FRESH workspace before the paid
    run): corpus present + parses, v6 system/think/answer seq builds,
    midtrain3_final loads clean, VRAM fits at seq 4096."""

    total_steps: int = 30
    warmup_steps: int = 5
    ckpt_interval: int = 9_999_999
    save_final_checkpoint: bool = False
    compile_enabled: bool = False
    stage_prefix: str = "sft_v3-sanity"
    wandb_run_name: str = "osrt-v6-sft-v3-sanity"


class SFTLongConfig(SFTConfig):
    """Long-context SFT — resumes from the seq-2048 SFT checkpoint and
    fine-tunes at seq_len 4096 with a Nemotron-heavy data mix.

    Why: NuminaMath CoT and longform responses cluster in the
    1500-3000 token range and got truncated under the seq 2048 base
    SFT. This phase teaches the model to maintain quality over longer
    completions. Includes Nvidia Nemotron splits (math, stem, code,
    tool_calling) which the base SFT didn't see — Nemotron has
    explicit `reasoning` field that maps directly to our
    `<|think|>{}<|/think|>` block.

    HRA contract:
      - `hra_before_load=True` because the saved SFT ckpt already has
        HRA params in its state_dict; injecting HRA structure first
        lets the load place them correctly.
      - `hra_enabled=True` (inherited) — keeps the existing rank-256
        adapters trained in the base SFT.

    LR contract:
      - Lower peak (5e-6 vs base SFT's 1.5e-5) because we're
        fine-tuning a fine-tune. Aggressive LR risks washing out the
        base SFT learning.
      - Cosine over total_steps=1000 cools to min_lr by the end.
    """

    total_steps: int = 1_000
    warmup_steps: int = 50
    seq_len: int = 4096
    batch_size: int = 4               # halved from 8 to fit longer ctx
    grad_accum_steps: int = 16        # doubled to keep effective batch 64
    peak_lr: float = 5e-6
    min_lr: float = 5e-7
    log_interval: int = 25
    ckpt_interval: int = 250

    # Resume from the base SFT checkpoint with HRA already applied.
    # Points at the explicit step file rather than osrt_v5_sft_final.pt
    # because base SFT was stopped early at step 2500 (Option B budget
    # plan — eval loss already at 1.02 train, no point in chasing the
    # remaining 2500 steps with diminishing returns and a loss curve
    # that's already in the "starting to memorise" zone).
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_sft_step_2500.pt"
    hra_before_load: bool = True
    stage_prefix: str = "sft_long"

    wandb_run_name: str = "osrt-sft-long"

    # Nvidia Nemotron-heavy mix (60%) plus 40% diversity from existing
    # SFT data to prevent over-fitting to one teacher's style.
    datasets: list = [  # noqa: RUF012
        # Nvidia Nemotron Post-Training (60% total)
        {
            "name": "nemotron-math",
            "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": "math",
            "weight": 0.30,
            "format": "nemotron",
        },
        {
            "name": "nemotron-stem",
            "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": "stem",
            "weight": 0.20,
            "format": "nemotron",
        },
        {
            "name": "nemotron-code",
            "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": "code",
            "weight": 0.15,
            "format": "nemotron",
        },
        {
            "name": "nemotron-tool-calling",
            "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": "tool_calling",
            "weight": 0.10,
            "format": "nemotron_tool_calling",
        },
        # Diversity (25%) — long-form reasoning + code from non-Nvidia
        # sources to keep the teacher distribution mixed.
        {
            "name": "numina-math-cot",
            "hf_id": "AI-MO/NuminaMath-CoT",
            "split": "train",
            "weight": 0.10,
            "format": "numina_math",
        },
        {
            "name": "evol-instruct-code",
            "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
            "split": "train",
            "weight": 0.10,
            "format": "evol_code",
        },
        {
            "name": "longform",
            "hf_id": "akoksal/LongForm",
            "split": "train",
            "weight": 0.05,
            "format": "longform",
        },
    ]


class SFTUltraLongConfig(SFTLongConfig):
    """Ultra-long-context SFT — resumes from the seq-4096 SFT-long
    checkpoint and pushes to seq_len 8192.

    Why: NuminaMath multi-page derivations and multi-file code
    generations exceed the 4096 context window. Pushing to 8192
    teaches the model to maintain coherence across genuinely long
    completions before GRPO (which runs at seq_len 8192).

    Compute cost: attention is O(N²) so seq 8192 vs 4096 is 4× the
    attention compute per token. With 4× more tokens per microbatch
    too (4096 → 8192 plus same effective batch shape), this is the
    most expensive SFT phase — kept short (200 steps) to fit budget.
    Trimmed from the original 500 once it became clear that SFT-long
    delivers most of the long-context adaptation; ultralong is a
    polish pass to anchor seq 8192 behaviour before GRPO, not a
    full curriculum stage.

    HRA contract:
      - hra_before_load=True (inherited) — sft_long_final.pt has HRA
        params already.
      - Same HRA rank 256 — keeps continuity with the previous SFT
        passes' learned adaptations.

    LR contract:
      - Even cooler peak (3e-6) — third successive fine-tune, the
        adapters are well-formed and need light polish, not heavy
        re-shaping.
      - Cosine over total_steps=200; warmup scaled down to 10 steps
        (5 % of total) to match the shorter horizon. Without this the
        old 25-step warmup would consume 12.5 % of the run before any
        cosine taper — too much.
    """

    total_steps: int = 200
    warmup_steps: int = 10
    seq_len: int = 8192
    # Halve batch_size again, double grad_accum_steps to keep effective
    # batch at 64 sequences per gradient step. Memory budget at 80 GB
    # H100: ~50-60 GB for activations at seq 8192 + batch 2 (estimated;
    # if it OOMs, drop batch to 1 and accum to 64).
    batch_size: int = 2
    grad_accum_steps: int = 32
    peak_lr: float = 3e-6
    min_lr: float = 3e-7
    log_interval: int = 10
    ckpt_interval: int = 50

    # Points at the explicit step file rather than osrt_v5_sft_long_final.pt
    # because SFT-long was stopped early at step 500 (budget-driven cut to
    # preserve compute for SFT-ultralong + GRPO; final loss 1.32 vs the
    # ~1.10 a full 1000-step run would have hit). The step-500 ckpt has the
    # same HRA contract as the final would have — no functional difference.
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_sft_long_step_500.pt"
    stage_prefix: str = "sft_ultralong"
    wandb_run_name: str = "osrt-sft-ultralong"

    # Same Nemotron-heavy + diversity mix as SFTLongConfig (inherited
    # from `datasets`). The mix doesn't change between SFT-long and
    # SFT-ultralong — only the context window does.


class SFTRefreshConfig(SFTConfig):
    """Short SFT pass to re-anchor chat format after pretrain_extend.

    Why this exists
    ───────────────
    Local probe on osrt_v5_extend_final.pt (post-pretrain-extend) showed
    chat-format degradation despite the 25 % rehearsal mix during
    extend. Symptoms:

      * Special tokens still emitted but in wrong positions —
        "<|think|>on<|/think|><|answer|><think>... reasoning ..."
        with <|/answer|> never closed. lm-eval's answer extractor
        sees broken/missing structure and returns [invalid].
      * Some prompts produce immediate <|end_of_text|> with no
        content at all.
      * Math content is genuinely improving (model decomposes
        17×23 as 17×20 + 17×3 — correct method, wrong arithmetic
        execution) — but the format wrapping is broken.

    Tool-call hallucination is FIXED by extend (no tool_calling data
    seen for 2,800 steps) — confirmed by probe. So this refresh
    deliberately keeps tool_calling OUT of the mix to preserve that
    win.

    Design
    ──────
    Very short (500 steps) at very low LR (5e-6, 33 % of SFTConfig's
    1.5e-5). Goal is to refine where the model places its existing
    chat tags, not to reshape what it knows about math/code. The
    extend gave us a genuinely better base; this just re-anchors the
    SFT format on top of it.

    HRA stays trainable here (hra_freeze_pretrained=False, inherited)
    so the adapters can re-tune toward the new pretrain-extended
    base. The frozen-HRA mode used in pretrain_extend is the opposite
    direction — preserve HRA's old SFT learning while base absorbed
    new content. Now that base has new content, HRA needs to adapt.

    Data mix: 70 % Nemotron post-training (math/stem/code, NO
    tool_calling) + 30 % SFTConfig diversity, all chat-formatted with
    response-only loss masking (standard SFT behaviour from
    sft_data.py).

    Expected outcome: clean <|think|>...<|/think|><|answer|>...
    <|/answer|> emission again, math reasoning improvements
    preserved. Should restore extraction validity for eval.
    """

    # Trimmed from 500 → 200 steps after first launch attempt
    # (sft_refresh run 1) ran at 106 sec/step due to cold HF cache on
    # 4 of 7 datasets (NuminaMath, Evol-Code, OpenHermes,
    # IFEval-like all new on the codhe-hugging-mcp workspace).
    # Single-threaded SFT loader + cold cache = GPU idle waiting on
    # HF Hub fetches every step. Killed and re-scoped to use only
    # already-cached Nemotron splits (math + stem + code, all warm
    # from pretrain_extend's rehearsal mix). Format anchoring needs
    # exposure, not deep content; 200 steps × ~6 sec/step ≈ 20 min ≈
    # $1.30, well within remaining $2.70 budget.
    total_steps: int = 200
    warmup_steps: int = 10            # 5 % of new total
    seq_len: int = 2048
    batch_size: int = 8
    grad_accum_steps: int = 8
    peak_lr: float = 5e-6
    min_lr: float = 5e-7
    log_interval: int = 10            # tighter logs for the short run
    ckpt_interval: int = 50

    # HRA: keep trainable so adapters re-tune to the new base.
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_scale: float = 1.0
    hra_lr: float = 2.5e-5            # 33 % of base SFT hra_lr (7.5e-5)
    hra_freeze_pretrained: bool = False
    hra_before_load: bool = True      # extend ckpt has HRA params

    # Resume from the post-extend checkpoint.
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_extend_final.pt"
    stage_prefix: str = "sft_refresh"
    wandb_run_name: str = "osrt-sft-refresh"

    # Diverse mix favouring SHORT examples that pack efficiently at
    # seq 2048. Reverted from the Nemotron-only experiment (sft_refresh
    # runs 1+4 on codhe-hugging-mcp) where most Nemotron examples
    # exceeded seq_len and were skipped by SFTStream's length filter,
    # starving the packing buffer to ~100+ sec/step.
    #
    # This mix mirrors most of the original SFTConfig — all proven to
    # stream cleanly at 6 sec/step on gradio-winter-hack workspace
    # during base SFT and SFT-long. Critical: NO tool_calling — that
    # habit was eliminated during pretrain_extend, keep it gone.
    datasets: list = [  # noqa: RUF012
        # Math (35 %) — short word problems pack well at seq 2048
        {
            "name": "gsm8k",
            "hf_id": "openai/gsm8k",
            "hf_config": "main",
            "split": "train",
            "weight": 0.15,
            "format": "gsm8k",
        },
        {
            "name": "orca-math",
            "hf_id": "microsoft/orca-math-word-problems-200k",
            "split": "train",
            "weight": 0.10,
            "format": "orca_math",
        },
        {
            "name": "math-instruct",
            "hf_id": "TIGER-Lab/MathInstruct",
            "split": "train",
            "weight": 0.10,
            "format": "math_instruct",
        },
        # Code (20 %) — Alpaca-style code is short
        {
            "name": "evol-instruct-code",
            "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
            "split": "train",
            "weight": 0.10,
            "format": "evol_code",
        },
        {
            "name": "code-instructions-122k",
            "hf_id": "TokenBender/code_instructions_122k_alpaca_style",
            "split": "train",
            "weight": 0.10,
            "format": "alpaca_code",
        },
        # General (30 %) — Alpaca + OpenHermes for chat-format anchor
        # diversity beyond pure math/code (probe revealed cats /
        # planet-question prompts triggered model failure modes
        # because they're outside the math distribution).
        {
            "name": "alpaca-cleaned",
            "hf_id": "yahma/alpaca-cleaned",
            "split": "train",
            "weight": 0.15,
            "format": "alpaca",
        },
        {
            "name": "openhermes",
            "hf_id": "teknium/OpenHermes-2.5",
            "split": "train",
            "weight": 0.15,
            "format": "openhermes",
        },
        # Instruction following (15 %)
        {
            "name": "ifeval-like",
            "hf_id": "argilla/ifeval-like-data",
            "hf_config": "filtered",
            "split": "train",
            "weight": 0.10,
            "format": "ifeval",
        },
        {
            "name": "longform",
            "hf_id": "akoksal/LongForm",
            "split": "train",
            "weight": 0.05,
            "format": "longform",
        },
    ]


class SFTMathConfig(SFTRefreshConfig):
    """Math-focused SFT pass between sft_refresh and GRPO.

    Why this exists
    ───────────────
    Math probe of osrt_v5_sft_refresh_final.pt (commit fa5c69a)
    revealed a specific failure mode: 8/8 clean format ✓, but 6/8
    math problems wrong because the answer block is *decoupled*
    from the think block. Examples:

      24-7-3:   think="24-7=15, 15-3=12" (correct steps)
                answer="15+12=27" (synthesis broken)
      6×8:      think="48" (correct)
                answer="(6+1)*(8+1)=5*9=45" (random wrong)

    The model learned the structure but doesn't commit the think
    block's conclusion to the answer block. A short math-only SFT
    pass hammers in (q → think → answer) examples where the answer
    IS the think conclusion, training the answer block to draw on
    its own reasoning.

    This is the "warm-up before GRPO" — gives RL a base where the
    think→answer pipeline is already coherent, so RL only has to
    optimise correctness, not also fix the decoupling.

    Design
    ──────
    200 steps × seq 2048 × batch 8 × accum 8 = ~26 M trained tokens.
    Peak LR 3e-6 (lower than refresh's 5e-6 — we just refreshed,
    don't want to disturb the format gain). All 4 datasets are
    short-example math (warm-cached on gradio-winter-hack), so
    packing rate is high.

    Resume from osrt_v5_sft_refresh_final.pt (HRA already injected
    + trained, both base and HRA stay trainable like sft_refresh).
    """

    total_steps: int = 1_000
    warmup_steps: int = 50          # 5 % of total
    peak_lr: float = 3e-6           # lower than refresh's 5e-6
    min_lr: float = 3e-7
    log_interval: int = 25
    ckpt_interval: int = 200        # 5 ckpts over 1,000 steps

    # HRA inherited as trainable. Resume from the freshly-format-
    # anchored ckpt (sft_refresh_final.pt has HRA params in its
    # state_dict).
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_sft_refresh_final.pt"
    stage_prefix: str = "sft_math"
    wandb_run_name: str = "osrt-sft-math"

    # Pure math mix — short examples that pack at seq 2048 cleanly.
    # All 4 datasets are warm-cached on gradio-winter-hack from
    # base SFT and sft_refresh. NuminaMath has some long examples
    # but at 15 % weight the long ones being filtered won't starve
    # the buffer (dominated by short GSM8K/Orca-Math/MathInstruct).
    datasets: list = [  # noqa: RUF012
        # GSM8K — gold-standard short math word problems
        {
            "name": "gsm8k",
            "hf_id": "openai/gsm8k",
            "hf_config": "main",
            "split": "train",
            "weight": 0.30,
            "format": "gsm8k",
        },
        # Orca-Math — large variety of word problems
        {
            "name": "orca-math",
            "hf_id": "microsoft/orca-math-word-problems-200k",
            "split": "train",
            "weight": 0.30,
            "format": "orca_math",
        },
        # MathInstruct — algebra/calculus heavy
        {
            "name": "math-instruct",
            "hf_id": "TIGER-Lab/MathInstruct",
            "split": "train",
            "weight": 0.25,
            "format": "math_instruct",
        },
        # NuminaMath-CoT — premium chain-of-thought math
        {
            "name": "numina-math-cot",
            "hf_id": "AI-MO/NuminaMath-CoT",
            "split": "train",
            "weight": 0.15,
            "format": "numina_math",
        },
    ]


class GRPOConfig:
    """GRPO config for v5 — verifiable math rewards on top of SFT."""

    batch_size: int = 4
    grad_accum_steps: int = 4
    # Cut from 2000 → 500 after GRPO run 1 measured ~100 sec/step at the
    # original config (2000 steps × 100 sec = 56h ≈ $220, way over the
    # $30 workspace budget). Run 1's first 10 steps showed acc 4.7% →
    # 26.6% — convergence is fast on the math-anchored base, so 500
    # steps captures most of the curve.
    # Extended from 500 → 700 after run 5 completed: probe of
    # grpo_final.pt showed format/structure consolidation but acc
    # plateaued at ~3/8 on hand probe. Cooling cosine tail (LR
    # 1.5e-7) hadn't fully consolidated; extra 200 steps at very
    # low LR may push the volatile peak band (15-28%) toward
    # sustained acc. Resume from step_500 (renamed from final.pt
    # on the volume so the scan picks it up).
    # Extended from 700 → 800 after the 501-700 run: mean acc rose
    # to ~14.5% (vs ~9% in 201-500) with peaks at 21.9%, 43.8%,
    # 21.9%, 34.4%. Trajectory still upward, but the cosine tail
    # had already cooled to near-zero by step 700. To get real
    # gradient on the new 100 steps, re-warm via lr_anchor_step:
    # the schedule treats `step - lr_anchor_step` as effective
    # step, so warmup runs over steps 700-720 and cosine cools
    # 720→800. peak_lr held at 1.5e-6, kl_coeff bumped 0.15 →
    # 0.20 because the policy has drifted further from the SFT
    # reference over 700 steps.
    total_steps: int = 800
    warmup_steps: int = 20          # short re-warm over steps 700-720
    # Steps elapsed before this LR phase. The warmup/cosine schedule
    # uses `step - lr_anchor_step` so re-warming after a resume
    # gives real gradient instead of the near-zero LR a continued
    # cosine would yield. Default 0 = no anchor (fresh run).
    lr_anchor_step: int = 700
    # Cut from 3e-6 → 1.5e-6 after GRPO run 2 collapsed at step 20.
    # Run 2 trace: acc 12.5 % (0) → 18.8 % (10) → 0 % (20, 30) → 3.1 %
    # (40) → 0 % (50). Only 2 of 6 logged steps gave learning signal;
    # the rest were stuck in GRPO's "all-rollouts-uniform-rewards" trap
    # where group-relative advantage normalisation produces zero
    # gradient. Smaller per-step updates (lower lr) prevent the early
    # drift that pushed the policy into a 0-reward region.
    peak_lr: float = 1.5e-6
    min_lr: float = 1.5e-7
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 10
    ckpt_interval: int = 100        # 5 ckpts over 500 steps (was 250)
    seq_len: int = 8192

    # GRPO-specific
    # group_size halved from 16 → 8 to halve the per-step rollout cost.
    # 8 rollouts per prompt is still enough for meaningful intra-prompt
    # advantage normalisation (DeepSeek-R1's "G" hyperparameter was 16
    # but their model was much larger; for ours, 8 gives plenty of
    # signal).
    group_size: int = 8
    # max_gen_len cut from 512 → 384 to reduce per-rollout token cost.
    # gsm8k-style answers (think + answer block) typically fit in
    # 200-300 tokens; 384 leaves comfortable headroom for the longer
    # multi-step problems while shaving generation time.
    max_gen_len: int = 384
    # Bumped from 0.8 → 1.0 to increase rollout diversity. With
    # group_size=8, we need at least one of the 8 rollouts to land
    # on a correct answer to get any advantage signal. Higher
    # temperature → wider sampling → better chance one rollout in a
    # group hits the right answer even on harder problems. Counters
    # the "all 8 rollouts wrong → zero signal" failure mode from
    # run 2.
    temperature: float = 1.0
    top_p: float = 0.95
    # Bumped from 0.05 → 0.15 to anchor the policy harder against the
    # frozen reference (sft_math_final.pt). Run 2 hit GRPO collapse
    # at step 20 because early policy drift (KL hit 0.017 at step 10)
    # pushed the model into a region where most batches gave 0 % acc
    # → uniform rewards → zero advantage → frozen updates. Stronger
    # KL coefficient prevents that drift in the first place.
    # Bumped 0.15 → 0.20 for the 700→800 re-warm extension. Reference
    # is still the original SFT baseline, but the policy has moved
    # 700 steps further from it; tighter KL keeps the re-warmed
    # update from over-shooting.
    kl_coeff: float = 0.20
    # Tighter PPO clip (0.2 → 0.15) further limits per-step policy
    # change, complementing the lower lr and higher kl_coeff.
    clip_range: float = 0.15

    # HRA (inherited on top of SFT-injected HRA weights)
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_lr: float = 1.5e-5
    hra_before_load: bool = True

    # Rewards
    correctness_reward: float = 1.0
    format_reward: float = 0.2
    reasoning_bonus: float = 0.3
    truncation_penalty: float = -0.5
    empty_think_penalty: float = -0.1
    length_penalty: float = 0.0

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "osrt"
    wandb_run_name: str = "osrt-grpo"
    wandb_run_id: str = ""

    # Checkpoint
    # Updated lineage: pretrain → SFT base → SFT-long → SFT-ultralong
    # → pretrain_extend → sft_refresh → sft_math → GRPO. The
    # `osrt_v5_sft_final.pt` filename never existed (base SFT was
    # stopped at step 2500, see SFTConfig docstring); originally
    # GRPO would have resumed from osrt_v5_sft_step_2500.pt. Now
    # that we have the math-focused chain, GRPO should resume from
    # osrt_v5_sft_math_final.pt — the model with cleaned format
    # (sft_refresh) AND tightened think→answer correlation
    # (sft_math).
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_sft_math_final.pt"
    stage_prefix: str = "grpo"

    # Prompt source
    prompt_dataset: str = "openai/gsm8k"
    prompt_config: str = "main"
    prompt_split: str = "train"

    # Chat format (native single-token tags, same as v4/SFT)
    user_tag: str = "<|user|>"
    assistant_tag: str = "<|assistant|>"
    think_open: str = "<|think|>"
    think_close: str = "<|/think|>"
    answer_open: str = "<|answer|>"
    answer_close: str = "<|/answer|>"


class MultiEnvGRPOConfig(GRPOConfig):
    """Multi-environment GRPO from mopd_final.pt.

    Trains across verifiable-reward environments instead of gsm8k-only,
    mirroring Nemotron 3 Ultra's RLVR design. Per micro-batch we sample
    one environment (weighted), so a full optimizer step sees a mix.

    Environments (V1)
    ─────────────────
    1. math (gsm8k)              weight 0.60
         prompts:  gsm8k train, "#### N" ground truth
         reward:   compose_template_rewards(answer_check=True)
                   → exact_format + per-tag + tiered numeric + strict

    2. ifeval (instruction)      weight 0.30
         prompts:  google/IFEval train (~500 verifiable constraints)
         reward:   compose_template_rewards(answer_check=False)
                   + ifeval_constraint_reward (regex-based check)

    3. mbpp_code                 weight 0.10
         prompts:  google-research-datasets/mbpp train (374 problems)
         reward:   compose_template_rewards(answer_check=False)
                   + mbpp_test_reward (exec test_list in subprocess)
         NOTE: subprocess exec adds ~1-3s per rollout. With group_size=8
         and 10% env weight, only ~5% of total rollouts trigger exec.
         Set env_weights[2]=0.0 to disable for V1 if rollout time
         becomes the bottleneck.

    Schedule
    ────────
    1500 steps from mopd_final.pt. Peak LR 5e-6 → cosine to 5e-7.
    aux_loop_loss_weight 0.03 (preserve depth without dominating
    policy gradient). KL coeff 0.05 (lower than math-only GRPO's
    0.20 — multi-env mix is self-stabilising, want bigger updates).

    Cost estimate
    ─────────────
    ~14 sec/step × 1500 = ~5.8 hr ≈ $22. Fits in the ~$34 cross-
    workspace budget. Extend to 2000 if EMA reward is still climbing
    at step 1200 and budget allows.

    Stop tokens
    ───────────
    During rollout generation we pass `stop_token_ids=[10, 11]` to
    halt cleanly at `<|/answer|>` or `<|user|>`. This prevents the
    multi-answer-block failure mode MOPD revealed (model emits a
    clean answer block then continues with more answer blocks). Stop
    means group rewards are computed on tightly-scoped completions.
    """

    # GRPO v2: fresh from mopd_final.pt with HRA-only + anti-hacking
    # defences. Step 75→150 in v1 regressed inference 4/6 → 2/6 with
    # both knob configs (tighter and original) — base weights drifted,
    # capabilities lost. v2 freezes base, only adapts the rank-256 HRA.
    total_steps: int = 150
    lr_anchor_step: int = 0  # fresh schedule from mopd_final
    warmup_steps: int = 15
    peak_lr: float = 5e-6
    min_lr: float = 5e-7

    # ORIGINAL architecture-fix knobs (these worked best in v1).
    aux_loop_loss_weight: float = 0.03
    loop_dropout_prob: float = 0.05
    loop_dropout_min_loops: int = 3
    per_loop_aux_weights: None = None

    # GRPO sampling
    group_size: int = 8
    max_gen_len: int = 384
    temperature: float = 1.0
    top_p: float = 0.95
    # kl_coeff bumped 0.05 → 0.15 for v2. v1 used 0.05 (loose anchor)
    # and the policy drifted hard; per-step KL was running 0.10-0.20.
    # 0.15 matches the math-only GRPO setting that completed 800 steps
    # without collapse. With HRA-only this is double-safety: base is
    # frozen AND the adapter contribution is anchored to ref.
    kl_coeff: float = 0.15
    clip_range: float = 0.20  # PPO default

    # Compose template rewards (Unsloth-style stack). These flow into
    # compose_template_rewards() in rewards.py. Math env uses all of
    # them; ifeval/code envs disable check_answer/check_numbers and
    # add their own env-specific reward on top.
    reward_exact_format: float = 3.0
    reward_approx_format_pos: float = 0.5
    reward_approx_format_neg: float = -1.0
    reward_number_match: float = 1.5
    reward_number_miss: float = -0.5
    reward_strict_template_weight: float = 0.5

    # Stop-token IDs for rollout generation. <|/answer|>=10, <|user|>=11.
    stop_token_ids: tuple[int, ...] = (10, 11)  # noqa: RUF012

    log_interval: int = 10
    ckpt_interval: int = 50  # 3 ckpts over 150 steps

    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_mopd_final.pt"
    # v2 stage_prefix so v2 ckpts don't collide with the v1 step_75/final
    # already on the volume. v1 artifacts preserved.
    stage_prefix: str = "grpo_v2"
    wandb_run_name: str = "osrt-grpo-v2"

    # Multi-env registry. Per micro-batch we sample ONE env according
    # to env_weights; group_size rollouts come from that env for the
    # step. Env-specific reward dispatch happens in the GRPO training
    # loop (see app.py::grpo_multi when wired). To ablate, set a
    # weight to 0.0.
    env_names: tuple[str, ...] = ("math", "ifeval", "mbpp_code")  # noqa: RUF012
    env_weights: tuple[float, ...] = (0.60, 0.30, 0.10)  # noqa: RUF012

    # Per-env prompt dataset registry. Each entry: how to load the
    # dataset + which fields hold the prompt and ground truth + how to
    # interpret the ground truth (gsm8k_hash | ifeval_constraints |
    # mbpp_tests). The reward dispatcher uses ground_truth_format to
    # pick which env-specific reward to add on top of the shared
    # format rewards.
    env_datasets: dict = {  # noqa: RUF012
        "math": {
            "hf_id": "openai/gsm8k",
            "hf_config": "main",
            "split": "train",
            "prompt_field": "question",
            "gt_field": "answer",
            "ground_truth_format": "gsm8k_hash",
        },
        "ifeval": {
            "hf_id": "google/IFEval",
            "hf_config": None,
            "split": "train",
            "prompt_field": "prompt",
            "gt_field": None,  # constraints live in separate fields
            "ground_truth_format": "ifeval_constraints",
        },
        "mbpp_code": {
            "hf_id": "google-research-datasets/mbpp",
            "hf_config": "full",
            "split": "train",
            "prompt_field": "text",
            "gt_field": "test_list",
            "ground_truth_format": "mbpp_tests",
        },
    }

    # ── HRA-only training (preserves base weights, only adapters update) ──
    # Closes the capability-regression failure mode that hit at step 76+:
    # base weight drift under policy-gradient pressure was costing us
    # capabilities the MOPD distillation built. With hra_only_training=True
    # the 363M base weights are frozen; only the 86M HRA adapters get
    # gradient updates. Equivalent to the standard "LoRA-only RL" pattern
    # (DPO/PPO/GRPO commonly do this).
    #
    # Benefits:
    #   - MOPD/SFT capability anchor is structurally preserved
    #   - KL drift bounded by construction (base contribution to logits
    #     stays fixed; only the additive adapter contribution shifts)
    #   - ~4× fewer params getting Adam state → faster + less memory
    #   - Lower risk of catastrophic forgetting
    #
    # Trade-offs:
    #   - Slower adaptation (smaller effective parameter capacity)
    #   - Some capabilities may not be reachable through the rank-256
    #     low-rank delta alone
    #   - If the model genuinely needs base-weight surgery (e.g. a new
    #     reasoning circuit), HRA-only can't deliver it
    #
    # ON by default for v2 (was added after the step 75→150 regression
    # in v1; this is the central architectural fix for v2).
    hra_only_training: bool = True

    # ── Troubleshoot generation (every N steps) ──
    # Prints a sample completion at the TRAINING temperature so we can
    # eyeball what rollouts actually look like during the run. Different
    # from ood_probe (which uses low temp for deterministic eval).
    # Useful for catching reward-hacking patterns visually — e.g. the
    # model emitting "I tried 50, then 32, but 18" hedge would jump
    # out here while reward EMA looked fine.
    troubleshoot_gen_interval: int = 10
    troubleshoot_gen_prompt: str = "What is 17 * 23?"
    troubleshoot_gen_max_new_tokens: int = 200

    # ── Anti-hacking knobs (added after the step 75→150 regression) ──
    # Strict numeric extraction closes the "last-number-wins" loophole
    # where the model dumps multiple candidate numbers in the answer
    # block hoping the last one matches GT. With strict_extraction=True,
    # check_answer_score uses extract_numeric_answer_strict which only
    # awards positive reward on single-number / boxed / concluding-phrase
    # answers. Ambiguous answer blocks (multiple unmarked numbers) get
    # `ambiguous_penalty` instead of a free 0.
    strict_answer_extraction: bool = True
    ambiguous_answer_penalty: float = -0.5

    # ── OOD probe: in-loop generalization monitor ──
    # Periodically run a held-out set of prompts the model is NOT
    # training on. If train reward EMA climbs but OOD score drops →
    # reward hacking, stop the run. Lets us catch the failure mode
    # the step 75→150 regression revealed (in-distribution gsm8k
    # rollout reward climbed; out-of-distribution direct-arithmetic
    # inference dropped) DURING training instead of post-hoc.
    ood_probe_interval: int = 25  # every N optimizer steps
    ood_probe_temperature: float = 0.3  # low temp for deterministic eval
    ood_probe_max_new_tokens: int = 200
    # 12 prompts spanning styles NOT in the training distribution:
    #   - direct arithmetic (NOT gsm8k word-problem style)
    #   - comparison
    #   - conversion
    #   - simple instruction following (NOT IFEval-style constraints)
    #   - short code (NOT mbpp test_list style)
    # Each has an expected_answer pattern that must appear in the
    # FIRST answer block. Scored as fraction-of-prompts-correct.
    ood_probe_prompts: tuple = (  # noqa: RUF012
        ("What is 23 + 14?", "37"),
        ("What is 17 * 23?", "391"),
        ("What is 12 + 8 * 3?", "36"),
        ("Compute 24 - 7 - 3.", "14"),
        ("Which is bigger: 0.9 or 0.11?", "0.9"),
        ("How many seconds are in 3 minutes?", "180"),
        ("Convert 100 centimeters to meters.", "1"),
        ("What is half of 50?", "25"),
        ("If a train travels 60 mph for 2 hours, how far does it go?",
         "120"),
        ("Round 3.7 to the nearest integer.", "4"),
        ("What is the square root of 64?", "8"),
        ("Count: how many letters are in 'banana'?", "6"),
    )


class GRPOv6Config(GRPOConfig):
    """v6 GRPO from the SFT-v4 checkpoint soup. Every knob below that differs
    from GRPOConfig was set by MEASUREMENT, not by inheriting a default —
    see the probes in app.py (grpo_signal_probe, strict_extraction_probe).

    FULL-PARAMETER, not HRA-only. The HRA adapter is a rank-256 parallel bypass
    around ATTENTION only (model.py:1442), so freezing the base would leave
    GRPO unable to touch the MoE experts, the router, mHC or the embeddings —
    where this architecture's capacity lives — and the adapters are already
    fitted through midtrain and SFT. Safety comes instead from peak_lr 1.5e-6
    (inherited; set after a prior collapse), kl_coeff 0.15, checkpoints every
    50 steps and the OOD probe.

    hra_native=True: v6 builds HRA from config so the checkpoint already
    carries adapters_a/adapters_b. inject_hra would add a SECOND set on top of
    the trained ones. The stage skips injection and collects the existing
    adapter tensors by name so they keep the differential hra_lr.

    MEASURED SETTINGS
    -----------------
    temperature 0.4 (NOT the inherited 1.0). Swept 0.2/0.4/0.7/0.9/1.0 on 100
    unseen prompts x 16 rollouts, scored strictly. What matters is within-group
    reward variance, since group-normalised advantage scales with it — a prompt
    at 0/16 or 16/16 gives exactly zero gradient, which is the recorded
    "uniform rewards -> zero advantage -> frozen updates" collapse:
        T     per-rollout  pass@16  gradient-bearing  variance
        0.2   7.9%         32%      32%               4.05
        0.4   6.5%         41%      41%               4.70  <- best
        0.7   5.9%         38%      38%               4.50
        0.9   3.1%         27%      27%               2.59
        1.0   3.2%         34%      34%               2.70  <- old default
    T=0.4 gives **+74% usable gradient** over the inherited 1.0.

    group_size 16 (NOT 8). It was halved to save rollout cost; the decode work
    made rollouts ~50x cheaper, and group size sets the fraction of prompts
    yielding ANY gradient at all.

    num_prompts_per_step 32 with grad_accum 32 => 512 rollouts/step, ~31s/step,
    so ~900 steps fit the budget. Sized for STEP COUNT: 8B models need >=300
    GRPO steps before answers improve, and a 601M needs at least that. Bigger
    waves (128 prompts) would have given only ~240 steps — generation is just
    ~19% of a step, so extra rollout throughput cannot be spent on learning.

    Prompts are DIFFICULTY-SCREENED and UNSEEN. 62-73% of unfiltered prompts
    are dead (0/16) at every temperature, so screening is essential, not
    optional; and SFT-v4 trained on 6,500 of GSM8K-train's 7,473 problems, so
    reusing that split would be RL on memorised solutions — high reward, no
    generalisation, invisible failure. Prompt file is built by
    build_grpo_prompts from orca-math past the rows v4 consumed.
    """

    # ── lineage ──────────────────────────────────────────────────────
    pretrained_checkpoint: str = (
        "/vol/checkpoints/v5/osrt_v5_sft_v4_soup_1200_1400_1600_1800.pt"
    )
    hra_native: bool = True          # v6: never inject_hra
    hra_enabled: bool = True         # adapters exist and stay trainable

    # ── measured rollout settings ────────────────────────────────────
    temperature: float = 0.4
    group_size: int = 16
    grad_accum_steps: int = 32       # prompts per optimiser step
    max_gen_len: int = 512           # responses measure ~357-430 tok

    # ── schedule sized for step count ────────────────────────────────
    total_steps: int = 900
    warmup_steps: int = 30
    lr_anchor_step: int = 0          # fresh cosine, not a resume
    ckpt_interval: int = 50          # 18 ckpts -> sweep accuracy, don't trust
                                     # the last one (the SFT-v4 lesson)

    # ── screened prompt set ──────────────────────────────────────────
    prompt_dataset: str = "json"
    prompt_config: str = ""
    prompt_split: str = "train"
    prompt_data_files: str = "/vol/rollouts/grpo_prompts.jsonl"

    # ── KL anchor: revert the extension-run bump ─────────────────────
    # GRPOConfig carries kl_coeff=0.20, which is the value a 700->800 EXTENSION
    # run raised it to. That bump is documented as harmful: KL pinned at
    # 0.16-0.20 (vs ~0.05 before), mean gsm8k acc 14.5% -> 5.6%, peaks
    # 43.8% -> 25.0%, and the resulting checkpoint was archived as
    # osrt_v5_grpo_step_800_overconstrained.pt. This is a FRESH run from a
    # fresh SFT base, so use the value that actually worked.
    #
    # 2026-08-10: 0.15 -> 0.04, as a MATCHED pair with the temperature fix in
    # grpo_train._seq_logprobs. Rationale: log-probs are now computed on the
    # T=0.4 distribution, which scales the policy term by ~1/T while the KL
    # approximation (~½·log_ratio² for small ratios) scales by ~1/T², so
    # holding beta constant would make the anchor ~1/T = 2.5x stronger
    # RELATIVE to the corrected policy gradient. Preserving the previous
    # balance implies beta_new ~ beta_old x T = 0.06; 0.04 is the DeepSeekMath
    # GRPO figure (arXiv 2402.03300) and sits just below that, deliberately
    # conservative. NOTE: 0.04 is *not* TRL's default — TRL 0.24 (the version
    # this repo locks) defaults beta to 0.0 and skips loading a reference model
    # entirely.
    kl_coeff: float = 0.04

    # ── few-shot reasoning exemplar + anti-echo penalty ──────────────
    # The observed failures are not format failures: the model sets equations up
    # correctly and then botches the arithmetic (250/3200 = 0.5 at step 190;
    # 4025.25/0.45 giving three different answers at step 220), or reaches a
    # value in the trace and submits a different one (computed 40, answered 10
    # at step 250). The existing 1-shot personas all demonstrate FORMAT with
    # trivial sums, so none of them address that. word_problem_verify_1shot
    # demonstrates name-the-unknown -> equation -> solve -> SUBSTITUTE BACK ->
    # answer-consistent-with-working.
    #
    # DISTRIBUTION SHIFT, declared: this trains pi(y | q, exemplar) while the
    # eval personas carry no exemplar, so unexemplified acc_on is the TRANSFER
    # test — the same structure as the hinted-prompt fork (prereg A1.7).
    #
    # few_shot_echo_penalty is deliberately large relative to what echoing can
    # earn. At -3.0: copy-and-correct nets +2.20 vs +5.20 for real work, and
    # copy-and-wrong lands at -3.3, below the worst non-copying tier (-2.3). So
    # echoing is strictly worse than both solving and failing honestly.
    # Measured 0/256 false positives on real step-390 rollouts at n=8..16.
    few_shot_echo_penalty: float = -3.0

    # ── sampling: top_p 1.0, matching TRL ────────────────────────────
    # Inherited 0.95 leaves a residual score-function mismatch that the
    # temperature fix does NOT close: nucleus sampling truncates and
    # RENORMALISES, so rollouts come from pi_{T,top-p} while _seq_logprobs
    # scores them under the untruncated pi_T. TRL's default is 1.0, which is
    # why upstream never hits this. The clean fix is to stop truncating rather
    # than implement differentiable nucleus-masked log-probs; at T=0.4 the
    # distribution is already sharp, so the behavioural cost is small.
    top_p: float = 1.0

    # ── anti-hacking (inherited defaults, restated for visibility) ───
    # WARNING: this flag is currently INERT on the grpo_train path —
    # compute_reward() has no such parameter, so nothing consumes it and
    # extract_numeric_answer_strict() is never called. The "verified LOSSLESS"
    # note below was measured on the SFT soup; the policy has drifted since, so
    # re-probe rather than assume before relying on it.
    strict_answer_extraction: bool = True   # verified LOSSLESS: strict==loose
                                            # at 18.0%, 0.0% ambiguous
    # ── SYSTEM PROMPT — load-bearing, not cosmetic ───────────────────
    # The v5 loop builds "<|user|>{q}<|assistant|>" with NO system block. The
    # SFT model was trained with one, and the system prompt is what carries the
    # think/answer instruction. Without it the model emits NO <|think|> block
    # and dumps its working straight into <|answer|>, so the answer block holds
    # several numbers, the strict extractor rules it `ambiguous`, and every such
    # rollout takes the -0.5 penalty. Observed at step 0: mean reward -0.984
    # with all three sampled rollouts starting "<|answer|>To find the total...".
    # Our 0.0%-ambiguity measurement used this persona; strip it and the whole
    # reward signal inverts.
    system_tag: str = "<|system|>"
    # REVERTED to unexemplified until mixed-stratum support passes the frozen
    # audit. Setting this to word_problem_verify_1shot makes EVERY prompt
    # exemplified, so the exemplified share of advantage mass is necessarily
    # 100% and prereg criterion 3.6 (<=60%) is violated by construction — a
    # ready-to-launch config that silently breaks its own preregistration.
    # The agreed design is a fixed 50/50 PROMPT-LEVEL mixture of
    # word_problem_verify_0shot / _1shot with every rollout group
    # persona-consistent, which needs per-prompt persona metadata that
    # generate_rollouts does not yet carry.
    system_persona: str = "minimal_format"
                                             # every eval/probe used

    # ── in-flight visibility ─────────────────────────────────────────
    # Print real rollouts periodically: scalars cannot show that the text has
    # gone degenerate, and reward hacking looks like a healthy curve over
    # rubbish output.
    sample_print_interval: int = 10
    # Held-out GSM8K accuracy is the ONLY trustworthy judge. Train reward can
    # climb while real accuracy does not — SFT-v4 showed loss and accuracy
    # dissociating three separate times.
    heldout_eval_interval: int = 50
    heldout_eval_n: int = 50

    # ── fresh lineage, so a restart cannot silently resume the old run ─
    # _latest_local() scans --ckpt-dir for <stage_prefix>_step_*.pt, so keeping
    # "grpo_v6" would let a soup restart pick up grpo_v6_step_400.pt instead —
    # defeating the restart and overwriting forensic checkpoints. The wave-2
    # artefacts (steps 100-400) were trained with a mis-specified score function
    # (log-probs at T=1 while sampling at T=0.4) and are diagnostic ONLY; they
    # must never be a parent checkpoint.
    stage_prefix: str = "grpo_v6b"
    wandb_run_name: str = "osrt-v6-grpo-b"


class GRPOv6SanityConfig(GRPOv6Config):
    """30-step GRPO probe. Measures the thing the rollout probes could NOT:
    training-phase VRAM (policy + grads + Muon/AdamW state + reference copy for
    the KL term) at the real wave size, plus seconds/step. Run before the paid
    run."""

    total_steps: int = 30
    warmup_steps: int = 5
    grad_accum_steps: int = 8        # smaller wave so 30 steps is quick
    ckpt_interval: int = 9_999_999
    stage_prefix: str = "grpo_v6-sanity"
    wandb_run_name: str = "osrt-v6-grpo-sanity"
