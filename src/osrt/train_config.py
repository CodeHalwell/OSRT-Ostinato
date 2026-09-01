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
    """Pre-training hyperparameters for v7.

    Budget and schedule are defined ONCE, here, and the phase table below is
    expressed as FRACTIONS of total_steps — so the two cannot disagree. (v6
    carried total_steps=3,500 beside a phase table hard-coded to 300,000
    steps; a default launch trained ~458M tokens, ended inside phase 1, and
    looked normal doing it.)

    The step counter persists across resumes, so a drip-funded run toward a
    FIXED total_steps is one continuous schedule. Change total_steps before the
    first chunk and never mid-run.
    """

    # ── Budget ─────────────────────────────────────────────────────────
    batch_size: int = 8
    grad_accum_steps: int = 8
    # Sized to ~1x Chinchilla on ACTIVE params (263M x 20 ≈ 5.3B tokens) at the
    # per-phase batch economics below: 17,500 steps ≈ 5.28B tokens. This is the
    # §14.8 assumption made operational — G3a decides whether the yardstick is
    # active or total, and if it is total this number must roughly quadruple.
    # `total_tokens()` reports the implied budget; check it, do not infer it.
    _total_steps: int = 17_500
    warmup_steps: int = 400          # ~2%; spins up Muon + the balance bias

    # ── Schedule ───────────────────────────────────────────────────────
    # "wsd" = warmup / stable / decay (roadmap item 0.2). The stable phase is
    # the point: a drip-funded run can stop and resume anywhere in it at zero
    # cost, and only the release branch pays the decay. "cosine" is retained
    # to reproduce v6 runs.
    lr_schedule: str = "wsd"
    # Final decay occupies the last wsd_decay_frac of the run. It is aligned
    # with the "anneal" data phase below on purpose: the LR decay and the
    # high-quality data both belong to the branch, not the trunk (§7.5.2).
    wsd_decay_frac: float = 0.15
    # DataLoader workers. 0 is correct on Colab: HF streaming + BPE inside
    # forked workers leaks semaphores, and on a session-capped runtime a dead
    # worker costs the session. Raise on a dedicated box.
    dataloader_num_workers: int = 0

    # ── Loop-collapse floor (roadmap §17.3) ────────────────────────────
    # Per-effective-layer residual update ||Δx||/||x||. Two independent groups
    # name residual-norm growth under weight-tied recurrence as THE failure
    # mode, and one shows per-layer RMSNorm does not prevent it. A deep loop
    # whose update → 0 has collapsed to a no-op; a loop whose hidden norm runs
    # away is exploding. Both are checked at every eval and fail the run.
    # 0 disables. OSRT's own probe measured a contracting iteration, so the
    # defaults are loose — tighten once a v7 baseline exists.
    min_loop_update_norm: float = 1e-3
    max_loop_hidden_norm_ratio: float = 50.0   # last-loop / first-loop hidden norm
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
    # magnitude is held constant, so it's a clean A/B toggle. Default ON per
    # the §14.1 committed shape (graduated; GLM-5's ablation credits it for
    # Muon+MLA ≥ GQA); the G3 ladder still A/Bs it off.
    per_head_muon: bool = True
    # DeepSeek-V4 Muon recipe (roadmap §14.1 item 1.3 — the one graduated line
    # that was still absent). Hybrid Newton-Schulz: 8 fast iterations for
    # convergence then 2 stabilising (2.0, -1.5, 0.5) passes, with the update
    # RMS rescaled to 0.18 rather than the shape-heuristic default. V4 runs
    # exactly this at 1.6T params (roadmap §12.2, verified against the report).
    muon_ns_steps: int = 8
    muon_ns_stable_steps: int = 2
    muon_update_rms: float = 0.18

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
    # ── Router-health early stop, expressed RELATIVE to the expert count ──
    # These were absolutes tuned at E=8 (min_raw_max_prob=0.30 was "2.4x
    # uniform 1/8"; min_marginal_entropy=1.80 was "87% of ln 8"). At E=28 the
    # same absolutes are wrong in BOTH directions: 0.30 is 8.4x uniform and
    # would kill a healthy top-4 router, while a 0.55-nat entropy drop is only
    # 17% of ln 28 instead of 26% and would miss a collapse. Each is now a
    # fraction of its natural scale — ln(E), 1/E, 1/top_k — and resolved from
    # the model config at check time. The fractions reproduce the v6 absolutes
    # exactly at E=8, top_k=2.
    per_token_entropy_drop_frac: float = 0.265     # of ln(E); 0.55/ln 8
    raw_max_prob_frac_of_topk: float = 0.60        # of 1/top_k; 0.30 at top-2
    top_margin_frac_of_topk: float = 0.20          # of 1/top_k; 0.10 at top-2
    marginal_entropy_frac: float = 0.866           # of ln(E); 1.80/ln 8
    prebias_marginal_entropy_frac: float = 0.745   # of ln(E); 1.55/ln 8
    prebias_expert_fraction_of_uniform: float = 0.08   # of 1/E; 0.01 at E=8
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
    _phase_spec: dict = {  # noqa: RUF012
        "foundation": {
            "frac": 0.05,          # broad, short-seq warm-in
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
            "frac": 0.80,          # the trunk: stable LR, broad mix
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
        "anneal": {
            "frac": 0.15,          # the branch: high-quality data under LR decay
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

    # ── Derived. Phase boundaries come from total_steps; do not hand-edit. ──
    def __init__(self, **overrides) -> None:
        import copy
        # Instance-private copy: the spec is a class attribute, and resolving
        # boundaries in place on a shared dict would leak between configs.
        self.phases = copy.deepcopy(type(self)._phase_spec)
        for k, v in overrides.items():
            if not hasattr(type(self), k) and k != "total_steps":
                raise TypeError(f"unknown PretrainConfig field {k!r}")
            setattr(self, k, v)
        self._resolve_phases()

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @total_steps.setter
    def total_steps(self, value: int) -> None:
        """Changing the horizon re-derives every phase boundary, so a caller
        that sets cfg.total_steps = N after construction still gets a
        consistent table (the ladder and sanity stages do exactly this)."""
        self._total_steps = int(value)
        if hasattr(self, "phases"):
            self._resolve_phases()

    def _resolve_phases(self) -> None:
        """Turn phase fractions into absolute step boundaries.

        Fail closed on inconsistency: v6 shipped total_steps=3,500 beside a
        phase table ending at 300,000, and nothing complained.
        """
        fracs = [ph.get("frac") for ph in self.phases.values()]
        if any(f is None for f in fracs):
            raise ValueError("every phase needs a 'frac' (fraction of total_steps)")
        total = sum(fracs)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"phase fractions must sum to 1.0, got {total:.6f}")
        cursor = 0
        names = list(self.phases)
        for k, name in enumerate(names):
            ph = self.phases[name]
            end = (self.total_steps if k == len(names) - 1
                   else cursor + round(ph["frac"] * self.total_steps))
            ph["start"], ph["end"] = cursor, end
            cursor = end

    def validate(self) -> None:
        """Cross-field checks, run once at train start rather than on every
        setter so callers may assign total_steps and warmup_steps in any
        order (the sanity and ladder stages set steps first)."""
        # NOTE: no "warmup must end inside phase 1" check. Warmup is an LR
        # concept and phases are a data/seq_len concept; they are orthogonal,
        # and tying them makes every short run (sanity at 30-200 steps, where
        # phase 1 is 2-10 steps) fail for no reason.
        if self.warmup_steps >= self.total_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) >= total_steps "
                f"({self.total_steps})")
        decay_start = int(self.total_steps * (1 - self.wsd_decay_frac))
        if self.lr_schedule == "wsd" and decay_start <= self.warmup_steps:
            raise ValueError(
                f"WSD decay would start at step {decay_start}, inside warmup "
                f"({self.warmup_steps}) — no stable phase exists")

    def total_tokens(self) -> int:
        """Implied token budget: sum over phases of steps x batch x accum x seq."""
        n = 0
        for ph in self.phases.values():
            bs = ph.get("batch_size", self.batch_size)
            ga = ph.get("grad_accum_steps", self.grad_accum_steps)
            n += (ph["end"] - ph["start"]) * bs * ga * ph["seq_len"]
        return n

    # Budget note: the schedule is aspirational — the user runs in chunks as
    # Modal credits allow. Checkpoints every 1K steps keep stop/resume cheap.
    # Any early stopping still leaves a usable model for SFT.


class V7SanityConfig(PretrainConfig):
    """The launch gate: a hard-capped run of the real committed shape.

    Exists as a class so it cannot be mistaken for a trunk recipe — nothing in
    it is tuned, it just has to build, fit, compile and step with loss falling.
    """
    _total_steps: int = 30
    warmup_steps: int = 5
    wsd_decay_frac: float = 0.3
    ckpt_interval: int = 10
    eval_interval: int = 10_000        # never, inside 30 steps
    save_final_checkpoint: bool = False
    wandb_log: bool = False
