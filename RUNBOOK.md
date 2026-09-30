# Runbook — run it

The design is committed (roadmap §14, §16, §19). No ladder, no launch gate:
the health checks run *during* the run instead of before it. §19 lists
every bet and what would falsify it — read results against that.

## 1 · Secrets

**Already there.** Four workspaces carry `hf-secret` (HF_TOKEN) and
`wandb-secret` (WANDB_API_KEY) from v6 — danielhalwell, build-small,
codhe-hugging-mcp, gradio-winter-hack — and the launcher uses those four.
`agents-of-output` has neither; to use it too:

```bash
MODAL_PROFILE=agents-of-output uv run modal secret create hf-secret HF_TOKEN=hf_...
MODAL_PROFILE=agents-of-output uv run modal secret create wandb-secret WANDB_API_KEY=...
```

**Colab:** `HF_TOKEN` (write access to your checkpoint repo) and
`WANDB_API_KEY` in the Secrets panel. The push daemon probes write access at
start and refuses to train on a read-only token.

## 2 · Run

**Colab — RTX PRO 6000, 96GB, free, session-capped.** Open
`notebooks/v7_pretrain_colab.ipynb`, set `HF_CKPT_REPO` to a private repo you
own, run top to bottom. `train_main` sizes the micro-batch from the card
(96 GB → `--micro-batch-scale 0.5`, tokens/step unchanged). When the session
dies, run it again: it pulls the newest checkpoint, restores the data-stream
position and the W&B run id it left in the checkpoint directory, and
continues. That is the whole resume story.

**Modal — B200, metered, no session cap.**

```bash
uv run modal run --detach app.py --trunk-run                 # volume-resumable
uv run modal run --detach app.py --trunk-run --hf-repo HallD/osrt-v7-ckpt   # ...and mirrored to HF
```

The trunk needs ~45 h at ~33K tok/s; Modal's ceiling is 24 h. The function
saves a rescue checkpoint at 23 h and **re-spawns itself** until the run
completes, keeping one W&B run (`/vol/trunk/wandb_run_id.txt`). Launch it from
ONE workspace only: volumes are per workspace, and two live invocations on the
same volume would both write `osrt_step_N.pt`. The two venues share the HF
repo, so a run can move between them.

Budget: `PretrainConfig` — 18,000 steps ≈ **5.43B tokens** (16×2048 / 6×4096 /
2×8192 micro-batches on B200 — roadmap §13b), ~1× Chinchilla on active params.
The first log lines print the exact number and the Muon per-element step.

**Resume is fail-closed.** Every checkpoint stamps the recipe (schedule, Muon
and AdamW LRs, the phase plan with tokens/step, the tokenizer's sha256). A
session that disagrees stops with a diff instead of splicing two runs; a
deliberate change needs `OSRT_ALLOW_RECIPE_DRIFT=1`. A finished run is
recognised (`osrt_step_18000.pt` alias) and re-invoking it does nothing.

## 3 · Watch

The run **ends itself** and names the criterion:

| when | what is checked |
|---|---|
| step 5,000 (one-shot gate) | router sharpening, balance, pre-bias health, bias saturation, loop collapse, residual explosion |
| every 50 steps after warmup | loop collapse, residual explosion |
| every 50 steps after the gate | the whole set above |
| every step | non-finite gradient norm (step skipped; 5 in a row fails the run) |
| on resume | recipe / data-plan / tokenizer drift |

A criterion must fail 3 consecutive checks (150 steps) to stop the run, except
at the gate. Short of that:

| signal | healthy | worry |
|---|---|---|
| `train/task_loss` | falling, spikes recover | flat, or spiking |
| `train/grad_norm` | settles, occasional spikes | growing, or `train/nonfinite_steps` > 0 |
| `moe/dead_experts_total` | 0 | > 0 — at E=28 each is 3.6% of a block |
| `moe/prebias_expert_min_mean` | ≳ 0.5/E | falling — the learned router is collapsing behind the bias |
| `loop/update_norm_l*` | every loop non-trivial | late loops → 0 |
| `loop/hidden_norm_ratio` | ≈ 1–2, flat | rising (§17.3) |
| `moe/b*/bias_loop_spread` | flat | rising — routing diverging across loops |
| `muon/ortho_err` | small, flat | rising — Newton–Schulz not converging |
| `train/dead_sources` | 0 | > 0 — a data source was dropped; the realised mix differs from the plan |

`run_training` returns `complete`, `already_complete`, `early_stop`,
`rescued` (23 h boundary; the trunk re-spawns) or `data_dead` (every data
source failed; a rescue checkpoint was written — fix the source, re-invoke).

## 4 · Read

Roadmap **§19.4**. Three outcomes; only one of them establishes anything
beyond stability, and §19 says which in advance.

## Afterwards, if you want to know *why*

The ladder is still here — `scripts/launch_ladder.sh` runs six arms
(`a b c dense hra g4`) across the Modal workspaces (§18); every arm inherits
the trunk recipe at dim 1024, and `compute_budget.py --arm <name>` prints its
counts. It is how you explain the trunk's result, not a prerequisite for it.
`scripts/recommend_loop_count.py` on any checkpoint tells you how many loops to
run at decode.
