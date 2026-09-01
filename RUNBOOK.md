# Runbook — run it

The design is committed (roadmap §14, §16, §19). No ladder, no launch gate:
the collapse detectors run *during* the run instead of before it. §19 lists
every bet and what would falsify it — read results against that.

## 1 · Secrets (once)

**Colab:** add `HF_TOKEN` (write) and `WANDB_API_KEY` in the Secrets panel.
**Modal:** in whichever workspace will run it:

```bash
uv run modal secret create osrt-secrets HF_TOKEN=hf_... WANDB_API_KEY=...
```

## 2 · Run

**Colab — RTX PRO 6000, 96GB, free, session-capped.** Open
`notebooks/v7_pretrain_colab.ipynb`, set `HF_CKPT_REPO` to a private repo you
own, run top to bottom. When the session dies, run it again: it pulls the
newest checkpoint and continues. That is the whole resume story.

**Modal — H100, metered, no session cap.**

```bash
uv run modal run app.py --trunk-run                          # volume-resumable
uv run modal run app.py --trunk-run --hf-repo HallD/osrt-v7-ckpt   # ...and mirrored to HF
```

The two venues share the HF repo, so a run can move between them.

Budget: `PretrainConfig` — 17,500 steps ≈ **5.28B tokens**, ~1× Chinchilla on
active params. The first log line prints the exact number.

## 3 · Watch

The run **ends itself** on router collapse, loop collapse, residual
explosion, or checkpoint drift, and names the criterion. Short of that:

| signal | healthy | worry |
|---|---|---|
| `loss` | falling, spikes recover | flat, or spiking |
| `moe/dead_experts_total` | 0 | > 0 |
| `loop/update_norm_l*` | every loop non-trivial | late loops → 0 |
| `loop_hidden_norm_ratio` | ≈ 1, flat | rising (§17.3) |
| `moe/b*/bias_loop_spread` | flat | rising |
| `muon/ortho_err` | small, flat | rising |

## 4 · Read

Roadmap **§19.4**. Three outcomes; only one of them establishes anything
beyond stability, and §19 says which in advance.

## Afterwards, if you want to know *why*

The ladder is still here — `scripts/launch_ladder.sh` runs six arms across
the Modal workspaces (§18). It is how you explain the trunk's result, not a
prerequisite for it. `scripts/recommend_loop_count.py` on any checkpoint
tells you how many loops to run at decode.
