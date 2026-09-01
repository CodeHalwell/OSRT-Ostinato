# Runbook — from zero to the first results

Everything below is copy-paste. Four stages, in order; each gates the next.

## 0 · Secrets (once per Modal workspace — yours to do, needs your tokens)

```bash
for ws in agents-of-output danielhalwell build-small codhe-hugging-mcp gradio-winter-hack; do
  MODAL_PROFILE=$ws uv run modal secret create osrt-secrets \
      HF_TOKEN=hf_... WANDB_API_KEY=...
done
```

`scripts/launch_ladder.sh --dry-run` confirms which workspaces have it.

## 1 · Probe the card (Colab, minutes, free) — gate G7 part 1

Open `notebooks/v7_pretrain_colab.ipynb` on the RTX PRO 6000 runtime and run
cells 1–2. You want to see:

```
capability : sm_120
_grouped_mm bf16      OK
_grouped_mm fp8 e4m3  unsupported   <- expected; roadmap §17.1
```

If it reports anything other than `sm_120`, stop — the card is not the one
the plan is sized for.

## 2 · Launch gate (Modal, ~5 min, ~$0.50)

The real 968M shape, 30 steps, hard-capped. Answers what no proxy can: does the
committed config build, fit, compile and step with loss falling.

```bash
uv run modal run app.py --sanity
```

Pass = it finishes and `loss` in the log decreases. Any shape/vocab complaint
here is a real bug; do not proceed past it.

## 3 · The ladder (Modal, ~2h wall-clock in parallel, ~$50 total)

Six arms, one per workspace, detached. G3a, G4, and E1 all at once; E2's
telemetry rides on every arm.

```bash
scripts/launch_ladder.sh
```

| arm | answers | vs |
|---|---|---|
| `a` | G3a control · E1 control | — |
| `b`, `c` | G3a: does total help at fixed active? | `a` |
| `dense` | G3a: are we before the MoE crossover? | `a` |
| `nohra` | E1: do adapters earn their FLOPs? | `a` (iso-compute, exact) |
| `g4` | G4: 4 blocks × 5 loops vs 3 × 6 | `a` (±2%) |

Watch W&B project `osrt`, runs `osrt-v7-ladder-*`.

## 4 · Read

All comparisons are **loss at matched tokens on the FineWeb held-out**, and the
threshold is 0.02 nats — below that is noise at this budget.

| result | meaning | action |
|---|---|---|
| `a` ≈ `b` ≈ `c` | token requirement tracks **active** | §14.8 holds; 968M shape is safe |
| loss *rises* a→b→c | tracks **total** | re-price §14 before the trunk; the shape shrinks |
| `dense` beats all MoE arms | before the crossover (§17.4) | 5.3B tokens is too few for sparsity to pay; revisit budget |
| `nohra` ≈ `a` | adapters inert given grouping | **drop HRA** from v7 (−14.2M params) |
| `nohra` worse by >0.02 | adapters earn their place | keep; report per-loop update-norm profile |
| `g4` beats `a` | MoEUT's G=4 prior holds here too | G4 → 4 blocks; rerun G3a at that shape |

E2 has no pass/fail — it is the trajectory. Plot `loop_hidden_norm_ratio`,
`moe/b*/bias_loop_spread`, and `muon/ortho_err` across all six arms. Flat is
the boring good result; monotonic growth in any of them is the finding.

## 5 · E3, on the first checkpoint that clears sanity

```bash
PYTHONPATH=src uv run python scripts/probe_cross_loop_kv.py --ckpt <path> --out probe.json
uv run python scripts/recommend_loop_count.py probe.json
```

It will say either "recommend running K" with the depth saving, or "no loop is
idle on all three signals — do not trim." On v6 it said the latter.

## What is deliberately NOT here

There is no trunk-run command. `app.py` cannot start one. The trunk is a
separate, explicit decision taken after stage 4 reports, on a box, with
`total_steps` set from what G3a says the yardstick is.
