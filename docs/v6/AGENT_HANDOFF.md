# OSRT v6 — Agent Handoff: midtrain3 → SFT v2 → GRPO

**Last updated:** 2026-07-06. Written for an agent resuming this work on an
always-on machine. Read this end-to-end before doing anything. The code is
ground truth (`src/osrt/`); this doc explains the *why* and the *how to run*.

---

## 0. TL;DR — what to do right now

The model (OSRT-605M, v6) is **undertrained, not capacity-capped**. The single
highest-value action is **more pretraining** ("midtrain3"), chained across
compute sessions, until it approaches Chinchilla-optimal — then re-run the
(already-built) SFT v2 and GRPO on the stronger base.

**Immediate task:** run/continue **midtrain3** — a 12,600-step continued-pretrain
from `osrt_v5_midtrain2_step_1750.pt`, resuming from checkpoints on the private
HF repo `HallD/osrt-v6-ckpt`. It has produced **no surviving checkpoint yet**
(two Colab attempts were reclaimed before banking one), so it starts from the
base at step 0.

**On an always-on machine with a CUDA GPU, skip Colab entirely** and run
`scripts/lightning_midtrain3.py` directly (see §4A). That removes every failure
mode we hit. If the machine has no GPU, drive Colab (§4B) or Modal (§4C).

---

## 1. What this project is

**OSRT** = Optimized Sparse Recursive Transformer. ~601M physical params /
~278M active. Key architecture (see `CLAUDE.md`, `ARCHITECTURE.md`, `docs/`):
- **3 physical decoder blocks applied 6× via a loop** → 18 effective layers from
  ⅓ the params (depth recurrence).
- Per block: 1 shared + 8 routed experts (top-2) MoE; GQA attention with an
  **MLA-style compressed-latent KV cache (KDV: value derived from the K latent)**;
  rank-256 native HRA adapters; 4-channel mHC residual.
- Trained with **Muon (2D matrices) + AdamW (embeddings/norms/biases)**.
- **This architecture is compute-bound, not bandwidth-bound** (recursion reuses
  weights, grad-checkpointing trades memory for compute, KDV shrinks the KV
  cache). Consequence: it runs well on high-compute/moderate-bandwidth GPUs
  (e.g. RTX PRO 6000 hit ~69% of H100 throughput despite ~half the bandwidth).

**Environment:** Python 3.11, `uv` (uv.lock committed). `PYTHONPATH=src`.
Tests: `uv run pytest` (196 pass). Pretraining requires CUDA (bf16 + compile).

---

## 2. The diagnosis that drives everything (READ THIS)

The v6 lineage went pretrain → midtrain → midtrain2 → SFT v1/v2. **SFT v2**
(from `midtrain2_step_1750`, verified reasoning corpus) produced a model that:
- ✅ has **perfect format** (`format_ok_on = 1.0`: clean `<|think|>…<|/think|>
  <|answer|>…<|/answer|>`, no loop-collapse — the SFT-v1 failure is gone), but
- ❌ scores **~0.04–0.06 on GSM8K** (= SFT v1, statistically flat), and
  generates fluent-but-wrong math ("50/80 = 6.25%", "32 inches in an inch").

**That is the signature of an undertrained base**, not a size ceiling. Token
accounting: the base has seen **~2.2B tokens ≈ 0.4× Chinchilla** for 278M active
params (Chinchilla-optimal ≈ 5.6B). Models this size that actually reason
(Qwen2.5-0.5B ~40% GSM8K) saw **trillions**. SFT/GRPO can only *elicit* latent
capability; they can't *create* it. **The fix is more pretraining.** The user
correctly rejected the "601M ceiling" framing.

**Budget reality:** ~$120/month of free Modal credit (4 × $30 workspaces, can't
pool) + ~550 Google Colab units + now an always-on machine. Reaching 1×
Chinchilla (+3.4B tokens) ≈ 3 months of drip, or faster with Colab/own-GPU
bursts. "Good" (5× Chinchilla) ≈ ~$2.5K / much longer. **Deliverable is a
working architecture + pipeline + as much pretraining as budget allows**, not a
frontier model.

---

## 3. Current state of all artifacts

**The base checkpoint (start midtrain3 from this):**
`osrt_v5_midtrain2_step_1750.pt` — 601,968,828 params, native HRA (18/18
adapters), eval ppl **28.2**. Locations:
- Local Mac: `checkpoints/v5/osrt_v5_midtrain2_step_1750.pt` (4.9GB)
- HF (private): `HallD/osrt-v6-ckpt` → `osrt_v5_midtrain2_step_1750.pt`
- Modal volume `osrt-checkpoints` on workspaces `danielhalwell`,
  `codhe-hugging-mcp`, `gradio-winter-hack` → `v5/osrt_v5_midtrain2_step_1750.pt`
- NOTE: `osrt_v5_midtrain2_final.pt` (step 2000) is CORRUPT (truncated on the
  volume, ~2.4GB, unloadable). Use step_1750 — it's the best intact artifact.

**midtrain3 checkpoints:** none banked yet. When the run works, they appear as
`osrt_v5_midtrain3_step_*.pt` on `HallD/osrt-v6-ckpt` (pushed every 100 steps).

**SFT v2 (built, DEFERRED until after midtrain3):**
- Corpus: `rollouts/sft_v2.jsonl` (gitignored; rebuild with
  `PYTHONPATH=src python scripts/build_sft_v2_data.py`, needs HF_TOKEN). 53,447
  verified records: ON 61% / OFF 22% / CHAT 16%. Sources: OpenR1-Math-220k
  (math_verify-correct only), Bespoke-Stratos-17k, mopd re-verified vs GSM8K
  gold (81.4% correct kept), chat slice. Domain-NEUTRAL personas (0/0
  persona↔domain mismatch). GSM8K-test decontaminated.
- `SFTv2Config`: base = `midtrain2_step_1750`, 1000 steps, peak 1e-5.
- **After midtrain3, RE-RUN SFT v2 from the new base.** The corpus builder now
  appends EOS to rollout targets (was missing → the model never learned to stop;
  fixed in `data.py`, test `tests/test_rollout_eos.py`). So a fresh
  `sft_v2.jsonl` build will teach stopping.

**Modal:** 4 workspaces, ~$30/mo each. Volumes on each: `osrt-checkpoints`,
`osrt-v6-tokenizer` (the 65K v6 tokenizer), `osrt-hf-cache`, `osrt-rollouts`.
Secrets: `hf-secret`, `wandb-secret` (pre-existing on used workspaces).

**W&B:** project `codhe-synextra/osrt`. midtrain3 logs as run name
`osrt-v6-midtrain3`. This is the RELIABLE monitor (see §6).

**Tokenizer:** `v6_tokenizer_export/` (3 files, committed in-repo). 65K vocab.

**Git:** branch `main`, remote `CodeHalwell/OSRT-605M-A269M`. **Push as user
`CodeHalwell`** — run `gh auth switch --user CodeHalwell` before every push (it
reverts to another account otherwise). Do NOT open PRs unless asked. Never
commit weights/tokens (.env is gitignored; tokens have leaked in chat — treat
as rotate-worthy, never echo them into commits).

---

## 4. How to run midtrain3

Config: `MidtrainExtend3Config` in `src/osrt/train_config.py`:
- 12,600-step cosine (+3.4B tok → ~5.6B = 1× Chinchilla), peak_lr 5e-5 → 1e-5,
  muon 1.65e-3, warmup 100, seq 4096, eff-batch 66 (batch 6 × accum 11).
- **eval_interval = 9,999,999 (in-loop eval DISABLED — see §5 gotcha #4).**
- **ckpt_interval = 100** (bank to HF ~1hr in).
- 0.75 reasoning-dense data share (Nemotron math/STEM/reasoning + textbooks).
- Resume-scan: on start, loads the highest `osrt_v5_midtrain3_step_*.pt` in
  `--ckpt-dir`; if none, loads the base. So it CHAINS across sessions.

### 4A. BEST: always-on machine with a CUDA GPU (no Colab, no Modal)
This removes every failure mode. VRAM needed: batch-6 seq-4096 grad-ckpt ≈
~58GB. If <58GB, use `--micro-batch`/`--grad-accum` to fit (keep eff-batch 66,
e.g. `--micro-batch 3 --grad-accum 22` for ~40GB; `--micro-batch 2 --grad-accum
33` for ~28GB).

```bash
cd <repo> && git pull
export HF_TOKEN=... WANDB_API_KEY=...
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # avoids frag OOM
# pull the base + any prior midtrain3 ckpt from HF into ./checkpoints/v5, resume,
# and push each new ckpt back to HF (survives reboots, chains sessions):
PYTHONPATH=src python scripts/lightning_midtrain3.py \
  --ckpt-dir ./checkpoints/v5 --hf-repo HallD/osrt-v6-ckpt --num-workers 0
# (drop --num-workers 0 if this GPU/host doesn't hit the PyGILState crash — see
#  §5 #2; workers give better throughput when stable. Test with --sanity first.)
```
Run the 30-step gate first: add `--sanity`. It must reach step 30 with no
`PyGILState_Release` / OOM. Then launch the full run in `tmux`/`nohup` so it
survives your SSH session. It resumes automatically on any restart.

### 4B. Colab (only while the machine stays on — keep-alive is LOCAL)
Full recipe: `docs/colab_midtrain3.md` and memory `colab_midtrain3_recipe.md`.
Essentials: `colab --auth=adc` (NOT oauth2 — it drops the `colaboratory` scope
on refresh → VM reclaimed), `--gpu G4` (RTX PRO 6000, 96GB — H100 often 503,
A100 is 40GB and too small), `--num-workers 0`, `expandable_segments`, launch a
**detached bootstrap** (nohup script on the VM), and **monitor via W&B/HF, never
`colab exec`** (its websocket is flaky — empty/partial reads). One-time:
`gcloud auth application-default login --scopes=openid,cloud-platform,userinfo.email,colaboratory`.
Keep the machine awake (`caffeinate -dis` on Mac). Colab caps sessions at 24h;
resume via `--hf-repo`.

### 4C. Modal (server-side, no local daemon — the reliable drip)
`modal run --detach app.py::run_midtrain3` (spawns `midtrain3()` which calls
`MidtrainExtend3Config`). Each workspace = ~$30 ≈ ~1000 steps. Between
workspaces: `modal volume get`/`put` the latest `osrt_v5_midtrain3_step_*.pt`
(and the base) across `osrt-checkpoints`, then relaunch — the resume-scan
continues. `--detach` is mandatory (a non-detached spawn dies with the client).

---

## 5. Gotchas / landmines (each cost real debugging — DO NOT rediscover)

1. **Native HRA — never `inject_hra` on v6.** `hra_native=True` builds adapters
   from config; the base already carries `adapters_a/b`. Injecting corrupts the
   key layout. `run_pretrain_extend` handles this via the `hra_native` flag.
2. **DataLoader worker crash: `Fatal Python error: PyGILState_Release`.** Spawned
   streaming workers (tokenizers/pyarrow C-ext + torch 2.11) hit a fatal
   teardown race, killed runs mid-stream. Fix: `--num-workers 0` (in-process).
   Downside is throughput on some hosts; if a host is stable with workers, use
   them. The eval loader was separately fixed to `num_workers=0` + non-fatal
   try/except.
3. **A100-40GB cannot fit batch-6** (~58GB needed) — OOMs on backward even with
   `expandable_segments`. Use `--micro-batch 2 --grad-accum 33`. RTX PRO 6000
   (96GB) / H100 (80GB) fit batch-6 as-is.
4. **The in-loop held-out eval is toxic on metered/reclaim GPUs** — its
   `skip=100M` fineweb build stalls the GPU 20–30 min (single-threaded), and
   Colab reclaims idle-GPU VMs → killed a run right before its checkpoint.
   DISABLED in midtrain3 (`eval_interval` huge). **Eval checkpoints OFFLINE**
   instead (§7).
5. **Colab reclaims VMs when the local keep-alive daemon dies** (Mac sleep/off,
   oauth2 scope drop, capacity). This is why an always-on machine or Modal is
   better. `ckpt_interval=100` bounds the loss to ≤100 steps.
6. **`colab exec` websocket is unreliable** — times out, returns empty/partial,
   shows stale steps. Trust **W&B + the HF repo**, not exec reads. `colab status`
   (REST) reliably says if the session is alive.
7. **EOS was missing from SFT rollout targets** (`data.py` `_build_sequence`) →
   the SFT model never learned to stop (ran to the length cap). FIXED (appends
   `eos_token_id` with a real label). Matters for the *next* SFT v2 build.
8. **Local MPS/CPU inference froze the Mac** (601M model, 4.9GB checkpoint in
   unified memory). NEVER eval/generate locally. Use the GPU-side Modal
   entrypoints (§7).
9. **`gh auth switch --user CodeHalwell`** before every `git push`.

---

## 6. Monitoring (reliable signals only)

- **W&B** (`codhe-synextra/osrt`, run `osrt-v6-midtrain3`): authoritative for
  step/loss/tok_per_sec/state. Query via `wandb.Api()` with `WANDB_API_KEY`.
  Watch: `_step` advancing, `extend/task_loss` (noisy ~1.3, roughly flat is
  EXPECTED for gentle continued-pretrain — capability shows as offline eval ppl,
  not per-step loss), `moe/drop_rate_mean` should stay ~0.
- **HF repo** `HallD/osrt-v6-ckpt`: `HfApi().list_repo_files(...)` — new
  `midtrain3_step_*.pt` every 100 steps confirms the run is banking + resumable.
- **Success signal:** step advancing + checkpoints appearing + GPU ~100% util.
  Healthy tok/s: ~7k on RTX PRO 6000, ~10k on H100, ~38s/step.

---

## 7. Offline evaluation (GPU-side, never local)

`app.py` has (run via Modal on a workspace with the checkpoint uploaded):
- `modal run app.py::run_sft_eval --step <N|final> --n 100 --max-new-tokens 768`
  — scored GSM8K reasoning-ON vs OFF (`acc_on`, `acc_off`,
  `acc_delta_on_minus_off`, `format_ok`, `resp_len`). ~$0.30.
- `modal run app.py::run_sft_sample --step <N> --n 3` — prints full generations
  (qualitative read).
Note: use `--max-new-tokens ≥768` (long CoT needs room to close into
`<|answer|>`; 400 truncates → false format_ok=0). The eval reads the answer
from inside the `<|answer|>` block, so trailing ramble doesn't corrupt it.

**To eval a midtrain3 checkpoint's raw ppl** (not SFT): easiest is to run a
short probe or a temporary `run_eval` on a checkpoint; the midtrain2 baseline to
beat is **ppl 28.2**. A drop confirms the extra tokens are working.

---

## 8. The roadmap (stages, in order)

1. **midtrain3 (NOW)** — pretrain toward 1× Chinchilla (~12,600 steps / +3.4B
   tok). Chain across sessions via HF. Periodically eval a checkpoint's ppl
   (offline) — expect it to drop below 28.2 and keep falling. When it plateaus
   or budget runs out, take the latest checkpoint as the new base.
2. **Re-run SFT v2** — rebuild `sft_v2.jsonl` (now EOS-fixed), repoint
   `SFTv2Config.pretrained_checkpoint` to the best midtrain3 checkpoint, run
   ~1000 steps (Modal `run_sft_v2` or a lightning entry). Then `run_sft_eval` —
   THIS is where GSM8K should first lift off the ~0.05 floor if the base
   strengthened. If it doesn't, the base needs more midtrain3 tokens (loop 1↔2).
3. **GRPO** — verifiable-reward RL (reuse grpo_v2 infra; `rewards.py`,
   `train_config.GRPOConfig`). Only worthwhile once SFT gives a base with
   ≥~15–20% GSM8K (GRPO amplifies correct rollouts; at 5% the signal is too
   thin). Mind `feedback_grpo_kl_coeff_caution` (don't bump kl_coeff on
   extensions).
4. **Polish** — checkpoint-soup the best 2–3, full lm-eval (`run app.py::evaluate`,
   v6 lane already fixed), merge/export.

---

## 9. Key files

- `src/osrt/train_config.py` — ALL configs (`MidtrainExtend3Config`,
  `SFTv2Config`, `GRPOConfig`, …). Change hyperparams here.
- `src/osrt/train.py` — `run_pretrain_extend` (the midtrain/SFT loop; resume
  scan ~line 1560; LR schedule `_set_param_group_lrs`).
- `scripts/lightning_midtrain3.py` — off-Modal entry (Colab/own-GPU). Flags:
  `--ckpt-dir --hf-repo --num-workers --micro-batch --grad-accum --peak-lr
  --total-steps --ckpt-interval --sanity`.
- `scripts/hf_ckpt_sync.py` — HF pull-latest-on-start / push-on-save.
- `scripts/build_sft_v2_data.py` — verified SFT corpus builder.
- `app.py` — Modal entrypoints (`run_midtrain3`, `run_sft_v2`, `run_sft_eval`,
  `run_sft_sample`, `evaluate`, …).
- `docs/colab_midtrain3.md` — Colab CLI runbook.
- Memory (`~/.claude/.../memory/`): `project_v6_posttrain_roadmap.md`,
  `colab_midtrain3_recipe.md`, `feedback_*`.

---

## 10. First actions for the resuming agent

1. `git pull`; `gh auth switch --user CodeHalwell`; `uv run pytest` (expect
   196 pass) to confirm a clean tree.
2. Confirm the base is reachable: `HallD/osrt-v6-ckpt` has
   `osrt_v5_midtrain2_step_1750.pt` (and any `midtrain3_step_*` if a prior
   session banked one).
3. Pick the compute path (§4A own-GPU is best on an always-on box).
4. **Run the 30-step `--sanity` gate first**, verify it clears step 30 (no
   PyGILState/OOM), THEN launch the full run in tmux/detached.
5. Monitor via W&B + HF (§6). First checkpoint at step 100 (~1hr) confirms
   banking works. Report ppl vs 28.2 periodically.
6. When midtrain3 has added meaningful tokens, proceed to §8 stage 2.

Good luck. The hard part (diagnosis, infra, all 9 landmines) is done — this is
now a matter of feeding it tokens and chaining checkpoints.
