# Checkpoint Sync & SFT-v2 Data Builder — Verified Findings

**Date:** 2026-07-26
**Status:** Findings record — **nothing implemented, no code changed.** Three
findings surfaced in review on 2026-07-26; each was verified against the tree
at commit `7ebc2e7` by code-reading. Fix sketches below are proposals, not
implementations.
**Companions:** `docs/specs/2026-07-26-precision-and-sft-objective.md`,
`scripts/hf_ckpt_sync.py`, `scripts/build_sft_v2_data.py`,
`docs/colab_midtrain3.md`.

---

## 0. TL;DR

| finding | severity | verdict |
|---|---|---|
| Checkpoints can be uploaded while still being written | **High** | **Confirmed.** `torch.save` writes directly to the glob-matched name; the sync daemon permanently marks a truncated upload as done. §1 |
| The completed run's final checkpoint is never synchronized | **Medium-High** | **Confirmed, and broader than reported:** the 23h *rescue* checkpoint is also structurally excluded, the daemon thread dies at process exit with no flush, **no `total_steps` value is exempt** (save-before-increment), and **widening the sync globs alone does not fix it** — the local resume scan also ignores `_final.pt`. §2 |
| The chat slice bypasses the advertised decontamination + dedup | **Medium** | **Confirmed.** Slices 1–3 gate through `admit()`; the ~8.7k-row chat slice never calls it. §3 |

Scope note: findings 1–2 bite only the Colab/Lightning path —
`lightning_midtrain3.py:154-159` is the sole consumer of the daemon; Modal runs
persist via `vol.commit()` instead. That path is the workflow in active use.

---

## 1. High — mid-write checkpoint uploads

### 1.1 The race

- `save_checkpoint` (`train.py:130-147`) calls `torch.save(state, path)`
  **directly to the final filename** — no write-to-temp + `os.replace`. The
  file exists at its glob-matched name from byte 0 and grows for the several
  seconds a ~4.9 GB checkpoint takes to serialize (size per
  `hf_ckpt_sync.py:58`'s own docstring).
- The push daemon polls every 60 s (`hf_ckpt_sync.py:66-79`): any
  `{prefix}_step_*.pt` not in `pushed` is uploaded immediately, then
  `pushed.add(name)` — **permanently**. A poll landing inside the write window
  uploads a truncated file and never re-uploads the completed one.
- The asymmetry that makes it permanent: a *failed* upload is retried (the
  broad `except` at `hf_ckpt_sync.py:86-88` skips the `pushed.add`), but a
  *successful* upload of a partial file is recorded as done. Success of the
  transfer is treated as success of the artifact.

### 1.2 Blast radius

`pull_latest` (`hf_ckpt_sync.py:45-49`) resumes the next session from the
**highest-step** remote file. If that is the truncated upload, `torch.load`
crashes at resume and the cross-session chain — the script's entire purpose —
is broken until the bad remote file is deleted by hand. Pruning keeps the
newest 3 (`hf_ckpt_sync.py:83-85`), so the corrupt newest is retained.

### 1.3 Fix sketch (not implemented)

1. **Atomic save at the source:** in `save_checkpoint`, `torch.save` to
   `path + ".tmp"` then `os.replace(tmp, path)`. One function, fixes every
   caller, and also protects the *local* resume-scan against a crash mid-save.
2. Optional daemon-side guard: upload only files whose size/mtime is stable
   across two consecutive polls, and add to `pushed` only after a
   verified-stable upload.
3. One-time audit: past midtrain3 sessions may already have fired this race —
   check the remote repo's `*_step_*.pt` sizes against local expectations (or
   try loading headers) before trusting a resume.

---

## 2. Medium-High — final AND rescue checkpoints are never synchronized

### 2.1 Three stacked gaps

1. **Final:** the end-of-run save is `osrt_v5_{prefix}_final.pt`
   (`train.py:2022-2023`), which can never match the daemon's
   `{prefix}_step_*.pt` glob (`hf_ckpt_sync.py:70`). The tail of training
   exists only in `_final.pt` on the ephemeral VM disk — lost when the VM is
   reclaimed.

   > **Correction (PR #5 review, verified).** An earlier version of this
   > paragraph read "unless `total_steps` lands exactly on `ckpt_interval`".
   > **That exception does not exist.** The loop is `while step < total_steps`
   > (`train.py:1748`) and saves *before* incrementing
   > (`:1998` check, `:2019` increment), so the periodic block never executes at
   > `step == total_steps`. The last iteration runs at `step = total_steps - 1`,
   > so a numbered save equivalent to the final model requires
   > **`(total_steps - 1) % ckpt_interval == 0`**, not `total_steps %
   > ckpt_interval == 0`.
   >
   > This matters for the configuration actually in use: midtrain3 at
   > **12,600 steps / interval 100** looks safe under the old rule but is not —
   > `12599 % 100 = 99`, so the newest numbered artifact is **step 12,500** and
   > the last 100 steps live only in `_final.pt`. **No current configuration is
   > exempt.**
2. **Rescue (not in the original finding):** the 23h-cap save is
   `osrt_v5_{prefix}_rescue_step_{step}.pt` (`train.py:2004-2008`).
   `_rescue_step_` does not match `_step_` immediately after the prefix, so
   the daemon never uploads it — and `pull_latest`'s regex
   (`hf_ckpt_sync.py:45`) never fetches it — even though the **local**
   resume-scan (`train.py:1636-1637`) explicitly includes rescue files. This
   file is written at the session time cap, i.e. moments before the disk
   becomes unreachable: in the Colab workflow it is the single most important
   file to persist, and it is structurally unsyncable. Each capped session
   silently loses up to `ckpt_interval` steps of progress.
3. **No flush at exit:** the sync thread is `daemon=True`
   (`hf_ckpt_sync.py:91`), and both the rescue path and the final path
   return/exit right after saving — so even a glob-*matching* file written
   less than one poll interval before exit can miss its last upload window.

### 2.2 Fix sketch (not implemented)

- Widen the daemon glob and `pull_latest` patterns to cover
  `_rescue_step_*.pt` and `_final.pt` (resume preference order must mirror
  the local scan's rescue-on-ties logic).
- ⚠️ **Widening the sync patterns is necessary but not sufficient** (PR #5
  review, verified). `run_pretrain_extend`'s local resume scan iterates only
  `osrt_v5_{prefix}_step_*.pt` and `osrt_v5_{prefix}_rescue_step_*.pt`
  (`train.py:1635-1638`) and derives its ordering key from the trailing step
  number. A downloaded `_final.pt` therefore has **no step to sort on and is
  silently skipped** — a restarted or extended run would resume from an older
  numbered checkpoint while the final weights sit unused on disk. The fix must
  also either (a) teach the local scan to rank `_final.pt` above every numbered
  file, or (b) persist the end-of-run save under a step-numbered name
  (e.g. `_step_{total_steps}.pt`) so both the scan and the daemon glob match it
  for free. **(b) is the smaller change and removes the special case entirely.**
- Export a synchronous `flush()` from `hf_ckpt_sync` and call it from the
  trainer (or `lightning_midtrain3.py`) after the rescue/final save, before
  exit. The background thread alone can never make the last write durable.

---

## 3. Medium — chat slice bypasses decontamination and dedup

### 3.1 The gap

`build_sft_v2_data.py:19-20` advertises:

> - GSM8K TEST decontamination (normalized-prefix match) **on every problem.**
> - Within-corpus dedup by problem hash.

The gate is `admit()` (`build_sft_v2_data.py:120-129`: contamination check +
`_phash` dedup). Slices 1–3 all call it — mopd (`:172`), OpenR1 (`:192`),
Bespoke-Stratos (`:232`). The chat slice (`:255-279`, `CHAT_TARGET = 8_700`)
applies only the length filter and **never calls `admit`**. Chat rows
therefore get neither the GSM8K-test prefix check nor the problem-hash dedup:
duplicates inside `system_prompt_sft.jsonl` pass through, as does any chat row
duplicating a math-slice problem.

### 3.2 Materiality

The chat source is UltraChat + OpenHermes-2.5 rows with system prompts
(`collect_system_rollouts.py:1-3`). OpenHermes-2.5 aggregates instruction
sets that include GSM8K-style math, so test contamination through this path is
plausible, not merely theoretical; the dedup gap is near-certain at 8.7k
sampled rows. Any GSM8K eval claim made for models trained on this corpus
inherits the caveat.

### 3.3 Fix sketch (not implemented)

Gate the chat emit loop with `admit(q)` — one line. The existing
`contaminated`/`dup` counters (`:294-295`) will then report how much was
actually slipping through. Corpora already built by earlier runs of the
script inherit the gap: regenerate, or at minimum run the decon/dedup check
offline over the existing JSONL before citing decontamination in any eval
claim.

---

## 4. Open items

- [ ] **Atomic `save_checkpoint`** (tmp + `os.replace`) — the one-place fix
      for §1; protects local resume too. (§1.3)
- [ ] **Sync coverage for `_rescue_step_*.pt` and `_final.pt`** in both the
      daemon glob and `pull_latest`, with rescue-aware resume ordering. (§2.2)
- [ ] **Synchronous `flush()` at end-of-run** — called after rescue/final
      saves, before process exit. (§2.2)
- [ ] **One-time audit of already-uploaded HF checkpoints** for truncation —
      the §1 race may have already fired in past sessions. (§1.3)
- [ ] **`admit(q)` on the chat slice**, then regenerate or offline-audit the
      existing sft_v2 corpus before relying on the decontamination claim. (§3.3)
