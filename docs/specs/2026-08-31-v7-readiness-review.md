# v7 readiness review — 2026-08-31

**Question:** is the repo ready to start v7 work tomorrow (2026-09-01)?

**Verdict: yes, with one thing to verify first (the v6 checkpoint repo) and
one ambiguity to settle (§14.1's `[graduated]` items are not in the code
defaults).** Everything the roadmap calls ready was re-verified green today
from a fresh clone. The day-one work is *building the gate harnesses* (G3a
ladder, G8 drafter), not launching the trunk run — the trunk is explicitly
blocked on G3a and correctly so.

All checks below were run on `main` @ `2fe8217` (branch and main identical).

---

## 1. Verified green today

| check | result |
|---|---|
| `uv sync --frozen` | clean (lockfile consistent, py3.11) |
| `uv run ruff check .` | all checks passed |
| `uv run pytest -q` | **213 passed, 1 skipped** — matches the last two commit messages |
| `compute_budget.py` | **968,468,355 physical / 263,035,779 active** — byte-identical to §16.5 |
| `cfg:` echo line | `h_routed=2112` — survives the model.py ×64 round-up as §14.1 predicted |
| GitHub CI | green on `main`, all 7 runs successful |

Structural checks:

- **`OSRT_V7` preset matches the §14 committed shape**: dim 1536, 3×6,
  28×h2112 top-4, shared h2816, GQA 24/8, HRA rank 256, MTP 2 heads,
  `router_balance_mode="quantile"` (§14.6 required — implemented and tested,
  `test_quantile_balancing.py`), mHC absent, vocab 49,280/49,184. No
  parameter count appears in any name.
- **G2 tokenizer is real, not just decided**: `tokenizer/` contains SmolLM2's
  49,152 base + exactly 32 OSRT specials at ids 49,152–49,183 (= 49,184 real),
  rebuildable via `build_tokenizer_v7.py`; `tokenizer_v6/` retained for G8.
- **The training path is wired for the card and the funding model**:
  `train_main.py` defaults to `./tokenizer` and warns on vocab drift against
  the preset; WSD is the default schedule; `--hf-repo` pull/push sync with a
  `finally` flush covers Colab pre-emption; the sync `create_repo(...,
  exist_ok=True)` means the v7 checkpoint repo need not pre-exist.
- **The Colab notebook is probe-first and current**: cell 1 gates on sm_120,
  `probe_gpu.py` is G7 part 1, the budget cell asserts the committed
  968,468,355 / 263,035,779, and the launch cell is a deliberately short run.
- **Hygiene holds**: weights/data/jsonl/PDFs gitignored; `docs/v6/` and
  `ARCHITECTURE.md` carry superseded banners; `train_config.py` honestly
  declares its values inherited-from-v6 and gated on G3a.

Gate board as actually verified: **G1 closed** (mHC off, §12.3 — but see
finding 3), **G2 closed** (shipped, not just decided), **G7 part 1 ready to
run** (needs the GPU box), **G3a / G3 / G4 / G8 open with no harness yet**
(finding 4).

---

## 2. Findings, in priority order

### 2.1 `HallD/osrt-v6-ckpt` did not resolve — verify before relying on G8

An authenticated HF API call (account `HallD`) found neither
`HallD/osrt-v6-ckpt` nor `HallD/osrt-v7-ckpt` as model repos. The v7 repo is
harmless — `hf_ckpt_sync.py` auto-creates it on first push. But CLAUDE.md
and the roadmap treat the frozen v6 checkpoint as a banked asset, and G8
("the cheapest real experiment currently available", §15.7) plus any
v6-vs-v7 matched-token control **depend on it existing**. If it lives under
a different name, type, or account, update CLAUDE.md; if it is gone, that is
a material loss to record before plans are built on it. Five-minute check,
do it first.

### 2.2 §14.1's `[graduated]` line is not in the code defaults — settle the reading

The committed-shape block says
`SiTU-GLU experts; per-head Muon; V4 Muon recipe  [graduated]` and its
router line includes `seq-balance 1e-4`. In code today:

| §14.1 says | code ships |
|---|---|
| seq-balance 1e-4 | `router_seq_balance_loss_coeff` defaults **0.0**; `OSRT_V7` does not set it (§7.2 item 1.5: "field exists, ships 0" — still true) |
| SiTU-GLU experts | `situ_glu` defaults **False**; `OSRT_V7` does not set it — experts run the hard clamp |
| per-head Muon | `per_head_muon: bool = **False**` in `PretrainConfig` |
| V4 Muon recipe (NS 8+2, update-RMS 0.18, WD∝LR²) | not implemented; `muon.py` is NS-5 + Nesterov |

This is defensible — G3's ladder explicitly A/Bs "SiTU vs clamp, per-head
Muon on/off" — but then §14.1 should say *ladder candidates*, not sit inside
a committed block tagged `[graduated]`. As written, a trunk run launched from
today's preset would not run the shape §14.1 commits to. Either flip the
preset/config defaults, or amend §14.1's wording. One sentence in either
place; decide before the ladder configs are written so they A/B against the
intended baseline.

### 2.3 CLAUDE.md is stale on exactly the things it warns about

- Gate board says **G2 open**; §16 resolved it on 2026-08-18 and the
  tokenizer is shipped. Same for the "Open: tokenizer (G2)" line.
- "mHC: OFF — decided permanently" predates the §12.3 **amendment** (GLM-5.3-
  Flash): the decision stands, but the G3 ladder slot returned as cheap
  insurance — "permanently" now overstates it.
- The `## Environment & commands` section is **empty** (heading with no
  body).

Item 0.1's own words apply: the methodology is the paper trail.

### 2.4 No harness exists yet for G3a, G3/G4, or G8 — this is the day-one build

`ladder` appears only in comments; there is no ~150M proxy preset, no G3a
runner, and no drafter/acceptance harness for G8. That is consistent with
the plan (these gates are the *next work*, and G3a "does not require the
full G3 ladder — run it first", §14.7) — but it means tomorrow starts with
harness-building on the CPU/always-on side plus `probe_gpu.py` (G7) on the
first GPU session, not with training the trunk.

### 2.5 Trivia

- `presets.py:72` — comment truncated mid-sentence: "NOTE: §14.6 makes
  Quantile Balancing REQUIRED for".
- `sanity_overfit.py` docstring still says "lean-v6" / "OSRT-605M
  architecture" and its proxy is 8-expert top-2; fine functionally, stale
  wording, and worth a QB-mode proxy variant once the ladder configs exist.
- §16.6's "240 passed" describes the pre-cut suite; 213 is correct since the
  v6-stack removal. No action, just so nobody chases a phantom regression.
- §12.6's `paper.tex` defect is not in this repo (file lives in the v6
  archive) — still open *there* if the paper is ever shown.

---

## 3. Suggested day-one order (unchanged from the roadmap, made concrete)

1. Verify where the frozen v6 checkpoint actually lives (2.1).
2. Settle the §14.1-vs-defaults reading (2.2); unstale CLAUDE.md (2.3).
3. First GPU session: `probe_gpu.py` — settles which card Colab serves and
   G7 part 1; fold in the E=28/h2112 grouped-GEMM timing (§14.6 item 2).
4. Build the G3a arm: a ~150M-active ladder preset + config, three runs at
   fixed active / varying total (§14.7). This blocks the trunk; nothing else
   does.
5. In parallel on CPU: the G8 drafter harness against frozen v6 (§15.7) —
   cheapest real experiment, blocks nothing.
6. Short smoke via the notebook's 200-step run before any real spend, per
   the standing rule.
