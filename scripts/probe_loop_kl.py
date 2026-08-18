"""Variable-loop output-perturbation probe — how much does dropping recursive
loops change the model's predictions?

Companion to scripts/probe_cross_loop_kv.py. The KV probe showed the recursion
is a contracting iteration that has largely converged by loops 4-5 — which
*motivates* the variable-loop inference knob (run K<L loops to save decode
compute). This probe measures the knob's cost directly, forward-only, on the
base model (no reasoning/SFT model required):

  for K in {3,4,5}: KL( P(full L=6 loops) || P(K loops) ) and top-1 agreement,
  plus the held-out next-token CE/ppl at each K.

Small KL + high top-1 agreement at K=4/5 => dropping loops barely moves the
output distribution => the compute saving is nearly free distributionally. The
ACCURACY validation (does on>off survive fewer loops?) waits for a checkpoint
with a measurable reasoning delta — this is the distributional half, bankable now.

Run:
  HF_TOKEN=... PYTHONPATH=src python scripts/probe_loop_kl.py \
      --ckpt checkpoints/v5/osrt_v5_midtrain_final.pt --texts general
      --out /tmp/loop_kl.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, _, v = _l.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Reuse the exact probe batches from the KV probe (scripts/ is on sys.path[0]
# when run as a script). DEFAULT_PROBE_TEXTS is the math-heavy set.
from probe_cross_loop_kv import (  # noqa: E402
    DEFAULT_PROBE_TEXTS as MATH_PROBE_TEXTS,
)
from probe_cross_loop_kv import (  # noqa: E402
    GENERAL_PROBE_TEXTS,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v5/osrt_v5_midtrain_final.pt")
    ap.add_argument("--tokenizer", default="v6_tokenizer_export")
    ap.add_argument("--texts", choices=["math", "general", "mixed"], default="general")
    ap.add_argument("--text-file", default=None,
                    help="One example per line; overrides --texts. Use a "
                         "reasoning-token-dense corpus (e.g. GSM8K) so the "
                         "math_op/connective flip-type categories have real N.")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config
    from osrt.train import load_model_state_or_raise

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"device={device}", flush=True)

    if args.text_file:
        texts = [ln for ln in Path(args.text_file).read_text().splitlines()
                 if ln.strip()]
        print(f"loaded {len(texts)} examples from {args.text_file}", flush=True)
    else:
        texts = {"math": MATH_PROBE_TEXTS, "general": GENERAL_PROBE_TEXTS,
                 "mixed": MATH_PROBE_TEXTS + GENERAL_PROBE_TEXTS}[args.texts]
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = build_config(vocab_size=len(tok), real_vocab_size=len(tok),
                       bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                       pad_token_id=tok.pad_token_id)
    batch = (texts * ((args.batch // len(texts)) + 1))[: args.batch]
    enc = tok(batch, return_tensors="pt", padding="max_length",
              truncation=True, max_length=args.seq_len)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    model = OSRTForCausalLM(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    load_model_state_or_raise(model, ck.get("model_state_dict", ck),
                              context=f"loop-kl {args.ckpt}")
    model = model.to(device).eval()

    full_loops = cfg.recursive_loops
    rv = cfg.real_vocab_size

    @torch.no_grad()
    def logits_at(k):
        out = model(input_ids=input_ids, attention_mask=attn, num_loops=k)
        return out.logits[..., :rv].float()

    # Valid NEXT-token positions: predict token t+1 from position t, both real.
    valid = (attn[:, 1:].bool()).reshape(-1)            # (B*(S-1),)
    gold = input_ids[:, 1:].reshape(-1)[valid]

    def ce_ppl(lg):
        lp = F.log_softmax(lg[:, :-1, :], dim=-1).reshape(-1, rv)[valid]
        ce = F.nll_loss(lp, gold)
        return float(ce), float(torch.exp(ce))

    full = logits_at(full_loops)
    full_lp = F.log_softmax(full[:, :-1, :], dim=-1).reshape(-1, rv)[valid]
    full_p = full_lp.exp()
    full_logits_valid = full[:, :-1, :].reshape(-1, rv)[valid]
    full_argmax = full_logits_valid.argmax(-1)
    ce_full, ppl_full = ce_ppl(full)

    rows = []
    argmax_by_k = {full_loops: full_argmax}   # per-loop-count argmax (valid pos)
    for k in range(2, full_loops):
        lg = logits_at(k)
        lp = F.log_softmax(lg[:, :-1, :], dim=-1).reshape(-1, rv)[valid]
        kl = float((full_p * (full_lp - lp)).sum(-1).mean())   # KL(full || k)
        k_argmax = lg[:, :-1, :].reshape(-1, rv)[valid].argmax(-1)
        top1 = float((k_argmax == full_argmax).float().mean())
        ce_k, ppl_k = ce_ppl(lg)
        rows.append({"loops": k, "kl_full_given_k": round(kl, 4),
                     "top1_agree": round(top1, 4),
                     "ce": round(ce_k, 4), "ppl": round(ppl_k, 3)})
        argmax_by_k[k] = k_argmax
    drop1_argmax = argmax_by_k.get(full_loops - 1)  # 6->5 headline boundary

    # ---- 6->5 flip-type analysis: what KIND of token does the last loop
    # change its mind about, and are those flips low-confidence near-ties?

    def categorize(tid: int) -> str:
        s = tok.decode([int(tid)]).strip().lower()
        if not s:
            return "punct/space"
        if any(c.isdigit() for c in s):
            return "digit"
        if all(not c.isalnum() for c in s):
            return "punct/space"
        if s in {"therefore", "thus", "hence", "because", "since", "so",
                 "if", "then", "implies", "equals"}:
            return "reasoning_connective"
        return "word/other"

    flip_report = None
    if drop1_argmax is not None:
        flip = (drop1_argmax != full_argmax)                  # (N_valid,) bool
        conf = full_p.max(-1).values                          # full-model top prob
        cats = ["digit", "math_op", "reasoning_connective", "punct/space",
                "word/other"]
        # math operators live in the gold token string; fold into categorize via
        # a quick override set.
        op_chars = set("+-*/=<>%^")

        def cat2(tid: int) -> str:
            s = tok.decode([int(tid)]).strip()
            if s and all(c in op_chars for c in s):
                return "math_op"
            return categorize(tid)

        gold_cpu = gold.tolist()
        flip_cpu = flip.tolist()
        all_counts = {c: 0 for c in cats}
        flip_counts = {c: 0 for c in cats}
        for g, f in zip(gold_cpu, flip_cpu, strict=True):
            c = cat2(g)
            all_counts[c] += 1
            if f:
                flip_counts[c] += 1
        n_all = max(sum(all_counts.values()), 1)
        n_flip = max(sum(flip_counts.values()), 1)
        # over-representation = (flip share of category) / (overall share)
        enrich = {c: round((flip_counts[c] / n_flip) /
                           max(all_counts[c] / n_all, 1e-9), 2) for c in cats}
        # P(flip | category) — the legible metric ("digits flip 22% vs words 9%").
        flip_rate_by_cat = {c: round(flip_counts[c] / max(all_counts[c], 1), 3)
                            for c in cats}
        flip_report = {
            "boundary": f"{full_loops}->{full_loops - 1}",
            "flip_rate": round(float(flip.float().mean()), 4),
            "mean_conf_flipped": round(float(conf[flip].mean()), 4)
            if flip.any() else None,
            "mean_conf_unflipped": round(float(conf[~flip].mean()), 4)
            if (~flip).any() else None,
            "gold_category_counts_all": all_counts,
            "gold_category_counts_flipped": flip_counts,
            "enrichment_flip_vs_all": enrich,
            "flip_rate_by_category": flip_rate_by_cat,
        }
        print("\n6->5 FLIP-TYPE ANALYSIS (gold-token category at flipped positions)")
        print(f"  flip rate {flip_report['flip_rate']:.3f} | mean full-model "
              f"confidence: flipped {flip_report['mean_conf_flipped']} vs "
              f"unflipped {flip_report['mean_conf_unflipped']}")
        print(f"  {'category':>22}  {'P(flip|cat)':>11}  {'enrich':>6}  "
              f"{'flip/all':>10}")
        for c in cats:
            flag = "  <LOW-N" if all_counts[c] < 20 else ""
            print(f"    {c:>20}  {flip_rate_by_cat[c]:>11.3f}  {enrich[c]:>6}  "
                  f"{flip_counts[c]:>4}/{all_counts[c]:<5}{flag}")

        # All-boundary sweep: per-category P(flip) at each adjacent k->k+1, to
        # test whether reasoning-token specialization lives in the early/mid
        # loops (where the representational work happens) rather than the
        # converged tail. gold_cat computed once and reused across boundaries.
        gold_cat = [cat2(g) for g in gold_cpu]
        boundary_rates = {}
        for kk in range(2, full_loops):
            fl = (argmax_by_k[kk] != argmax_by_k[kk + 1]).tolist()
            allc = {c: 0 for c in cats}
            flc = {c: 0 for c in cats}
            for gc, f in zip(gold_cat, fl, strict=True):
                allc[gc] += 1
                if f:
                    flc[gc] += 1
            boundary_rates[f"{kk + 1}->{kk}"] = {
                c: round(flc[c] / max(allc[c], 1), 3) for c in cats}
        flip_report["per_boundary_flip_rate_by_category"] = boundary_rates
        bnames = list(boundary_rates)
        print("\n  ALL-BOUNDARY P(flip|cat) — is specialization in the early loops?")
        print(f"    {'category':>20}  " + "  ".join(f"{b:>8}" for b in bnames))
        for c in cats:
            print(f"    {c:>20}  "
                  + "  ".join(f"{boundary_rates[b][c]:>8.3f}" for b in bnames))

    report = {"ckpt": args.ckpt, "texts": args.texts, "full_loops": full_loops,
              "full_ce": round(ce_full, 4), "full_ppl": round(ppl_full, 3),
              "rows": rows, "flip_types": flip_report}
    print("\n" + "=" * 60)
    print(f"VARIABLE-LOOP OUTPUT PERTURBATION (full L={full_loops}, "
          f"ppl={ppl_full:.2f})")
    print("=" * 60)
    print(f"{'loops':>5} {'KL(full||k)':>12} {'top1-agree':>11} {'ppl':>8}")
    for r in rows:
        print(f"{r['loops']:>5} {r['kl_full_given_k']:>12.4f} "
              f"{r['top1_agree']:>11.3f} {r['ppl']:>8.2f}")
    print(f"{full_loops:>5} {0.0:>12.4f} {1.0:>11.3f} {ppl_full:>8.2f}  (full)")
    print("=" * 60)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
