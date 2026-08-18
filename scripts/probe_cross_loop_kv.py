"""P0 probe — does the recursive stack put DISTINCT information in per-loop K/V?

The decision gate for cross-loop KV reuse (see
docs/specs/2026-06-16-cross-loop-kv-reuse.md).
Cross-loop reuse is a potential ~6x decode-bandwidth win (stacks with KDV's 2x),
but it is only cheap if the cached latent `c_kv` is similar across the 6 loops of
a physical block. This script measures exactly that — forward-only, no training,
no Modal — so the answer is known before any GPU dollars are spent.

How: the model already returns the per-(loop, block) cached latents. A
`use_cache=True` forward yields `past_key_values` as a flat list of
`num_blocks * recursive_loops` latents (each (B, S, kv_dim)), indexed
`idx = loop * num_blocks + block_idx` (model.py:1660, :2087). We regroup them by
physical block and, per block, measure how much loop k's latent resembles the
latent that a reuse scheme would force it to share.

Metrics per physical block (padding tokens masked out):
  - linear CKA across the loops (rotation/scale-invariant representation match),
    incl. the adjacent-loop contraction series 1-CKA(k,k+1) — a monotonically
    shrinking move size is the fixed-point/DEQ-like contraction signature.
  - mean per-token cosine(loop_k, loop_0)
  - INJECTED relative-L2 error for two concrete schemes:
      * first-loop share  (cache loop 0 only; loops 1..L-1 reuse it)  -> ~6x
      * two-group split   (cache loops 0 and L/2; reuse within halves) -> ~3x
    This is literally the perturbation the scheme imposes, so it is the most
    decision-relevant number.

Triangulation (all from the SAME forward, so the signals are comparable):
  - residual update |dx|/|x| per (block, loop)  [last_loop_update_norm]
  - per-loop routing entropy (marginal + per-token)  [block.moe telemetry]
  Three independent signals showing the same front-loaded contraction is a far
  stronger non-degeneracy claim than KV-CKA alone.

Robustness: --texts {math,general,mixed} swaps the built-in batch so the
cross-loop structure can be confirmed off the math-heavy default.

Interpretation hint (heuristic, not a hard rule):
  CKA(loop_k, loop_0) high (>~0.9) AND injected first-share error low (<~0.10)
      => first-loop share (6x) looks nearly free; try it.
  Mid                                  => grouped g=3 (3x) is the safe middle.
  CKA low / error high                 => loops carry distinct KV (the recursion
                                          IS refining) => prefer low-rank delta,
                                          do NOT full-share.

Run (real checkpoint, on the box that has it):
  HF_TOKEN=... PYTHONPATH=src python scripts/probe_cross_loop_kv.py \
      --ckpt checkpoints/v5/osrt_v5_midtrain_final.pt --seq-len 512 --batch 8

Smoke test (no checkpoint needed, proves the plumbing):
  PYTHONPATH=src python scripts/probe_cross_loop_kv.py --random-init --tiny
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Load .env (HF_TOKEN etc.) the same way the other local scripts do.
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, _, v = _l.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# A small, varied default probe batch (held-out-ish; math-heavy to match the
# mid-trained domain). Override with --text-file (one example per line).
DEFAULT_PROBE_TEXTS = [
    "The derivative of x^3 + 2x with respect to x is 3x^2 + 2.",
    "A train travels 60 miles in 1.5 hours. Its average speed is 40 miles per hour.",
    "To solve 2x + 5 = 17, subtract 5 from both sides to get 2x = 12, so x = 6.",
    "The capital of France is Paris, a city on the river Seine.",
    "In Python, a list comprehension builds a list from an iterable in one line.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "The sum of the first n positive integers is n(n+1)/2.",
    "If a fair coin is flipped three times, the probability of three heads is 1/8.",
    "Newton's second law states that force equals mass times acceleration.",
    "The integral of 1/x dx is the natural logarithm of the absolute value of x.",
    "Binary search runs in O(log n) time on a sorted array.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "A prime greater than 2 is always odd, since even numbers divide by 2.",
    "The Pythagorean theorem relates a right triangle: a^2 + b^2 = c^2.",
    "Gradient descent steps parameters toward the steepest decrease of the loss.",
    "Twelve divided by four equals three, and three times four equals twelve.",
]

# A non-math, general-domain batch for the robustness re-run (--texts general):
# confirms the cross-loop structure is not an artifact of the math-heavy default.
GENERAL_PROBE_TEXTS = [
    "The Roman Empire reached its greatest territorial extent under Trajan.",
    "She opened the window and listened to the rain falling on the quiet street.",
    "Coffee is one of the most widely traded commodities in the world.",
    "The novel follows a young cartographer who maps a city that keeps changing.",
    "Most migratory birds navigate using a combination of the sun and magnetic fields.",
    "He had never seen the ocean before, and the size of it stopped him cold.",
    "The committee postponed the vote until the following Tuesday afternoon.",
    "Old libraries smell of paper, dust, and the slow patience of decades.",
    "The recipe calls for letting the dough rest overnight in the refrigerator.",
    "After the storm, the villagers spent the week repairing their fishing boats.",
    "Jazz emerged in New Orleans before spreading north along the Mississippi.",
    "The interview lasted an hour, but the real conversation happened afterward.",
    "A good map tells you not only where you are but where you might go.",
    "The garden was overgrown, yet somehow more beautiful for the neglect.",
    "Negotiations broke down late in the evening over a single disputed clause.",
    "They walked home in comfortable silence, the city humming around them.",
]


def _masked_rows(latent, mask):
    """(B, S, D) latent + (B, S) bool mask -> (N_valid, D) of non-pad rows."""

    D = latent.shape[-1]
    flat = latent.reshape(-1, D)
    m = mask.reshape(-1).bool()
    return flat[m].float()


def linear_cka(X, Y) -> float:
    """Linear CKA between two (N, D) matrices. 1.0 = identical up to rotation/scale."""

    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (X.t() @ Y).pow(2).sum()
    xx = (X.t() @ X).pow(2).sum().sqrt()
    yy = (Y.t() @ Y).pow(2).sum().sqrt()
    denom = (xx * yy).clamp_min(1e-12)
    return float((xy / denom).item())


def mean_cosine(X, Y) -> float:
    """Mean per-row cosine similarity between two aligned (N, D) matrices."""
    return float(torch.nn.functional.cosine_similarity(X, Y, dim=-1).mean().item())


def rel_l2(target, approx) -> float:
    """Mean per-row ||target - approx|| / ||target||: the error a reuse injects."""

    num = (target - approx).norm(dim=-1)
    den = target.norm(dim=-1).clamp_min(1e-12)
    return float((num / den).mean().item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints/v5/osrt_v5_midtrain_final.pt")
    ap.add_argument("--tokenizer", default="v6_tokenizer_export")
    ap.add_argument("--text-file", default=None, help="One probe example per line.")
    ap.add_argument("--texts", choices=["math", "general", "mixed"], default="math",
                    help="Built-in probe set (ignored if --text-file is given). "
                         "Use 'general' for the robustness re-run.")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--device", default=None)
    ap.add_argument("--random-init", action="store_true",
                    help="Skip checkpoint load (smoke test only — not science).")
    ap.add_argument("--tiny", action="store_true",
                    help="Use a tiny config for a fast plumbing smoke test.")
    ap.add_argument("--out", default=None, help="Write the full report as JSON here.")
    args = ap.parse_args()

    from osrt.presets import build_config

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"device={device} dtype={args.dtype}", flush=True)

    # ---- tokenizer + input batch ------------------------------------------
    if args.text_file:
        texts = [ln for ln in Path(args.text_file).read_text().splitlines()
                 if ln.strip()]
    elif args.texts == "general":
        texts = GENERAL_PROBE_TEXTS
    elif args.texts == "mixed":
        texts = DEFAULT_PROBE_TEXTS + GENERAL_PROBE_TEXTS
    else:
        texts = DEFAULT_PROBE_TEXTS

    tok = None
    if not args.tiny:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            if tok.pad_token is None:
                # Some tokenizers (LLaMA/GPT-2) ship no pad token; padding
                # ="max_length" would raise. Fall back to eos.
                tok.pad_token = tok.eos_token
        except Exception as e:  # noqa: BLE001
            # Fall back to a tiny random model: the full checkpoint cannot load
            # into a tiny config (shape mismatch), so force random-init too.
            print(f"tokenizer load failed ({e}); falling back to --tiny synthetic "
                  f"ids + random init", flush=True)
            args.tiny = True
            args.random_init = True

    if args.tiny:
        # Synthetic ids — plumbing smoke test, no real tokenizer/checkpoint.
        cfg = build_config(
            dim=128, heads=4, head_dim=32, num_kv_heads=2, vocab_size=512,
            real_vocab_size=512, num_blocks=2, recursive_loops=3,
            num_routed_experts=8, top_k_experts=2, expert_hidden=64,
            shared_expert_hidden=128, max_position_embeddings=args.seq_len,
        )
        g = torch.Generator().manual_seed(0)
        input_ids = torch.randint(0, 512, (args.batch, args.seq_len), generator=g)
        attn = torch.ones_like(input_ids)
    else:
        cfg = build_config(
            vocab_size=len(tok), real_vocab_size=len(tok),
            bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        batch_texts = (texts * ((args.batch // len(texts)) + 1))[: args.batch]
        enc = tok(batch_texts, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=args.seq_len)
        input_ids, attn = enc["input_ids"], enc["attention_mask"]

    from osrt.model import OSRTForCausalLM
    model = OSRTForCausalLM(cfg)

    if not args.random_init:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            print(f"ERROR: checkpoint {ckpt_path} not found. Use --random-init for a "
                  f"smoke test, or point --ckpt at the real file.", flush=True)
            return 2
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = ck.get("model_state_dict", ck)
        # FAIL on any key drift — these metrics drive a KV-reuse decision, so a
        # stale/mismatched checkpoint that leaves params randomly initialised must
        # not masquerade as a real probe. Same contract as the training loader.
        from osrt.train import load_model_state_or_raise
        load_model_state_or_raise(model, sd, context=f"probe load {ckpt_path}")
    else:
        print("random-init: results are a PLUMBING CHECK ONLY, not evidence.",
              flush=True)

    model = model.to(device=device, dtype=dtype).eval()
    input_ids, attn = input_ids.to(device), attn.to(device)

    # Arm the loop-collapse hook (last_loop_update_norm) + per-MoE-layer routing
    # telemetry so a SINGLE forward yields all three triangulation signals on the
    # same batch: KV-latent CKA, residual-update norm, per-loop routing entropy.
    model.set_moe_telemetry(True)

    # ---- forward, capture per-(loop, block) latents ------------------------
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    latents = out.past_key_values  # list of (B, S, kv_dim), len = blocks*loops
    n_blocks, n_loops = cfg.num_blocks, cfg.recursive_loops
    assert len(latents) == n_blocks * n_loops, (
        f"expected {n_blocks * n_loops} latents, got {len(latents)}")

    # Telemetry from the same forward. update_norm is indexed per effective layer
    # (idx = loop*n_blocks + block_idx, like the latents); routing entropy lives
    # per loop on each physical block's MoE.
    inner = model.model
    upd_norm = list(inner.last_loop_update_norm)  # len n_blocks*n_loops

    def _moe_stat(b: int, name: str) -> list[float]:
        return [round(float(v), 4) for v in getattr(inner.blocks[b].moe, name)]

    # Guard against an all-padding/empty batch: _masked_rows would return empty
    # tensors and every metric would be NaN / divide-by-zero.
    valid_tokens = int(attn.sum().item())
    if valid_tokens == 0:
        print("ERROR: no valid (non-padding) tokens in the input batch.", flush=True)
        return 1

    # idx = loop * n_blocks + block_idx  -> per_block[block][loop]
    per_block: list[list] = [[None] * n_loops for _ in range(n_blocks)]
    for idx, lat in enumerate(latents):
        loop, block = idx // n_blocks, idx % n_blocks
        per_block[block][loop] = _masked_rows(lat, attn)

    # ---- metrics -----------------------------------------------------------
    report: dict = {
        "config": {"num_blocks": n_blocks, "num_loops": n_loops,
                   "kv_dim": int(latents[0].shape[-1]),
                   "valid_tokens": valid_tokens,
                   "random_init": bool(args.random_init), "tiny": bool(args.tiny)},
        "blocks": [],
    }

    print("\n" + "=" * 72)
    print("PER-PHYSICAL-BLOCK CROSS-LOOP LATENT SIMILARITY")
    print("=" * 72)
    for b in range(n_blocks):
        loops = per_block[b]
        cka = [[round(linear_cka(loops[i], loops[j]), 4) for j in range(n_loops)]
               for i in range(n_loops)]
        cos0 = [round(mean_cosine(loops[k], loops[0]), 4) for k in range(n_loops)]

        # Scheme 1: first-loop share — every loop reuses loop 0.
        first_err = [round(rel_l2(loops[k], loops[0]), 4) for k in range(1, n_loops)]
        first_mean = round(sum(first_err) / max(len(first_err), 1), 4)

        # Scheme 2: two-group split at h = L//2 — loops [0..h) reuse loop 0,
        # loops [h..L) reuse loop h (caches 2 of L; == "recompute every L/2
        # loops", i.e. g=3 only in the default L=6 case).
        h = n_loops // 2
        grouped_err = (
            [rel_l2(loops[k], loops[0]) for k in range(1, h)]
            + [rel_l2(loops[k], loops[h]) for k in range(h + 1, n_loops)]
        )
        grouped_mean = round(sum(grouped_err) / max(len(grouped_err), 1), 4)

        # Contraction series: adjacent-loop CKA(k, k+1) and the per-step KV move
        # size 1-CKA. A monotonically shrinking move size is the fixed-point /
        # DEQ-like contraction signature (the headline mechanistic claim).
        adj_cka = [cka[k][k + 1] for k in range(n_loops - 1)]
        kv_move = [round(1.0 - a, 4) for a in adj_cka]

        # Triangulation signals from the SAME forward, regrouped per loop:
        #  - residual update |dx|/|x| at this (block, loop) effective layer
        #  - per-loop routing entropy (marginal = balance, per-token = sharpness)
        upd = [round(float(upd_norm[loop * n_blocks + b]), 4)
               for loop in range(n_loops)]
        route_marg = _moe_stat(b, "last_marginal_entropy")
        route_tok = _moe_stat(b, "last_per_token_entropy")

        block_report = {
            "block": b, "cka_matrix": cka, "cosine_vs_loop0": cos0,
            "first_share_rel_l2": first_err, "first_share_rel_l2_mean": first_mean,
            "two_group_split_rel_l2_mean": grouped_mean,
            "adjacent_cka": adj_cka, "kv_move_size": kv_move,
            "loop_update_norm": upd, "route_marginal_entropy": route_marg,
            "route_per_token_entropy": route_tok,
        }
        report["blocks"].append(block_report)

        print(f"\n-- physical block {b} --")
        print(f"  cosine(loop_k, loop_0): {cos0}")
        print(f"  CKA(loop_k, loop_0):    "
              f"{[cka[0][k] for k in range(n_loops)]}")
        print(f"  contraction — KV move 1-CKA(k,k+1): {kv_move}")
        print(f"  triangulate — update |dx|/|x|/loop: {upd}")
        print(f"  triangulate — route entropy (marg): {route_marg}")
        print(f"  injected rel-L2 if FIRST-LOOP SHARE (caches 1 of {n_loops} "
              f"= {n_loops}x): per-loop {first_err} => mean {first_mean}")
        print(f"  injected rel-L2 if 2-GROUP SPLIT (caches 2 of {n_loops} "
              f"= {n_loops / 2:.1f}x): mean {grouped_mean}")

    # ---- decision hint -----------------------------------------------------
    # Average CKA(loop_k, loop_0) over k=1..L-1 ONLY — excluding the self term
    # CKA(loop_0, loop_0)=1.0, which would otherwise inflate the gate (e.g. five
    # real 0.66s would report as (1+5*0.66)/6=0.72 and trip the >0.70 hint).
    denom = max(n_loops - 1, 1)
    mean_cka0 = sum(
        sum(blk["cka_matrix"][0][1:]) / denom for blk in report["blocks"]
    ) / n_blocks
    mean_first = sum(b["first_share_rel_l2_mean"] for b in report["blocks"]) / n_blocks
    report["summary"] = {"mean_cka_vs_loop0": round(mean_cka0, 4),
                         "mean_first_share_rel_l2": round(mean_first, 4)}

    print("\n" + "=" * 72)
    print(f"SUMMARY  mean CKA(*,loop0)={mean_cka0:.3f}  "
          f"mean first-share rel-L2={mean_first:.3f}")
    if args.random_init:
        print("  (random-init: ignore the numbers; this only proves the probe runs.)")
    elif mean_cka0 > 0.90 and mean_first < 0.10:
        print("  HINT: loops are near-collinear -> first-loop share (6x) looks cheap.")
    elif mean_cka0 > 0.70:
        print("  HINT: partial overlap -> grouped g=L/2 (~3x) is the safe middle; "
              "re-check at the reasoning-on>off gate.")
    else:
        print("  HINT: loops carry distinct KV (recursion is refining) -> prefer "
              "low-rank per-loop delta; do NOT full-share.")
    print("=" * 72)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
