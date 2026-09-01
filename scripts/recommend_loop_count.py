"""Turn the cross-loop probe into a decision: how many loops to run at inference.

Reads the JSON that scripts/probe_cross_loop_kv.py writes and recommends the
smallest loop count K such that every loop beyond K is measurably a no-op on
THREE independent signals from the same forward pass:

  1. KV move size  1 - CKA(k, k+1)  — the representation stopped changing
  2. residual update ||dx||/||x||     — the layer stopped writing
  3. routing entropy drift           — the router stopped re-deciding

Requiring all three guards against the failure mode where one signal is
flat for a boring reason (e.g. CKA saturates because the latent is
low-rank, not because the loop is idle).

This is the §17.2 idea: the contracting-iteration measurement as a tool,
not just a paper figure. Decode latency is ~depth-bound (§13.4), so trimming
loops 6 -> 4 is a ~33% latency cut IF the later loops are genuinely idle.

Usage:
    python scripts/recommend_loop_count.py research/probe_cross_loop_kv_general.json
    python scripts/recommend_loop_count.py report.json --kv-tol 0.08 --upd-tol 0.02
"""
from __future__ import annotations

import argparse
import json
import sys


def recommend(report: dict, kv_tol: float, upd_tol: float, ent_tol: float) -> dict:
    n_loops = report["config"]["num_loops"]
    verdicts = []
    for blk in report["blocks"]:
        # len n_loops-1, index k = move k->k+1
        move = blk["kv_move_size"]
        upd = blk["loop_update_norm"]                  # len n_loops
        ent = blk["route_marginal_entropy"]            # len n_loops
        # loop k (0-indexed) is "idle" if the move INTO it was tiny, its own
        # residual write was tiny, and routing entropy barely changed.
        idle = []
        for k in range(1, n_loops):
            idle.append(
                move[k - 1] < kv_tol
                and upd[k] < upd_tol
                and abs(ent[k] - ent[k - 1]) < ent_tol
            )
        # K = first loop index from which ALL later loops are idle
        keep = n_loops
        for k in range(n_loops - 1, 0, -1):
            if idle[k - 1]:
                keep = k
            else:
                break
        verdicts.append({"block": blk["block"], "keep_loops": keep,
                         "idle_flags_loops_1_to_L": idle,
                         "kv_move": move, "upd": upd})
    # The model runs every block each loop, so K is the MAX over blocks.
    K = max(v["keep_loops"] for v in verdicts)
    return {
        "recommended_loops": K,
        "trained_loops": n_loops,
        "decode_latency_saving": round(1 - K / n_loops, 3),
        "per_block": verdicts,
        "thresholds": {"kv_move": kv_tol, "update_norm": upd_tol,
                       "entropy_drift": ent_tol},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--kv-tol", type=float, default=0.08,
                    help="max 1-CKA(k,k+1) to call a loop idle")
    ap.add_argument("--upd-tol", type=float, default=0.02,
                    help="max ||dx||/||x|| to call a loop idle")
    ap.add_argument("--ent-tol", type=float, default=0.05,
                    help="max routing-entropy change to call a loop idle")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    rep = json.load(open(a.report))
    out = recommend(rep, a.kv_tol, a.upd_tol, a.ent_tol)
    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    L, K = out["trained_loops"], out["recommended_loops"]
    print(f"trained with {L} loops; recommend running {K} at inference "
          f"({out['decode_latency_saving']*100:.0f}% of the depth term removed)")
    for v in out["per_block"]:
        flags = "".join("." if f else "#" for f in v["idle_flags_loops_1_to_L"])
        print(f"  block {v['block']}: keep {v['keep_loops']}  loops1..{L-1} "
              f"[{flags}]  (# active, . idle)")
        print(f"           kv move {v['kv_move']}")
        print(f"           |dx|/|x| {v['upd']}")
    if K == L:
        print("no loop is idle on all three signals — do not trim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
