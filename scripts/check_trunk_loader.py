"""Does `main` compute the trunk's forward function?

The trunk (branch `trunk-pinned`, commit 1db3d89) predates `router_bias_in_gates`
and the fp32 router. This check builds the SAME tiny model in both trees
(each from its own OSRT_V7 preset, so SiTU-GLU, Quantile Balancing and the
sqrt-softplus affinity are all exercised), copies the pinned weights into
`main`, and compares logits and loss in eval mode on identical inputs:

  * with `router_bias_in_gates=True`  -> must match (that is the loader setting)
  * with `router_bias_in_gates=False` -> must NOT match (proves the flag matters)

    uv run python scripts/check_trunk_loader.py [--pinned ~/osrt-trunk-pinned]

Exit code 0 iff the legacy-gates build reproduces the pinned forward.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

TINY = dict(
    dim=64, heads=2, head_dim=32, num_kv_heads=1,
    vocab_size=128, real_vocab_size=100,
    num_blocks=1, recursive_loops=3,
    num_routed_experts=4, top_k_experts=2,
    expert_hidden=64, shared_expert_hidden=64,
    max_position_embeddings=64,
)

PINNED_SIDE = r'''
import json, sys, torch
from osrt.presets import build_config
from osrt.model import OSRTForCausalLM
tiny = json.loads(sys.argv[1]); out = sys.argv[2]
torch.manual_seed(0)
cfg = build_config(**tiny)
model = OSRTForCausalLM(cfg).eval()
# A fresh router has a ZERO balancing bias, under which both gate semantics
# coincide. Give every MoE bias buffer a trained-looking value (the trunk's
# bias_abs_max ran 0.2-0.4) so the check can tell the two apart.
g = torch.Generator().manual_seed(1)
n_pert = 0
for name, buf in model.named_buffers():
    if ".moe." in name and "bias" in name and buf.is_floating_point():
        buf.copy_(0.3 * torch.randn(buf.shape, generator=g))
        n_pert += 1
print("pinned: perturbed router bias buffers:", n_pert)
ids = torch.randint(0, tiny["real_vocab_size"], (2, 24), generator=g)
with torch.no_grad():
    o = model(ids, labels=ids)
torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict(),
            "input_ids": ids, "logits": o.logits.float(),
            "loss": float(o.loss)}, out)
print("pinned: params", sum(p.numel() for p in model.parameters()),
      "loss", float(o.loss))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", default=os.path.expanduser("~/osrt-trunk-pinned"))
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    import torch

    from osrt.config import OSRTConfig
    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config

    with tempfile.TemporaryDirectory() as td:
        ref_path = os.path.join(td, "pinned.pt")
        env = {**os.environ, "PYTHONPATH": os.path.join(args.pinned, "src")}
        cmd = [sys.executable, "-c", PINNED_SIDE, json.dumps(TINY), ref_path]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            return 2
        print(r.stdout.strip())
        ref = torch.load(ref_path, map_location="cpu", weights_only=False)

    # Keep only the config fields main knows; the trunk's dict predates
    # `router_bias_in_gates` and `fused_main_head`.
    known = set(OSRTConfig.__init__.__code__.co_varnames)
    base = {k: v for k, v in ref["config"].items() if k in known}
    dropped = sorted(set(ref["config"]) - known)
    if dropped:
        print("pinned config keys unknown to main (dropped):", dropped)

    verdicts = {}
    for legacy in (True, False):
        torch.manual_seed(0)
        cfg = build_config(**{**base, "router_bias_in_gates": legacy})
        model = OSRTForCausalLM(cfg).eval()
        missing, unexpected = model.load_state_dict(ref["state_dict"], strict=False)
        with torch.no_grad():
            o = model(ref["input_ids"], labels=ref["input_ids"])
        dl = (o.logits.float() - ref["logits"]).abs().max().item()
        dloss = abs(float(o.loss) - ref["loss"])
        match = dl <= args.atol and dloss <= args.atol
        verdicts[legacy] = match
        print(f"main router_bias_in_gates={legacy!s:5s} | missing={len(missing)} "
              f"unexpected={len(unexpected)} | max|dlogits|={dl:.2e} "
              f"|dloss|={dloss:.2e} -> {'MATCH' if match else 'DIFFERS'}")

    ok = verdicts[True] and not verdicts[False]
    print("VERDICT:", "legacy-gates build reproduces the trunk forward"
          if ok else "FAILED — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
