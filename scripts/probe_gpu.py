"""Report what the attached GPU can actually do, and what fits (gate G7, part 1).

Answers three questions that decide v7's plan:

1. WHICH card. "RTX 6000" is ambiguous: the RTX 6000 Ada (AD102, sm_89, 48GB)
   has FP8 but NO NVFP4; the RTX PRO 6000 Blackwell (GB202, sm_120, 96GB) has
   both. Roadmap §13.3 rests on NVFP4 training being available.
2. WHETHER THE EXPERT PATH CAN GO LOW-PRECISION. Routed experts are ~84% of
   v7's parameters and run through the private torch._grouped_mm. Transformer
   Engine covers dense/attention cleanly; the grouped path is the open
   question, and G7's decision rule hangs on it.
3. WHAT FITS. Prints the optimizer/weights floor for the committed shape and
   the activation headroom left on this card.

Usage:  PYTHONPATH=src python scripts/probe_gpu.py
"""
from __future__ import annotations

import torch


def _fmt(b: float) -> str:
    return f"{b / 1024**3:.2f} GB"


def main() -> None:
    if not torch.cuda.is_available():
        print("No CUDA device. Run this on the GPU box / Colab runtime.")
        return

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"device        : {name}")
    print(f"capability    : sm_{major}{minor}")
    print(f"VRAM          : {_fmt(total)}")
    print(f"torch         : {torch.__version__}")

    if (major, minor) >= (12, 0):
        verdict = "Blackwell — NVFP4 available (roadmap §13.3 applies)"
    elif (major, minor) == (8, 9):
        verdict = "Ada — FP8 yes, NVFP4 NO. §13.3's NVFP4 case does not apply here"
    elif major >= 8:
        verdict = "Ampere — bf16 yes, no FP8/FP4"
    else:
        verdict = "pre-Ampere — NO bf16; this codebase will not run"
    print(f"verdict       : {verdict}")

    print("\n── dtype support ──")
    for label, dt in (("bf16", torch.bfloat16),
                      ("fp8 e4m3", getattr(torch, "float8_e4m3fn", None)),
                      ("fp8 e5m2", getattr(torch, "float8_e5m2", None))):
        if dt is None:
            print(f"  {label:<10} not in this torch build")
            continue
        try:
            torch.zeros(8, 8, dtype=dt, device="cuda")
            print(f"  {label:<10} allocates OK")
        except Exception as e:  # noqa: BLE001
            print(f"  {label:<10} FAILED: {type(e).__name__}")

    print("\n── grouped-GEMM (the expert path, G7) ──")
    gmm = getattr(torch, "_grouped_mm", None)
    if gmm is None:
        print("  torch._grouped_mm ABSENT — the private API this model's MoE "
              "dispatch depends on is gone. Portability risk realised.")
    else:
        for label, dt in (("bf16", torch.bfloat16),
                          ("fp8 e4m3", getattr(torch, "float8_e4m3fn", None))):
            if dt is None:
                continue
            try:
                a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16).to(dt)
                b = torch.randn(2, 128, 96, device="cuda", dtype=torch.bfloat16).to(dt)
                offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)
                _ = gmm(a, b, offs=offs)
                print(f"  _grouped_mm {label:<9} OK")
            except Exception as e:  # noqa: BLE001
                msg = str(e).split("\n")[0][:80]
                print(f"  _grouped_mm {label:<9} unsupported — "
                      f"{type(e).__name__}: {msg}")

    print("\n── what fits: OSRT_V7 committed shape ──")
    try:
        from osrt.model import OSRTForCausalLM
        from osrt.presets import build_config
        cfg = build_config()
        with torch.device("meta"):
            m = OSRTForCausalLM(cfg)
        tot = sum(p.numel() for p in m.parameters())
        emb = sum(p.numel() for n, p in m.named_parameters()
                  if "embedding" in n or "lm_head" in n)
        hidden = tot - emb
        floor = tot * 4 + tot * 4 + hidden * 4 + emb * 8   # master+grad+Muon+AdamW
        print(f"  params        : {tot:,}")
        print(f"  optim+weights : {_fmt(floor)}")
        head = total - floor
        print(f"  headroom      : {_fmt(head)} for activations + workspace")
        if head < 8 * 1024**3:
            print("  WARNING: under 8GB of headroom — expect batch 1 or OOM.")
    except Exception as e:  # noqa: BLE001
        print(f"  budget unavailable: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
