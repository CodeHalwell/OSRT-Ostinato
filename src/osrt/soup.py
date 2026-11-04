"""Checkpoint soup: the fp32 mean of model weights (scripts/eval_trunk.py)."""

from __future__ import annotations

import torch


def average_state_dicts(sds: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Mean of every floating tensor across `sds`; non-float tensors (buffers,
    counters) are taken from the last one. Keys must match."""
    if not sds:
        raise ValueError("nothing to soup")
    keys = set(sds[-1])
    if any(set(sd) != keys for sd in sds):
        raise ValueError("state dicts differ in keys")
    out = {}
    for k, v in sds[-1].items():
        if v.is_floating_point():
            out[k] = (sum(sd[k].float() for sd in sds) / len(sds)).to(v.dtype)
        else:
            out[k] = v.clone()
    return out
