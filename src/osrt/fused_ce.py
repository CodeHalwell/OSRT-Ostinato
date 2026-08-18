"""Chunked fused linear-cross-entropy (architecture review item B2).

The OSRT training step projects the hidden state to a 65 536-way vocabulary and
upcasts to fp32 for cross-entropy — once for the main +1 head, once per MTP head,
and once per intermediate per-loop aux head. At seq_len 8192 each such
``(B, S, vocab)`` fp32 logit tensor is multiple GB, and the aux/MTP heads add up
to ~7 of them, dominating activation memory.

``fused_linear_cross_entropy`` computes the identical loss while only ever
materialising one chunk of the ``(N, vocab)`` logits at a time. Each chunk's
linear + cross-entropy is gradient-checkpointed (``use_reentrant=False``), so the
per-chunk logits are recomputed in backward instead of being retained — peak logit
memory drops from O(N·vocab) to O(chunk·vocab). Gradients come straight from
autograd (no hand-written backward), so they are correct by construction.

This is opt-in (a config flag, default off) so the standard path stays
bit-identical; the parity tests in ``tests/test_fused_ce.py`` prove turning it on
does not change the loss or the gradients.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def fused_linear_cross_entropy(
    hidden: Tensor,
    weight: Tensor,
    labels: Tensor,
    real_vocab_size: int,
    ignore_index: int = -100,
    n_chunks: int = 8,
) -> Tensor:
    """Memory-frugal equivalent of::

        F.cross_entropy(
            F.linear(hidden, weight[:real_vocab_size]).float(),
            labels, ignore_index=ignore_index,
        )

    Args:
        hidden: (N, D) hidden states, already flattened and shifted by the caller.
        weight: (vocab_pad, D) tied embedding weight; only the first
            ``real_vocab_size`` rows participate (matches the model's logit slice).
        labels: (N,) long targets, may contain ``ignore_index``.
        real_vocab_size: number of real (non-padding) vocab rows.
        ignore_index: target value to skip (default -100).
        n_chunks: number of row-chunks to split the loss over. More chunks = lower
            peak logit memory, slightly more Python overhead. 1 = no chunking.

    Returns:
        Scalar fp32 loss = mean cross-entropy over non-ignored tokens. An
        all-ignored batch returns a finite ``0.0`` (per-chunk sum-reduction is 0),
        which is safer than ``F.cross_entropy``'s NaN for degenerate micro-batches.
    """
    if hidden.dim() != 2:
        raise ValueError(f"hidden must be (N, D), got {tuple(hidden.shape)}")
    n = hidden.shape[0]
    w = weight[:real_vocab_size]

    valid = (labels != ignore_index).sum()
    denom = valid.clamp_min(1).float()
    total = hidden.new_zeros((), dtype=torch.float32)

    chunk = max(1, math.ceil(n / max(1, n_chunks)))

    def _chunk_ce(h_chunk: Tensor, w_full: Tensor, lbl: Tensor) -> Tensor:
        logits = F.linear(h_chunk, w_full).float()
        return F.cross_entropy(
            logits, lbl, ignore_index=ignore_index, reduction="sum",
        )

    for start in range(0, n, chunk):
        h_c = hidden[start:start + chunk]
        l_c = labels[start:start + chunk]
        if torch.is_grad_enabled() and (h_c.requires_grad or w.requires_grad):
            # Checkpoint so the chunk's logits are recomputed in backward rather
            # than retained — this is what bounds peak memory.
            ce_sum = checkpoint(_chunk_ce, h_c, w, l_c, use_reentrant=False)
        else:
            ce_sum = _chunk_ce(h_c, w, l_c)
        total = total + ce_sum

    return total / denom
