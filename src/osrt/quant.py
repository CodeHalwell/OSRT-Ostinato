"""TurboQuant-style int4 quantization for the MLA KV-cache latent.

ARCHITECTURE.md §13.3 / §14.1: at deployment the per-effective-layer K_DOWN
latent (the ONLY thing cached — V is recomputed from it) is compressed from
bf16 to int4 with a TurboQuant random-rotation + symmetric int4 quantizer. This
pairs with the MLA latent cache to hit the §13.3 deployment KV-cache size
targets (~9-18 MB at 4K context for the K-only baseline, ~2-5 MB with a
sliding window on top).

This is a STANDALONE deployment / RL-rollout utility. It is NOT wired into the
training forward (training keeps the full-precision latent) and is not enabled
in generate() by default — the model code is unchanged. A caller that wants a
quantized rollout cache quantizes each layer's latent with quantize_kv_latent()
and dequantizes it back with dequantize_kv_latent() before feeding it to the
attention math, e.g. in an offline rollout collector.

Why a random rotation (the "Turbo" in TurboQuant)
-------------------------------------------------
Symmetric int4 has only 15 usable levels; a single large-magnitude channel in a
block forces a big quantization step and wastes precision on the many small
channels (the classic outlier problem in low-bit KV quantization). A fixed
random ORTHOGONAL rotation applied before quantization mixes every channel into
every other, so a lone outlier is spread across the block and the per-block max
(which sets the step) drops toward the RMS rather than the peak. Because the
rotation is orthogonal it is exactly invertible: dequantize rotates back with
the transpose. We use a normalized Hadamard rotation when the block size is a
power of two and fall back to a seeded random orthogonal matrix (QR of a
Gaussian) otherwise. Both are currently materialized as a dense n×n block matrix
and applied by matmul (O(n²) per block — fine at the small block sizes used
here; a fast Walsh–Hadamard transform would make the power-of-two path
O(n log n) if larger blocks are ever needed). The matrix is rebuilt on demand
from a fixed seed rather than persisted, so encode and decode agree with no side
channel and nothing matrix-sized is stored.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch
from torch import Tensor

# int4 symmetric range. We use the symmetric 15-level grid [-7, 7] (dropping
# the asymmetric -8 level) so that quantize(-x) == -quantize(x) exactly — the
# rotation produces zero-mean, near-symmetric blocks, and a symmetric grid
# avoids a half-step DC bias. Codes are stored in an int8 tensor (one int4
# value per byte); see pack_int4 / unpack_int4 for the optional 2-per-byte
# packing used when on-disk size actually matters.
INT4_QMAX = 7


# ── Random rotation (Hadamard-style, seeded & invertible) ───────────────────


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _hadamard_matrix(n: int, device, dtype) -> Tensor:
    """Dense normalized Sylvester-Hadamard matrix H_n (n a power of two).

    H is symmetric and orthogonal: H @ H.T == I. Normalizing by 1/sqrt(n)
    makes the transform norm-preserving. Built once per (n, device, dtype);
    n is the block size (<= a few hundred) so the n×n matrix is tiny.
    """
    assert _is_pow2(n), "Hadamard size must be a power of two"
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0,
        )
    return h / math.sqrt(n)


@functools.lru_cache(maxsize=32)
def _cached_rotation_cpu(dim: int, seed: int) -> Tensor:
    """CPU-side fixed orthogonal rotation matrix build."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if _is_pow2(dim):
        h = _hadamard_matrix(dim, device="cpu", dtype=torch.float32)
        signs = torch.randint(
            0, 2, (dim,), generator=gen, dtype=torch.float32,
        ) * 2.0 - 1.0  # ±1
        r = h * signs.view(1, -1)
    else:
        a = torch.randn((dim, dim), generator=gen, dtype=torch.float32)
        q, _ = torch.linalg.qr(a)
        r = q
    return r


def make_rotation(dim: int, seed: int, device=None, dtype=torch.float32) -> Tensor:
    """Build a fixed, seeded orthogonal rotation matrix of shape (dim, dim).

    Power-of-two dim: a randomized Hadamard rotation — a normalized Sylvester
    Hadamard matrix with a random ±1 diagonal sign flip (the standard cheap
    random rotation; the signs are what randomize it so a fixed Hadamard
    doesn't align with the data axes). Otherwise: the Q factor of a QR
    decomposition of a seeded Gaussian, a general random orthogonal matrix.
    Either way the result R satisfies R @ R.T == I, so the inverse rotation is
    just R.T — dequantization needs no separately stored inverse.
    """
    r = _cached_rotation_cpu(dim, seed)
    return r.to(device=device, dtype=dtype)


# ── int4 pack / unpack (optional 2-values-per-byte storage) ─────────────────


def pack_int4(codes: Tensor) -> Tensor:
    """Pack an int8 tensor of int4 values (last dim even) into uint8, two
    nibbles per byte. Halves the on-disk/in-RAM footprint vs one-per-byte.
    Values are stored two's-complement in [-7, 7] mapped to nibbles [0, 15].
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError(
            f"pack_int4 needs an even last dim, got {codes.shape[-1]}"
        )
    nib = (codes.to(torch.int16) & 0x0F).to(torch.uint8)  # 4-bit two's complement
    low = nib[..., 0::2]
    high = nib[..., 1::2]
    return (low | (high << 4)).to(torch.uint8)


def unpack_int4(packed: Tensor, last_dim: int) -> Tensor:
    """Inverse of pack_int4. Returns an int8 tensor with signed values in
    [-7, 7] and last dim == last_dim."""
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    # Sign-extend the 4-bit two's-complement nibble back to int8.
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    out = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    return out[..., :last_dim]


# ── Quantized-latent container ──────────────────────────────────────────────


@dataclass
class QuantizedKV:
    """Round-trippable int4 representation of one cached latent tensor.

    codes  : int8, the int4 quantization codes in [-7, 7], same shape as the
             rotated input (one value per element; pack with pack_int4 for the
             2-per-byte on-disk form).
    scale  : float32 per-block step size, shape (..., n_blocks, 1).
    seed   : the rotation seed (so dequantize rebuilds the exact rotation).
    block_size : last-dim block length the per-block scale was computed over.
    orig_dim   : original last-dim size before any block padding.
    rotated    : whether a random rotation was applied (False = plain int4).
    """

    codes: Tensor
    scale: Tensor
    seed: int
    block_size: int
    orig_dim: int
    rotated: bool


# ── Public API ──────────────────────────────────────────────────────────────


def quantize_kv_latent(
    x: Tensor,
    block_size: int | None = None,
    seed: int = 0,
    rotate: bool = True,
) -> QuantizedKV:
    """Quantize a KV latent (..., kv_dim) to symmetric int4 with a TurboQuant
    random rotation.

    Steps:
      1. (optional) rotate the LAST dim by a fixed seeded orthogonal matrix,
         spreading outliers so the per-block max — and hence the int4 step —
         shrinks toward the block RMS.
      2. split the last dim into blocks of `block_size` (default: the whole
         last dim, i.e. per-tensor-row / per-token-per-layer), compute a
         symmetric per-block scale = max|.| / 7, and round to the int4 grid.

    Returns a QuantizedKV; pair with dequantize_kv_latent() to reconstruct.
    The rotation is applied per block (block_size must then match the rotation
    dim), so for a whole-row rotation leave block_size=None.
    """
    if x.dim() < 1:
        raise ValueError("quantize_kv_latent expects a tensor with a last dim")
    orig_dim = x.shape[-1]
    bs = orig_dim if block_size is None else block_size
    if orig_dim % bs != 0:
        raise ValueError(
            f"last dim {orig_dim} must be divisible by block_size {bs}"
        )

    work = x.float()
    if rotate:
        # Rotate each block by the same seeded R (block_size × block_size).
        r = make_rotation(bs, seed=seed, device=work.device, dtype=torch.float32)
        lead = work.shape[:-1]
        blocked = work.reshape(*lead, orig_dim // bs, bs)
        blocked = blocked @ r  # (..., n_blocks, bs) rotated along the last dim
        work = blocked
    else:
        work = work.reshape(*work.shape[:-1], orig_dim // bs, bs)

    # Symmetric per-block scale. clamp_min avoids a zero step for an all-zero
    # block (which would divide by zero); such a block quantizes to all zeros.
    amax = work.abs().amax(dim=-1, keepdim=True)
    scale = (amax / INT4_QMAX).clamp_min(1e-12)
    codes = torch.round(work / scale).clamp_(-INT4_QMAX, INT4_QMAX).to(torch.int8)

    return QuantizedKV(
        codes=codes,
        scale=scale,
        seed=seed,
        block_size=bs,
        orig_dim=orig_dim,
        rotated=rotate,
    )


def dequantize_kv_latent(q: QuantizedKV) -> Tensor:
    """Reconstruct the (..., kv_dim) latent from a QuantizedKV.

    Inverse of quantize_kv_latent: scale the int4 codes back up, then (if the
    encoder rotated) un-rotate with R.T — R is orthogonal so its transpose is
    its inverse and is rebuilt from the stored seed. Returns a float32 tensor
    of the original shape.
    """
    work = q.codes.float() * q.scale  # (..., n_blocks, block_size)
    if q.rotated:
        r = make_rotation(
            q.block_size, seed=q.seed, device=work.device, dtype=torch.float32,
        )
        work = work @ r.T  # un-rotate (R orthogonal ⇒ R^{-1} == R^T)
    lead = work.shape[:-2]
    x_hat = work.reshape(*lead, q.orig_dim)
    return x_hat


def kv_quant_rel_error(x: Tensor, q: QuantizedKV) -> float:
    """Mean relative reconstruction error ‖x - x_hat‖ / ‖x‖ (Frobenius), a
    convenience for tests / deployment calibration."""
    x_hat = dequantize_kv_latent(q)
    num = (x.float() - x_hat).norm().item()
    den = x.float().norm().clamp_min(1e-12).item()
    return num / den
