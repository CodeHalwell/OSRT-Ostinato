"""Unit tests for the TurboQuant int4 KV-cache quantizer (osrt.quant).

Covers ARCHITECTURE.md §13.3 / §14.1: a random-rotation + symmetric int4
quantize/dequantize round-trip on the MLA K_DOWN latent, the orthogonality of
the seeded rotation, the int4 pack/unpack helper, and — the point of the
"Turbo" rotation — that the random rotation reduces reconstruction error on
heavy-tailed (outlier-heavy) latents versus quantizing without it.
"""

import torch

from osrt.quant import (
    INT4_QMAX,
    QuantizedKV,
    dequantize_kv_latent,
    kv_quant_rel_error,
    make_rotation,
    pack_int4,
    quantize_kv_latent,
    unpack_int4,
)


def _mean_row_rel_error(x: torch.Tensor, q: QuantizedKV) -> float:
    """Mean over rows of the per-row relative L2 reconstruction error."""
    x_hat = dequantize_kv_latent(q)
    num = (x.float() - x_hat).pow(2).sum(dim=-1).sqrt()
    den = x.float().pow(2).sum(dim=-1).sqrt().clamp_min(1e-12)
    return (num / den).mean().item()


# ── Rotation properties ─────────────────────────────────────────────────


def test_rotation_is_orthogonal_pow2_and_nonpow2():
    """make_rotation returns an orthogonal matrix (R @ R.T == I) on both the
    Hadamard (power-of-two) path and the QR (general) path."""
    for dim in (512, 384):  # 512 = Hadamard, 384 = QR fallback
        r = make_rotation(dim, seed=0)
        ident = r @ r.T
        err = (ident - torch.eye(dim)).abs().max().item()
        assert err < 1e-4, f"rotation not orthogonal at dim={dim}: {err}"


def test_rotation_is_deterministic_given_seed():
    """Same seed → same rotation (encode/decode must agree without storing R)."""
    a = make_rotation(256, seed=42)
    b = make_rotation(256, seed=42)
    c = make_rotation(256, seed=43)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ── Round-trip numerical accuracy ───────────────────────────────────────


def test_int4_roundtrip_small_error_on_normal_latents():
    """Symmetric int4 + rotation round-trips a normal-distributed latent with
    small relative error. 4-bit (15 levels) has an inherent ~0.10-0.13 floor
    on Gaussian data; a per-block scale keeps us at the low end. Codes must
    live on the symmetric int4 grid stored as int8."""
    torch.manual_seed(0)
    x = torch.randn(64, 16, 512)  # (B, S, kv_dim) — a per-token-per-layer latent
    q = quantize_kv_latent(x, block_size=64, seed=0)
    assert q.codes.dtype == torch.int8
    assert int(q.codes.min()) >= -INT4_QMAX and int(q.codes.max()) <= INT4_QMAX
    err = _mean_row_rel_error(x, q)
    assert err < 0.15, f"int4 round-trip error too high: {err}"
    # Reconstruction preserves shape and is finite.
    x_hat = dequantize_kv_latent(q)
    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()


def test_whole_row_quantization_default_block():
    """block_size=None quantizes over the whole last dim (per token, per
    effective layer) and still round-trips at the 4-bit floor."""
    torch.manual_seed(1)
    x = torch.randn(32, 512)
    q = quantize_kv_latent(x, seed=3)
    assert q.block_size == 512
    err = kv_quant_rel_error(x, q)
    assert err < 0.2, f"whole-row int4 error too high: {err}"


def test_rotation_improves_heavy_tailed_quantization():
    """The defining TurboQuant property: on a heavy-tailed (outlier-heavy)
    latent, the random rotation spreads the outlier energy so the per-block
    int4 step fits the bulk, beating un-rotated int4. Student-t (df=2.1) is
    reliably fat-tailed."""
    torch.manual_seed(0)
    t = torch.distributions.StudentT(2.1)
    x = t.sample((128, 256))

    q_rot = quantize_kv_latent(x, seed=11, rotate=True)
    q_norot = quantize_kv_latent(x, seed=11, rotate=False)
    err_rot = _mean_row_rel_error(x, q_rot)
    err_norot = _mean_row_rel_error(x, q_norot)

    assert err_rot < err_norot, (
        f"rotation did not help on heavy-tailed input: "
        f"rotated={err_rot:.4f} vs no-rotation={err_norot:.4f}"
    )
    # And the win should be substantial, not marginal noise.
    assert err_rot < 0.7 * err_norot, (
        f"rotation gain too small: rotated={err_rot:.4f} "
        f"no-rotation={err_norot:.4f}"
    )


def test_all_zero_block_quantizes_without_nan():
    """An all-zero block must not divide by zero — it quantizes to all zeros
    and reconstructs to (approximately) zero."""
    x = torch.zeros(4, 128)
    q = quantize_kv_latent(x, seed=0)
    x_hat = dequantize_kv_latent(q)
    assert torch.isfinite(x_hat).all()
    assert x_hat.abs().max().item() < 1e-6


# ── int4 packing (2 values per byte) ────────────────────────────────────


def test_pack_unpack_int4_roundtrip():
    """pack_int4 / unpack_int4 are exact inverses and halve the storage."""
    torch.manual_seed(0)
    x = torch.randn(8, 256)
    q = quantize_kv_latent(x, seed=0)
    packed = pack_int4(q.codes)
    assert packed.dtype == torch.uint8
    assert packed.numel() == q.codes.numel() // 2  # 2 nibbles per byte
    restored = unpack_int4(packed, q.codes.shape[-1])
    assert torch.equal(restored, q.codes)


def test_quantized_dequantized_pipeline_via_packing():
    """End-to-end: quantize → pack → unpack → dequantize reproduces the same
    latent as the un-packed path (packing is lossless)."""
    torch.manual_seed(2)
    x = torch.randn(16, 512)
    q = quantize_kv_latent(x, block_size=64, seed=7)
    direct = dequantize_kv_latent(q)
    # codes are (..., n_blocks, block_size); pack/unpack act on the last
    # (block) dim, so unpack restores block_size, not orig_dim.
    packed = pack_int4(q.codes)
    q2 = QuantizedKV(
        codes=unpack_int4(packed, q.block_size),
        scale=q.scale, seed=q.seed, block_size=q.block_size,
        orig_dim=q.orig_dim, rotated=q.rotated,
    )
    via_pack = dequantize_kv_latent(q2)
    assert torch.equal(direct, via_pack)
