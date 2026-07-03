'''
Testing RMSNorm, DynamicTanh, and DynamicErf implementations
'''

import torch

from src.transformers.normalization.rmsnorm import RMSNorm
from src.transformers.normalization.dyt import DynamicTanh
from src.transformers.normalization.derf import DynamicErf


# RMSNorm 

def test_rmsnorm_output_shape():
    B, T, C = 2, 5, 16
    x = torch.randn(B, T, C)

    norm = RMSNorm(emb_dim=C)
    out = norm(x)

    assert out.shape == x.shape


def test_rmsnorm_no_mean_centering():
    # Core distinction from LayerNorm: RMSNorm does NOT subtract mean.
    # So a constant-shifted input should NOT produce the same output
    # as the original (unlike LayerNorm which would fully absorb shift).
    torch.manual_seed(0)
    C = 8
    norm = RMSNorm(emb_dim=C)

    x = torch.randn(1, 1, C)
    x_shifted = x + 5.0  # add constant offset

    out = norm(x)
    out_shifted = norm(x_shifted)

    assert not torch.allclose(out, out_shifted, atol=1e-3)


def test_rmsnorm_scale_invariance_direction():
    # RMSNorm's core property: normalizing by RMS should make output
    # roughly independent of input's overall magnitude scale
    # (before applying learnable weight, which starts at 1.0).
    torch.manual_seed(1)
    C = 8
    norm = RMSNorm(emb_dim=C)

    x = torch.randn(1, 3, C)
    x_scaled = x * 10.0

    out = norm(x)
    out_scaled = norm(x_scaled)

    # with weight initialized to 1.0, outputs should be very close
    # regardless of input scale
    torch.testing.assert_close(out, out_scaled, rtol=1e-3, atol=1e-3)


def test_rmsnorm_preserves_dtype():
    # Verify the float32 upcast-then-downcast logic works correctly
    C = 8
    norm = RMSNorm(emb_dim=C)
    x = torch.randn(1, 2, C, dtype=torch.float16)

    out = norm(x)

    assert out.dtype == torch.float32


# DynamicTanh (DyT)

def test_dyt_output_shape():
    B, T, C = 2, 5, 16
    x = torch.randn(B, T, C)

    dyt = DynamicTanh(emb_dim=C)
    out = dyt(x)

    assert out.shape == x.shape


def test_dyt_output_bounded():
    # tanh output must always be in (-1, 1), regardless of input magnitude
    C = 8
    dyt = DynamicTanh(emb_dim=C)

    x = torch.randn(1, 5, C) * 1000  # extreme values
    out = dyt(x)

    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


def test_dyt_zero_input_gives_zero_output():
    C = 8
    dyt = DynamicTanh(emb_dim=C)
    x = torch.zeros(1, 3, C)

    out = dyt(x)

    torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-6, atol=1e-6)


# Dynamic Erf (Derf)

def test_dynamic_erf_output_shape():
    B, T, C = 2, 5, 16
    x = torch.randn(B, T, C)

    derf = DynamicErf(embedding_dim=C)
    out = derf(x)

    assert out.shape == x.shape


def test_dynamic_erf_output_bounded():
    # erf output must always be in (-1, 1), same bound as tanh
    C = 8
    derf = DynamicErf(embedding_dim=C)

    x = torch.randn(1, 5, C) * 1000
    out = derf(x)

    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


def test_dynamic_erf_shift_breaks_symmetry():
    # With shift != 0, erf(alpha*x + shift) should NOT be odd-symmetric
    # around zero anymore (unlike default DyT which has no shift param).
    torch.manual_seed(0)
    C = 8
    derf = DynamicErf(embedding_dim=C)

    with torch.no_grad():
        derf.shift.fill_(2.0)  # force non-zero shift

    x = torch.randn(1, 1, C)
    out_pos = derf(x)
    out_neg = derf(-x)

    # if symmetric, out_neg should equal -out_pos; with shift, it should NOT
    assert not torch.allclose(out_neg, -out_pos, atol=1e-3)