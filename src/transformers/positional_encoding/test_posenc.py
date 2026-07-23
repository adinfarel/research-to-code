import math

import pytest
import torch

from src.transformers.positional_encoding.pos_enc import (
    PositionalEncoding,
    ROPEBase,
    ROPE,
    NTKAwareROPE,
    YaRNROPE,
)


class TestPositionalEncoding:
    def test_output_shape_matches_input(self):
        pe = PositionalEncoding(emb_dim=8, max_position=16)
        x = torch.randn(2, 5, 8)
        assert pe(x).shape == x.shape

    def test_matches_manual_formula(self):
        pe = PositionalEncoding(emb_dim=4, max_position=10, base=10000.0)
        x = torch.zeros(1, 3, 4)
        out = pe(x)

        position = torch.arange(3, dtype=torch.float32).unsqueeze(1)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4))
        angles = position * inv_freq
        expected = torch.zeros(3, 4)
        expected[:, 0::2] = torch.sin(angles)
        expected[:, 1::2] = torch.cos(angles)

        assert torch.allclose(out[0], expected, atol=1e-6)

    def test_seq_len_exceeding_max_position_raises(self):
        pe = PositionalEncoding(emb_dim=8, max_position=4)
        x = torch.randn(1, 10, 8)
        with pytest.raises(AssertionError):
            pe(x)

    def test_odd_emb_dim_raises(self):
        with pytest.raises(AssertionError):
            PositionalEncoding(emb_dim=7)

    def test_is_additive_not_rotational(self):
        pe = PositionalEncoding(emb_dim=4, max_position=8)
        x = torch.zeros(1, 3, 4)
        assert not torch.allclose(pe(x), x)

class TestRotaryBase:
    def test_base_class_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            ROPEBase(emb_dim=8)  # type: ignore

    def test_odd_emb_dim_raises(self):
        with pytest.raises(AssertionError):
            ROPE(emb_dim=7)

    def test_inv_freq_shape(self):
        rope = ROPE(emb_dim=8)
        assert rope.inv_freq.shape == (4,)

    def test_default_mscale_is_one(self):
        rope = ROPE(emb_dim=8)
        assert rope.mscale == 1.0

@pytest.mark.parametrize("cls,kwargs", [
    (ROPE, {}),
    (NTKAwareROPE, {}),
    (YaRNROPE, {}),
])
class TestSharedRotaryProperties:
    def test_output_shape_matches_input(self, cls, kwargs):
        rope = cls(emb_dim=8, **kwargs)
        x = torch.randn(2, 5, 8)
        assert rope(x).shape == x.shape

    def test_zero_vector_stays_zero(self, cls, kwargs):
        rope = cls(emb_dim=8, **kwargs)
        x = torch.zeros(1, 5, 8)
        assert torch.allclose(rope(x), torch.zeros_like(x))

    def test_position_zero_is_identity_up_to_mscale(self, cls, kwargs):
        rope = cls(emb_dim=8, **kwargs)
        x = torch.randn(1, 1, 8)
        out = rope(x)
        assert torch.allclose(out, x * rope.mscale, atol=1e-5)

    def test_norm_scales_by_mscale_per_pair(self, cls, kwargs):
        rope = cls(emb_dim=8, **kwargs)
        x = torch.randn(2, 6, 8)
        out = rope(x)
        ratio = out.norm(dim=-1) / x.norm(dim=-1)
        assert torch.allclose(ratio, torch.full_like(ratio, rope.mscale), atol=1e-4)


class TestInterleavedVsHalfRotate:
    def test_matches_manual_interleaved_formula(self):
        torch.manual_seed(0)
        dim, T = 4, 3
        rope = ROPE(emb_dim=dim, base=10000.0)
        x = torch.randn(1, T, dim)
        out = rope(x)

        inv_freq = rope.inv_freq  # (dim/2,)
        expected = torch.zeros_like(x)
        for t in range(T):
            for i in range(dim // 2):
                theta_i = t * inv_freq[i] # type: ignore
                x0, x1 = x[0, t, 2 * i], x[0, t, 2 * i + 1]
                expected[0, t, 2 * i] = x0 * math.cos(theta_i) - x1 * math.sin(theta_i)
                expected[0, t, 2 * i + 1] = x1 * math.cos(theta_i) + x0 * math.sin(theta_i)

        assert torch.allclose(out, expected, atol=1e-5)

    def test_rotate_interleaved_helper_pattern(self):
        rope = ROPE(emb_dim=6)
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])  # (1,1,6)
        rotated = rope._rotate_interleaved(x)
        # (x0,x1,x2,x3,x4,x5) -> (-x1,x0,-x3,x2,-x5,x4)
        expected = torch.tensor([[[-2.0, 1.0, -4.0, 3.0, -6.0, 5.0]]])
        assert torch.equal(rotated, expected)

    def test_differs_from_half_rotate_scheme(self):
        torch.manual_seed(0)
        dim, T = 8, 4
        rope = ROPE(emb_dim=dim)
        x = torch.randn(1, T, dim)
        out_interleaved = rope(x)

        l = torch.arange(T, dtype=torch.float32)
        theta = torch.einsum('i,j->ij', l, rope.inv_freq)
        hat_theta = torch.cat([theta, theta], dim=-1)
        sin, cos = torch.sin(hat_theta), torch.cos(hat_theta)
        xu, xd = x[..., :dim // 2], x[..., dim // 2:]
        hatx = torch.cat([-xd, xu], dim=-1)
        out_half_rotate = x * cos + hatx * sin

        assert not torch.allclose(out_interleaved, out_half_rotate, atol=1e-4)

class TestNTKAwareRoPE:
    def test_reduces_to_vanilla_when_scale_is_one(self):
        rope = ROPE(emb_dim=8, base=10000.0)
        ntk = NTKAwareROPE(emb_dim=8, base=10000.0, orig_max_pos=2048, target_max_pos=2048)
        assert torch.allclose(rope.inv_freq, ntk.inv_freq, atol=1e-6) # type: ignore

    def test_larger_target_shrinks_base_frequency_growth(self):
        ntk_small_scale = NTKAwareROPE(emb_dim=8, orig_max_pos=2048, target_max_pos=4096)
        ntk_large_scale = NTKAwareROPE(emb_dim=8, orig_max_pos=2048, target_max_pos=16384)
        assert torch.all(ntk_large_scale.inv_freq <= ntk_small_scale.inv_freq + 1e-8) # type: ignore
 
    def test_repr_contains_key_params(self):
        ntk = NTKAwareROPE(emb_dim=8)
        assert "NTKAwareROPE" in repr(ntk)
        assert "target_max_pos" in repr(ntk)

class TestYaRNRoPE:
    def test_mscale_greater_than_one_when_scaling_up(self):
        yarn = YaRNROPE(emb_dim=32, orig_max_position=2048, target_max_position=8192)
        assert yarn.mscale > 1.0

    def test_mscale_is_one_when_no_scaling(self):
        yarn = YaRNROPE(emb_dim=32, orig_max_position=2048, target_max_position=2048)
        assert yarn.mscale == pytest.approx(1.0)

    def test_mscale_all_dim_overrides_mscale(self):
        yarn_default = YaRNROPE(emb_dim=32, orig_max_position=2048, target_max_position=8192, mscale=1.0)
        yarn_all_dim = YaRNROPE(emb_dim=32, orig_max_position=2048, target_max_position=8192, mscale_all_dim=2.0)
        assert yarn_default.mscale != yarn_all_dim.mscale

    def test_inv_freq_interpolates_between_extrapolation_and_interpolation(self):
        yarn = YaRNROPE(emb_dim=32, orig_max_position=2048, target_max_position=8192)
        pos_freqs = yarn.base ** (torch.arange(0, yarn.dim, 2, dtype=torch.float32) / yarn.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (yarn.scale_factor * pos_freqs)

        lower_bound = torch.minimum(inv_freq_extrapolation, inv_freq_interpolation) - 1e-6
        upper_bound = torch.maximum(inv_freq_extrapolation, inv_freq_interpolation) + 1e-6
        assert torch.all(yarn.inv_freq >= lower_bound) # type: ignore
        assert torch.all(yarn.inv_freq <= upper_bound) # type: ignore

    def test_correction_dim_uses_configured_base(self):
        yarn_base_10k = YaRNROPE(emb_dim=32, base=10000.0)
        yarn_base_500k = YaRNROPE(emb_dim=32, base=500000.0)
        low_10k = yarn_base_10k._find_correction_dim(32)
        low_500k = yarn_base_500k._find_correction_dim(32)
        assert low_10k != pytest.approx(low_500k)


class TestGradientFlowSanity:
    @pytest.mark.parametrize("cls,kwargs", [
        (ROPE, {}),
        (NTKAwareROPE, {}),
        (YaRNROPE, {}),
        (PositionalEncoding, {}),
    ])
    def test_backward_runs_without_error(self, cls, kwargs):
        module = cls(emb_dim=8, **kwargs)
        x = torch.randn(1, 4, 8, requires_grad=True)
        out = module(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
