'''
Testing GLUFamily (GeGLU, SwiGLU, ReGLU variants)
'''

import torch

from src.transformers.feed_forward.glu_family import GLUFamily


def test_glu_output_shape():
    B, T, C = 2, 5, 16
    x = torch.randn(B, T, C)

    glu = GLUFamily(emb_dim=C, func_act="silu")
    out = glu(x)

    assert out.shape == (B, T, C)


def test_glu_param_count_matches_4x_ffn_approx():
    # Core design claim in the docstring: hid_dim = emb_dim * 8/3 should
    # give GLU roughly the same total param count as a standard FFN
    # with 4x expansion (2 matrices), since GLU uses 3 matrices instead.
    C = 512
    glu = GLUFamily(emb_dim=C, func_act="silu")

    glu_params = sum(p.numel() for p in glu.parameters())
    standard_ffn_params = 2 * (C * 4 * C)  # up_proj + down_proj, 4x expansion

    # should be close (within ~5%), not exact due to rounding hid_dim to int
    ratio = glu_params / standard_ffn_params
    assert 0.95 <= ratio <= 1.05


def test_glu_invalid_activation_raises():
    glu = GLUFamily(emb_dim=16, func_act="not_a_real_activation")
    x = torch.randn(1, 3, 16)

    try:
        glu(x)
        assert False, "Expected ValueError for unknown activation"
    except ValueError:
        pass


def test_glu_gating_zeroes_out_when_gate_saturated_negative():
    # Sanity check on the gating mechanism itself: if we manually force
    # the gate activation to near-zero (simulating a "closed gate"),
    # the corresponding value features should be suppressed in output.
    torch.manual_seed(0)
    C = 8
    glu = GLUFamily(emb_dim=C, func_act="relu")  # relu is easiest to force to exact 0

    # force gated linear layer weights/bias so output is very negative
    # (relu will zero it out completely)
    with torch.no_grad():
        glu.gated.weight.zero_()

    x = torch.randn(1, 3, C)
    value_out = glu.value(x)
    gated_out = torch.relu(glu.gated(x))

    assert torch.all(gated_out == 0)  # gate fully closed
    hidden = value_out * gated_out
    assert torch.all(hidden == 0)  # value contribution fully suppressed


def test_glu_different_activations_produce_different_outputs():
    torch.manual_seed(42)
    C = 16
    x = torch.randn(1, 4, C)

    torch.manual_seed(1)
    glu_silu = GLUFamily(emb_dim=C, func_act="silu")
    torch.manual_seed(1)
    glu_gelu = GLUFamily(emb_dim=C, func_act="gelu")

    out_silu = glu_silu(x)
    out_gelu = glu_gelu(x)

    # same weights (same seed), different activation -> different output
    assert not torch.allclose(out_silu, out_gelu)