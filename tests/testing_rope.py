'''
Testing rope whether running correctly
'''

import torch
import math

from src.transformers.positional_encoding.rope import RotaryPositionalEncoding

def test_rope_output_shape():
    rope = RotaryPositionalEncoding(emb_dim=8)
    x = torch.randn(2, 5, 8)  # (batch, seq_len, dim)

    out = rope(x)

    assert out.shape == x.shape


def test_rope_norm_preserved():
    # Rotation should NOT change vector magnitude (key property
    # of RoPE -- it's an orthogonal transform per pair).
    rope = RotaryPositionalEncoding(emb_dim=8)
    x = torch.randn(1, 4, 8)

    out = rope(x)

    norm_before = x.norm(dim=-1)
    norm_after = out.norm(dim=-1)

    torch.testing.assert_close(norm_before, norm_after, rtol=1e-5, atol=1e-5)


def test_rope_relative_position_invariance():
    # Core property: dot product between two rotated vectors should
    # depend only on relative distance (m - n), not absolute position.
    rope = RotaryPositionalEncoding(emb_dim=8)

    x = torch.randn(1, 1, 8)  # single token vector, reused at different positions
    seq_a = x.repeat(1, 10, 1)  # token placed at position 0..9
    seq_b = x.repeat(1, 10, 1)  # same token, will be shifted

    out_a = rope(seq_a)
    out_b = rope(seq_b)

    # dot product between position 2 and 5 (distance=3)
    dot_1 = (out_a[0, 2] * out_a[0, 5]).sum()
    # dot product between position 4 and 7 (distance=3, shifted by 2)
    dot_2 = (out_b[0, 4] * out_b[0, 7]).sum()

    torch.testing.assert_close(dot_1, dot_2, rtol=1e-4, atol=1e-4)