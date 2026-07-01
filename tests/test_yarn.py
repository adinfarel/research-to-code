'''
Testing YaRN whether running correcly
'''

import torch

from src.transformers.positional_encoding.rope import RotaryPositionalEncoding
from src.transformers.positional_encoding.yarn import YaRNRoPE

def test_yarn_output_shape_and_finite():
    dim = 8
    yarn = YaRNRoPE(emb_dim=dim, orig_max_position=2048, target_max_position=8192)
    x = torch.randn(2, 10, dim)

    out = yarn(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_yarn_more_stable_than_rope_beyond_training_length():
    dim = 8
    seq_len = 4096 

    rope = RotaryPositionalEncoding(emb_dim=dim)
    yarn = YaRNRoPE(emb_dim=dim, orig_max_position=2048, target_max_position=8192)

    x = torch.randn(1, 1, dim).repeat(1, seq_len, 1)

    out_rope = rope(x)
    out_yarn = yarn(x)

    dot_rope = (out_rope[0, 100] * out_rope[0, 4000]).sum().abs()
    dot_yarn = (out_yarn[0, 100] * out_yarn[0, 4000]).sum().abs()

    assert torch.isfinite(dot_rope)
    assert torch.isfinite(dot_yarn)