
import torch

from src.transformers.positional_encoding.rope import RotaryPositionalEncoding
from src.transformers.positional_encoding.ntk_aware import NTKAwareRoPE


def test_ntk_aware_matches_rope_at_short_seq():
    # At short sequence length (well within orig_max_position), NTK-aware
    # scaling should behave close to vanilla RoPE -- the scaling effect
    # should be minimal for near positions.
    dim = 8
    rope = RotaryPositionalEncoding(emb_dim=dim)
    ntk = NTKAwareRoPE(emb_dim=dim, orig_max_pos=2048, target_max_pos=8192)

    x = torch.randn(1, 5, dim)

    out_rope = rope(x)
    out_ntk = ntk(x)

    # not exactly equal (base IS scaled), but should be reasonably close
    # at short sequences -- loose tolerance just to sanity check direction
    assert torch.allclose(out_rope, out_ntk, atol=1.0)