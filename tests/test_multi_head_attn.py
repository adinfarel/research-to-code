'''
Testing multi-head attention implementation whether running correctly or not
'''

import numpy as np
import pytest

from src.transformers.attention.multi_head_attn import MultiHeadAttn

def test_mha_output_shape():
    B, T, C = 2, 5, 8
    n_head  = 2
    
    mha = MultiHeadAttn(embed_dim=C, n_head=n_head)
    X = np.random.randn(B, T, C)
    
    out = mha(X, causal_mask=False)
    
    assert out.shape == (B, T, C)

def test_mha_causal_mask_leakage():
    B, T, C = 1, 3, 4
    n_head  = 2
    
    mha = MultiHeadAttn(embed_dim=C, n_head=n_head)
    
    X1 = np.array([[[1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.5, 0.5, 0.5, 0.5]]]) 
                    
    X2 = np.array([[[1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [99., 99., 99., 99.]]])
    
    out_1 = mha(X1, causal_mask=True)
    out_2 = mha(X2, causal_mask=True)
    
    np.testing.assert_allclose(out_1[:, 0, :], out_2[:, 0, :], atol=1e-6)
    np.testing.assert_allclose(out_1[:, 1, :], out_2[:, 1, :], atol=1e-6)
    
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(out_1[:, 2, :], out_2[:, 2, :], atol=1e-6)

def test_mha_softmax_distribution():
    B, T, C = 2, 4, 6
    mha = MultiHeadAttn(embed_dim=C, n_head=2)
    
    dummy_affinity = np.random.rand(B, mha.n_head, T, T)
    weights = mha._softmax(dummy_affinity, axis=-1)
    
    expected_sum = np.ones((B, mha.n_head, T))
    np.testing.assert_allclose(np.sum(weights, axis=-1), expected_sum, atol=1e-6)