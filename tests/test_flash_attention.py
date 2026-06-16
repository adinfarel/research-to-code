'''
Testing flash attention whether running correctly
'''

import numpy as np
import pytest

from src.transformers.optimization.flash_attention import MultiHeadAttn

def test_flash_attn_shape():
    mha = MultiHeadAttn(embed_dim=4, n_head=2)
    T, d = 8, mha.head_dim  # Seq_Len = 8, Head_Dim = 2
    
    Q = np.random.randn(T, d)
    K = np.random.randn(T, d)
    V = np.random.randn(T, d)

    out = mha._flash_attn_2d(Q, K, V, causal_mask=False, B_r=2, B_c=2, verbose=False)
    
    assert out.shape == (T, d)

def test_flash_attn_matches_naive():
    np.random.seed(42)
    mha = MultiHeadAttn(embed_dim=8, n_head=2)
    T, d = 6, mha.head_dim
    
    Q = np.random.randn(T, d)
    K = np.random.randn(T, d)
    V = np.random.randn(T, d)

    out_flash = mha._flash_attn_2d(Q, K, V, causal_mask=True, B_r=2, B_c=2, verbose=False)
    
    scale = 1.0 / np.sqrt(d)
    scores= (Q @ K.T) * scale
    
    masking = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(masking, float('-inf'), scores)
    
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    out_naive = weights @ V
    
    np.testing.assert_allclose(
        out_flash,
        out_naive,
        atol=1e-6,
        rtol=1e-6
    )
    
def test_flash_attn_causal_mask_leakage():
    mha = MultiHeadAttn(embed_dim=4, n_head=2)
    T, d = 4, mha.head_dim
    
    Q = np.random.randn(T, d)
    K = np.random.randn(T, d)
    
    V1 = np.array([
        [10., 10.],
        [20., 20.],
        [30., 30.],
        [40., 40.]  
    ])
    
    V2 = np.array([
        [10., 10.],
        [20., 20.],
        [30., 30.],
        [99., 99.]  
    ])
    
    out_1 = mha._flash_attn_2d(Q, K, V1, causal_mask=True, B_r=2, B_c=2, verbose=False)
    out_2 = mha._flash_attn_2d(Q, K, V2, causal_mask=True, B_r=2, B_c=2, verbose=False)
    
    np.testing.assert_allclose(out_1[0], out_2[0], atol=1e-6)
    np.testing.assert_allclose(out_1[1], out_2[1], atol=1e-6)
    np.testing.assert_allclose(out_1[2], out_2[2], atol=1e-6)
    
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(out_1[3], out_2[3], atol=1e-6)
    
def test_flash_attn_block_size_invariance():
    mha = MultiHeadAttn(embed_dim=4, n_head=2)
    T, d = 4, mha.head_dim
    
    Q = np.random.randn(T, d)
    K = np.random.randn(T, d)
    V = np.random.randn(T, d)
    
    out_tile_small = mha._flash_attn_2d(Q, K, V, causal_mask=True, B_r=2, B_c=2, verbose=False)
    
    out_tile_large = mha._flash_attn_2d(Q, K, V, causal_mask=True, B_r=4, B_c=2, verbose=False)
    
    np.testing.assert_allclose(
        out_tile_small,
        out_tile_large,
        atol=1e-6,
    )