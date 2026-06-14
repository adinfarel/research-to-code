'''
Testing causal mask whether running correctly
'''

import numpy as np

from src.transformers.attention.causal_mask import scaled_dot_product, _causal_mask, _matrix_multiplication, _softmax

def test_attention_causal_mask_attention():
    Q = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    K = Q.copy()
    
    V = np.array([
        [10., 10.],
        [20., 20.],
        [30., 30.]
    ])
    
    # mask = _causal_mask(_matrix_multiplication(Q, K.T))
    out_masked = scaled_dot_product(Q, K, V, causal_mask=True)
    
    # print(mask)
    # print(_softmax(mask))
    # print(out_masked)
    
    np.testing.assert_allclose(out_masked[0], V[0], atol=1e-6)

def test_attention_causal_mask_future_independence():
    Q = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    K_version_1 = np.array([
        [1.0, 0.0],
        [1.0, 0.0]  
    ])
    K_version_2 = np.array([
        [1.0, 0.0],
        [0.0, 1.0]  
    ])
    V = np.array([
        [10., 20.],
        [30., 40.]
    ])
    
    out_1 = scaled_dot_product(Q, K_version_1, V, causal_mask=True)
    out_2 = scaled_dot_product(Q, K_version_2, V, causal_mask=True)
    
    np.testing.assert_allclose(out_1[0], out_2[0], atol=1e-6)
    
    assert not np.allclose(out_1[1], out_2[1], atol=1e-6)
    
def test_attention_causal_mask_toggle():
    Q = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    K = Q.copy()
    V = np.array([
        [10., 20.],
        [30., 40.]
    ])
    
    out_no_mask = scaled_dot_product(Q, K, V, causal_mask=False)
    
    assert not np.allclose(out_no_mask[0], V[0])