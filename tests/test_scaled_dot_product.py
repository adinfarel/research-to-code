'''
Testing scaled dot-product attention whether running correctly?
'''

import numpy as np

from src.transformers.attention.scaled_dot_product import scaled_dot_product, _matrix_multiplication, _softmax

def test_attention_output_shape():
    Q = np.random.randn(4, 8)
    K = np.random.randn(4, 8)
    V = np.random.randn(4, 8)
    
    out = scaled_dot_product(Q=Q, K=K, V=V) #type: ignore
    
    assert out.shape == (4, 8)

def test_attention_weights_sum_to_one():
    Q = np.random.randn(3, 4)
    K = np.random.randn(3, 4)
    
    scores = _matrix_multiplication(Q, K.T)
    scores /= np.sqrt(K.shape[-1])
    
    weight = _softmax(scores)
    
    expected = np.ones(3)
    
    np.testing.assert_allclose(
        np.sum(weight, axis=-1),
        expected,
        atol=1e-6
    )

def test_manual_tiny_example():
    Q = np.array([
        [1., 0.]
    ])

    K = np.array([
        [1., 0.]
    ])

    V = np.array([
        [10., 20.]
    ])
    
    out = scaled_dot_product(Q, K, V)
    
    expected = np.array([[10., 20.]])
    
    np.testing.assert_allclose(out, expected)

def test_attention_matches_numpy():
    Q = np.random.randn(4, 8)
    K = np.random.randn(4, 8)
    V = np.random.randn(4, 8)

    out = scaled_dot_product(Q, K, V)

    scores = Q @ K.T
    scores /= np.sqrt(K.shape[-1])

    weights = np.exp(
        scores - np.max(scores, axis=-1, keepdims=True)
    )

    weights /= np.sum(
        weights,
        axis=-1,
        keepdims=True
    )

    expected = weights @ V

    np.testing.assert_allclose(
        out,
        expected,
        atol=1e-6,
        rtol=1e-6
    )