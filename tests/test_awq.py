'''
Testing AWQ
'''

import numpy as np

from src.serving.compression.quant.awq import AWQuant


def test_awq_output_shape_matches_input():
    np.random.seed(0)
    out_features, in_features = 8, 128  
    W = np.random.randn(out_features, in_features)
    X = np.random.randn(4, 10, in_features)  # (B, T, C_in)

    awq = AWQuant(group_size=128, bits=4)
    W_awq, scale, alpha = awq.fit_and_quantize(W, X)

    assert W_awq.shape == W.shape


def test_awq_selects_best_alpha_from_candidates():
    np.random.seed(1)
    out_features, in_features = 4, 128
    W = np.random.randn(out_features, in_features)
    X = np.random.randn(2, 5, in_features)

    awq = AWQuant(group_size=128, bits=4)
    _, _, best_alpha = awq.fit_and_quantize(W, X)

    assert best_alpha in [0.0, 0.25, 0.5, 0.75, 1.0]


def test_awq_reduces_reconstruction_error_vs_plain_quant_when_activations_skewed():
    np.random.seed(7)
    out_features, in_features = 8, 128
    W = np.random.randn(out_features, in_features)

    X = np.random.randn(4, 10, in_features)
    X[:, :, :10] *= 50.0  

    awq = AWQuant(group_size=128, bits=4)
    W_awq, best_s, best_alpha = awq.fit_and_quantize(W, X)

    Y_true = X @ W.T
    Y_awq = X @ W_awq.T

    W_plain_quant = awq._quantize_weight_groupwise(W, bits=4, group_size=128)
    Y_plain = X @ W_plain_quant.T

    error_awq = np.mean((Y_true - Y_awq) ** 2)
    error_plain = np.mean((Y_true - Y_plain) ** 2)

    assert error_awq <= error_plain