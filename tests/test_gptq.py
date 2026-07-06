'''
Testing GPTQ
'''

import numpy as np

from src.serving.compression.quant.gptq import GPTQ


def test_gptq_output_shape_matches_input():
    np.random.seed(0)
    C = 8
    W = np.random.randn(C, C)
    X = np.random.randn(20, C)  # (N_samples, C_in)

    gptq = GPTQ(bits=4)
    W_quant = gptq.fit_and_quantize(W, X)

    assert W_quant.shape == W.shape


def test_gptq_error_correction_reduces_reconstruction_error():
    # Core claim: GPTQ's error compensation should produce lower output
    # reconstruction error (X @ W_gptq) vs naive column-independent quant
    # (no error propagation) on the SAME calibration data.
    np.random.seed(42)
    C = 16
    W = np.random.randn(C, C)
    X = np.random.randn(50, C)

    gptq = GPTQ(bits=4)
    W_gptq = gptq.fit_and_quantize(W, X)

    # naive quant: just round each column independently, no correction
    qmax = 2 ** (4 - 1) - 1
    W_naive = np.zeros_like(W)
    for i in range(C):
        col = W[:, i]
        scale = np.max(np.abs(col)) / qmax if np.max(np.abs(col)) > 0 else 1.0
        W_naive[:, i] = scale * np.round(col / scale)

    Y_true = X @ W.T
    Y_gptq = X @ W_gptq.T
    Y_naive = X @ W_naive.T

    error_gptq = np.mean((Y_true - Y_gptq) ** 2)
    error_naive = np.mean((Y_true - Y_naive) ** 2)

    assert error_gptq <= error_naive