'''
testing whether function running correctly
'''

import numpy as np

from src.core.backward.linear_backward import (
    linear_backward,
    linear_forward,
    relu_backward,
    relu_forward,
)

def test_linear_forward_shape():
    x = np.random.randn(4, 3)
    w = np.random.randn(3, 2)
    b = np.random.randn(2)
    
    z, _ = linear_forward(x, w, b)
    
    assert z.shape == (4, 2)

def test_linear_backward_shape():
    x = np.random.randn(4, 3)
    w = np.random.randn(3, 2)
    b = np.random.randn(2)
    
    z, cache = linear_forward(x, w, b)
    dout = np.random.randn(*z.shape) # random upstream grad
    
    dx, dw, db = linear_backward(dout, cache)
    
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert db.shape == b.shape

def test_linear_backward_manual_small():
    x = np.array(
        [[1.0, 2.0]]
    )
    
    w = np.array(
        [
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    
    b = np.array([0.0, 0.0])
    
    _, cache = linear_forward(x, w, b)
    
    dout = np.array([[10.0, 20.0]])
    
    dx, dw, db = linear_backward(dout, cache)
    
    expected_dx = np.array([[110.0, 170.0]])
    
    expected_dw = np.array(
        [
            [10.0, 20.0],
            [20.0, 40.0]
        ]
    )
    
    expected_db = np.array([10.0, 20.0])
    
    np.testing.assert_allclose(dx, expected_dx)
    np.testing.assert_allclose(dw, expected_dw)
    np.testing.assert_allclose(db, expected_db)

def test_relu_forward():
    z = np.array(
        [
            [-1.0, 0.0, 2.0],
            [3.0, -4.0, 5.0],
        ]
    )
    
    out, _ = relu_forward(z)
    
    expected = np.array(
        [
            [0.0, 0.0, 2.0],
            [3.0, 0.0, 5.0]
        ]
    )
    
    np.testing.assert_allclose(out, expected)

def test_relu_backward():
    z = np.array(
        [
            [-1.0, 0.0, 2.0],
            [3.0, -4.0, 5.0],
        ]
    )
    
    _, cache = relu_forward(z)
    
    dout = np.ones_like(z)
    dz = relu_backward(dout, cache)
    
    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    
    np.testing.assert_allclose(dz, expected)

def test_linear_relu_composed_backward():
    x = np.array([[2.0, -1.0]])
    w = np.array(
        [
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    b = np.array([10.0, -1.0])
    
    z, linear_cache = linear_forward(x, w, b)
    out, relu_cache = relu_forward(z)
    
    dout = np.array([[2.0, 3.0]])
    
    dz = relu_backward(dout, relu_cache)
    dx, dw, db = linear_backward(dz, linear_cache)
    
    # Forward:
    # z[0] = 2*3 + (-1)*5 + 10 = 11 > 0
    # z[1] = 2*4 + (-1)*6 + -1 = 1 > 0
    
    # ReLU passes both gradients, so dz = dout [[2, 3]]
    expected_dz = np.array([[2.0, 3.0]])
    expected_dx = expected_dz @ w.T
    expected_dw = x.T @ expected_dz
    expected_db = np.sum(expected_dz, axis=0)
    
    np.testing.assert_allclose(dz, expected_dz)
    np.testing.assert_allclose(dx, expected_dx)
    np.testing.assert_allclose(dw, expected_dw)
    np.testing.assert_allclose(db, expected_db)

def test_linear_backward_numerical_gradient_w():
    rng = np.random.default_rng(42) # for reproducebility
    
    x = rng.random(size=(3, 4))
    w = rng.random(size=(4, 2))
    b = rng.random(size=(2, ))
    
    z, cache = linear_forward(x, w, b)
    
    dout = rng.random(size=(z.shape))
    
    _, dw, _ = linear_backward(dout, cache)
    
    h = 1e-6
    numerical_dw = np.zeros_like(w)
    
    def loss_fn(w_cand: np.ndarray):
        z_cand, _ = linear_forward(x, w_cand, b)
        
        # if L = sum(z * dout), then dL/dz = dout.
        return float(np.sum(z_cand * dout))
    
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            old_value = w[i, j]
            
            w[i, j] = old_value + h
            loss_plus = loss_fn(w)
            
            w[i, j] = old_value - h
            loss_minus = loss_fn(w)
            
            w[i, j] = old_value
            
            numerical_dw[i, j] = (loss_plus - loss_minus) / (2*h)
    
    np.testing.assert_allclose(dw, numerical_dw, rtol=1e-5, atol=1e-5)

def test_linear_backward_numerical_gradient_x():
    rng = np.random.default_rng(123)

    x = rng.normal(size=(3, 4))
    w = rng.normal(size=(4, 2))
    b = rng.normal(size=(2,))

    z, cache = linear_forward(x, w, b)

    dout = rng.normal(size=z.shape)

    dx, _, _ = linear_backward(dout, cache)

    h = 1e-6
    numerical_dx = np.zeros_like(x)

    def loss_fn(x_candidate: np.ndarray) -> float:
        z_candidate, _ = linear_forward(x_candidate, w, b)
        return float(np.sum(z_candidate * dout))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            old_value = x[i, j]

            x[i, j] = old_value + h
            loss_plus = loss_fn(x)

            x[i, j] = old_value - h
            loss_minus = loss_fn(x)

            x[i, j] = old_value

            numerical_dx[i, j] = (loss_plus - loss_minus) / (2 * h)

    np.testing.assert_allclose(dx, numerical_dx, rtol=1e-5, atol=1e-5)
    
def test_linear_backward_numerical_gradient_b():
    rng = np.random.default_rng(456)

    x = rng.normal(size=(3, 4))
    w = rng.normal(size=(4, 2))
    b = rng.normal(size=(2,))

    z, cache = linear_forward(x, w, b)

    dout = rng.normal(size=z.shape)

    _, _, db = linear_backward(dout, cache)

    h = 1e-6
    numerical_db = np.zeros_like(b)

    def loss_fn(b_candidate: np.ndarray) -> float:
        z_candidate, _ = linear_forward(x, w, b_candidate)
        return float(np.sum(z_candidate * dout))

    for i in range(b.shape[0]):
        old_value = b[i]

        b[i] = old_value + h
        loss_plus = loss_fn(b)

        b[i] = old_value - h
        loss_minus = loss_fn(b)

        b[i] = old_value

        numerical_db[i] = (loss_plus - loss_minus) / (2 * h)

    np.testing.assert_allclose(db, numerical_db, rtol=1e-5, atol=1e-5)