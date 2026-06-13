'''
Testing SGD optimizer whether running correctly and passes all test
'''

import numpy as np
import pytest

from src.core.optimizer.optimizer import SGD

def test_sgd_single_step_vector():
    data = np.array([1.0, 2.0, 3.0])
    grad = np.array([0.1, 0.2, 0.3])
    
    opt = SGD(data=data, lr=0.1)
    updated = opt.update(grad)
    
    expected = np.array([0.99, 1.98, 2.97])
    np.testing.assert_allclose(updated, expected)
    
def test_sgd_updates_internal_data():
    data = np.array([1.0, 2.0])
    grad = np.array([0.5, -0.5])
    
    opt = SGD(data, lr=0.2)
    opt.update(grad)
    
    expected = np.array([0.9, 2.1])
    np.testing.assert_allclose(opt.data, expected)

def test_sgd_multiple_steps():
    data = np.array([1.0])
    grad = np.array([0.5])
    
    opt = SGD(data, lr=0.1)
    opt.update(grad)
    opt.update(grad)
    
    expected = np.array([0.9])
    np.testing.assert_allclose(opt.data, expected)
    
def test_sgd_accepts_integer_data_but_converts_to_float():
    data = np.array([1, 2, 3])
    grad = np.array([0.1, 0.1, 0.1])

    opt = SGD(data, lr=0.1)
    updated = opt.update(grad)

    expected = np.array([0.99, 1.99, 2.99])
    np.testing.assert_allclose(updated, expected)
    assert opt.data.dtype == np.float64

def test_sgd_rejects_invalid_lr_type():
    with pytest.raises(TypeError):
        SGD(np.array([1.0]), lr="0.1") #type: ignore


def test_sgd_rejects_non_positive_lr():
    with pytest.raises(ValueError):
        SGD(np.array([1.0]), lr=0.0)

    with pytest.raises(ValueError):
        SGD(np.array([1.0]), lr=-0.1)

def test_sgd_rejects_wrong_grad_shape():
    opt = SGD(np.array([1.0, 2.0]), lr=0.1)

    with pytest.raises(AssertionError):
        opt.update(np.array([[0.1, 0.2]]))

def test_sgd_minimizes_simple_quadratic():
    x = np.array([5.0])
    opt = SGD(x, lr=0.1)
    
    for _ in range(20):
        grad = 2 * opt.data
        opt.update(grad)
    
    assert abs(opt.data[0]) < 5.0