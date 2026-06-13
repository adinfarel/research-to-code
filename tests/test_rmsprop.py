'''
Testing RMSProp whether running correctly or not
'''

import numpy as np
import pytest

from src.core.optimizer.optimizer import RMSProp

def test_rmsprop_first_step_manual():
    opt = RMSProp(np.array([1.0]), lr=0.1, beta=0.9, eps=1e-8)
    updated = opt.update(np.array([2.0]))
    
    # v = 0.9 * 0 + 1 - 0.9 * 2^2 = 0.4
    # x = 1 - 0.1 * 2 / sqrt(0.4)
    expected = np.array([1.0 - 0.1 * 2.0 / (np.sqrt(0.4) + 1e-8)])
    
    np.testing.assert_allclose(updated, expected)

def test_rmsprop_accumulator_updates():
    opt = RMSProp(np.array([1.0]), lr=0.1, beta=0.9)

    opt.update(np.array([2.0]))
    first_v = opt.acm_grad.copy()

    opt.update(np.array([2.0]))
    second_v = opt.acm_grad.copy()

    assert second_v[0] > first_v[0]
    
def test_rmsprop_minimizes_simple_quadratic():
    opt = RMSProp(np.array([5.0]), lr=0.1)

    for _ in range(50):
        grad = 2 * opt.data  # f(x) = x^2
        opt.update(grad)

    assert abs(opt.data[0]) < 5.0

def test_rmsprop_rejects_bad_beta():
    with pytest.raises(ValueError):
        RMSProp(np.array([1.0]), lr=0.1, beta=1.0)

    with pytest.raises(ValueError):
        RMSProp(np.array([1.0]), lr=0.1, beta=-0.1)


def test_rmsprop_rejects_bad_eps():
    with pytest.raises(ValueError):
        RMSProp(np.array([1.0]), lr=0.1, eps=0.0)