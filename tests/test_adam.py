'''
Testing Adam function whether running correctly or not
'''

import numpy as np
import pytest

from src.core.optimizer.optimizer import Adam

def test_adam_first_step_manual():
    opt = Adam(np.array([1.0]), lr=0.1, beta_1=0.9, beta_2=0.999, eps=1e-8)

    updated = opt.update(np.array([2.0]))
    
    # step 1:
    # m = 0.1 * 2.0 = 0.2
    # v = 0.001 * 4 = 0.004
    # m_hat = 0.2 / (1 - 0.9) = 2
    # v_hat = 0.004 / (1 - 0.999) = 4
    # update = 0.1 * 2 / sqrt(4) = 0.1
    expected = np.array([0.9])
    
    np.testing.assert_allclose(updated, expected, rtol=1e-6)

def test_adam_state_updates():
    opt = Adam(np.array([1.0, 2.0]), lr=0.1)

    opt.update(np.array([0.5, -0.5]))

    assert opt.t == 1
    np.testing.assert_allclose(opt.momentum, np.array([0.05, -0.05]))
    np.testing.assert_allclose(opt.acm_grad, np.array([0.00025, 0.00025]))

def test_adam_rejects_wrong_grad_shape():
    opt = Adam(np.array([1.0, 2.0]), lr=0.1)

    with pytest.raises(ValueError):
        opt.update(np.array([[0.1, 0.2]]))


def test_adam_minimizes_simple_quadratic():
    opt = Adam(np.array([5.0]), lr=0.1)

    for _ in range(100):
        grad = 2 * opt.data
        opt.update(grad)

    assert abs(opt.data[0]) < 5.0