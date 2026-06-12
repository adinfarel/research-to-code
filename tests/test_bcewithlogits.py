'''
Testing BCEWithLogits whether running correctly
'''

import numpy as np

from src.core.criterion.binaryentropy import BCEWithLogits

def test_bce_confident_correct_low_loss():
    loss_fn = BCEWithLogits()
    
    logits = np.array([10.0, -10.0])
    labels = np.array([1.0, 0.0])
    
    loss = loss_fn(logits, labels)
    
    assert loss < 1e-3
    
def test_bce_confident_wrong_high_loss():
    loss_fn = BCEWithLogits()

    logits = np.array([-10.0, 10.0])
    labels = np.array([1.0, 0.0])

    loss = loss_fn(logits, labels)

    assert loss > 9.0
    
def test_bce_backward_shape():
    loss_fn = BCEWithLogits()

    logits = np.array([[1.0, -1.0], [0.5, -0.5]])
    labels = np.array([[1.0, 0.0], [1.0, 0.0]])

    _ = loss_fn(logits, labels)
    grad = loss_fn.backward()

    assert grad.shape == logits.shape

def test_bce_backward_manual():
    loss_fn = BCEWithLogits()
    
    logits = np.array([0.0])
    labels = np.array([1.0])
    
    _ = loss_fn(logits, labels)
    grad = loss_fn.backward()
    
    # sigmoid(0) = 0.5
    # grad = sigmoid(logit) - label = 0.5 - 1 = -0.5
    expected = np.array([-0.5])
    
    np.testing.assert_allclose(grad, expected)
    
def test_bce_backward_multielement_mean_scaling():
    loss_fn = BCEWithLogits()

    logits = np.array([0.0, 0.0])
    labels = np.array([1.0, 0.0])

    _ = loss_fn(logits, labels)
    grad = loss_fn.backward()

    # raw grad = [0.5 - 1, 0.5 - 0] = [-0.5, 0.5]
    # mean over 2 elements -> [-0.25, 0.25]
    expected = np.array([-0.25, 0.25])

    np.testing.assert_allclose(grad, expected)

def test_bce_numerical_gradient():
    loss_fn = BCEWithLogits()

    logits = np.array([0.2, -1.0, 2.0], dtype=np.float64)
    labels = np.array([1.0, 0.0, 1.0], dtype=np.float64)

    _ = loss_fn(logits, labels)
    grad = loss_fn.backward()

    h = 1e-6
    numerical_grad = np.zeros_like(logits)

    for i in range(logits.shape[0]):
        old = logits[i]

        logits[i] = old + h
        loss_plus = loss_fn(logits, labels)

        logits[i] = old - h
        loss_minus = loss_fn(logits, labels)

        logits[i] = old

        numerical_grad[i] = (loss_plus - loss_minus) / (2 * h)

    np.testing.assert_allclose(grad, numerical_grad, rtol=1e-5, atol=1e-5)