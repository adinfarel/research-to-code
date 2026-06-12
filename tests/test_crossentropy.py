'''
Testing cross entropy function whether running correctly
'''

import numpy as np

from src.core.criterion.crossentropy import CrossEntropy

def test_softmax_rows_sum_to_one():
    ce = CrossEntropy()
    logits = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
    
    probs = ce._softmax(logits)
    
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(2))

def test_confident_correct_prediction_low_loss():
    ce = CrossEntropy()
    logits = np.array([[10.0, 0.0, 0.0]])
    labels = np.array([0])
    
    loss = ce(logits, labels)
    
    assert loss < 1e-3

def test_backward_shape_and_row_sum():
    ce = CrossEntropy()
    logits = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
    labels = np.array([2, 0])
    
    loss = ce(logits, labels)
    grad = ce.backward()
    
    assert grad.shape == logits.shape
    np.testing.assert_allclose(grad.sum(axis=-1), np.zeros(2), atol=1e-7)