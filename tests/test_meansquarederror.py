'''
Testing all case for checking whether running correctly
'''

import numpy as np

from src.core.criterion.mean_squared_error import MeanSquaredError


def test_mse_forward_manual():
    loss_fn = MeanSquaredError()

    preds = np.array([1.0, 2.0, 3.0])
    labels = np.array([1.0, 0.0, 5.0])

    loss = loss_fn(preds, labels)

    # diff = [0, 2, -2]
    # squared = [0, 4, 4]
    # mean = 8/3
    expected = 8.0 / 3.0

    np.testing.assert_allclose(loss, expected)


def test_mse_backward_manual():
    loss_fn = MeanSquaredError()

    preds = np.array([1.0, 2.0, 3.0])
    labels = np.array([1.0, 0.0, 5.0])

    _ = loss_fn(preds, labels)
    grad = loss_fn.backward()

    # grad = 2 * (preds - labels) / num_elements
    expected = np.array([0.0, 4.0 / 3.0, -4.0 / 3.0])

    np.testing.assert_allclose(grad, expected)


def test_mse_backward_matrix_uses_all_elements():
    loss_fn = MeanSquaredError()

    preds = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.zeros_like(preds)

    _ = loss_fn(preds, labels)
    grad = loss_fn.backward()

    expected = 2.0 * preds / preds.size

    np.testing.assert_allclose(grad, expected)


def test_rmse_forward_manual():
    loss_fn = MeanSquaredError()

    preds = np.array([1.0, 2.0, 3.0])
    labels = np.array([1.0, 0.0, 5.0])

    loss = loss_fn(preds, labels, root=True)

    expected_mse = 8.0 / 3.0
    expected_rmse = np.sqrt(expected_mse)

    np.testing.assert_allclose(loss, expected_rmse)


def test_rmse_backward_manual():
    loss_fn = MeanSquaredError()

    preds = np.array([1.0, 2.0, 3.0])
    labels = np.array([1.0, 0.0, 5.0])

    rmse = loss_fn(preds, labels, root=True)
    grad = loss_fn.backward()

    mse_grad = 2.0 * (preds - labels) / preds.size
    expected = mse_grad / (2.0 * rmse)

    np.testing.assert_allclose(grad, expected)


def test_mse_numerical_gradient():
    loss_fn = MeanSquaredError()

    preds = np.array([0.2, -1.0, 2.0], dtype=np.float64)
    labels = np.array([1.0, 0.0, 1.0], dtype=np.float64)

    _ = loss_fn(preds, labels)
    grad = loss_fn.backward()

    h = 1e-6
    numerical_grad = np.zeros_like(preds)

    for i in range(preds.shape[0]):
        old = preds[i]

        preds[i] = old + h
        loss_plus = loss_fn(preds, labels)

        preds[i] = old - h
        loss_minus = loss_fn(preds, labels)

        preds[i] = old

        numerical_grad[i] = (loss_plus - loss_minus) / (2 * h)

    np.testing.assert_allclose(grad, numerical_grad, rtol=1e-5, atol=1e-5)