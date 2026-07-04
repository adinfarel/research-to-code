import numpy as np

from src.evaluation.regression import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)


def test_mse_perfect_prediction_is_zero():
    y_true = np.array([3.0, 5.0, 7.0])
    y_pred = np.array([3.0, 5.0, 7.0])

    assert mean_squared_error(y_true, y_pred) == 0.0


def test_mse_known_value():
    y_true = np.array([3.0, 5.0])
    y_pred = np.array([2.5, 5.0])
    # errors = [0.5, 0.0] -> squared = [0.25, 0.0] -> mean = 0.125

    result = mean_squared_error(y_true, y_pred)
    np.testing.assert_allclose(result, 0.125, rtol=1e-6)


def test_rmse_is_sqrt_of_mse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.0])

    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    np.testing.assert_allclose(rmse, np.sqrt(mse), rtol=1e-6)


def test_mae_less_sensitive_to_outlier_than_mse():
    # Same base errors, but one has a large outlier. MAE should scale
    # linearly, MSE should scale quadratically (much larger jump).
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred_normal = np.array([1.0, 1.0, 1.0])
    y_pred_outlier = np.array([1.0, 1.0, 10.0])

    mae_normal = mean_absolute_error(y_true, y_pred_normal)
    mae_outlier = mean_absolute_error(y_true, y_pred_outlier)
    mse_normal = mean_squared_error(y_true, y_pred_normal)
    mse_outlier = mean_squared_error(y_true, y_pred_outlier)

    mae_ratio = mae_outlier / mae_normal
    mse_ratio = mse_outlier / mse_normal

    assert mse_ratio > mae_ratio  # MSE blows up faster than MAE