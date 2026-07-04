import numpy as np

from src.evaluation.classification import (
    confusion_matrix,
    accuracy_score,
    precision,
    recall,
    f1_score,
)


# Fixed example, hand-verified:
# y_true = [1,0,1,1,0], y_pred = [1,0,0,1,1]
# TP=2 (idx 0,3), FP=1 (idx 4), TN=1 (idx 1), FN=1 (idx 2)
Y_TRUE = np.array([1, 0, 1, 1, 0])
Y_PRED = np.array([1, 0, 0, 1, 1])


def test_confusion_matrix_values():
    cm = confusion_matrix(Y_TRUE, Y_PRED)
    # layout: [[TP, FP], [TN, FN]]
    assert cm[0][0] == 2  # TP
    assert cm[0][1] == 1  # FP
    assert cm[1][0] == 1  # TN
    assert cm[1][1] == 1  # FN


def test_accuracy_known_value():
    acc = accuracy_score(Y_TRUE, Y_PRED)
    np.testing.assert_allclose(acc, 0.6, atol=1e-4)


def test_precision_recall_f1_known_values():
    prec = precision(Y_TRUE, Y_PRED)
    rec = recall(Y_TRUE, Y_PRED)
    f1 = f1_score(Y_TRUE, Y_PRED)

    np.testing.assert_allclose(prec, 2 / 3, atol=1e-4)
    np.testing.assert_allclose(rec, 2 / 3, atol=1e-4)
    np.testing.assert_allclose(f1, 2 / 3, atol=1e-4)


def test_accuracy_misleading_on_imbalanced_data():
    # gets high accuracy despite being useless for minority class.
    y_true = np.array([0] * 9 + [1])  # 90% class 0, 10% class 1
    y_pred_lazy = np.array([0] * 10)  # always predicts majority

    acc = accuracy_score(y_true, y_pred_lazy)
    rec = recall(y_true, y_pred_lazy)

    assert acc >= 0.89  # accuracy looks great...
    assert rec == 0.0  # ...but completely fails to catch positive class