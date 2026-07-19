import numpy as np
import pytest

from criterion import MeanSquaredError, CrossEntropy, BCEWithLogits, Criterion

def numerical_grad(loss_fn, logits: np.ndarray, h: float = 1e-5) -> np.ndarray:
    '''
    loss_fn: callable (logits_perturbed) -> scalar loss
    Central difference per elemen: (f(x+h) - f(x-h)) / 2h
    '''
    grad = np.zeros_like(logits, dtype=np.float64)
    it = np.nditer(logits, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = logits[idx]

        logits[idx] = original + h
        plus = loss_fn(logits)

        logits[idx] = original - h
        minus = loss_fn(logits)

        logits[idx] = original
        grad[idx] = (plus - minus) / (2 * h)
    return grad

class TestMeanSquaredError:
    def test_forward_matches_manual(self):
        mse = MeanSquaredError()
        logits = np.array([1.0, 2.0, 3.0])
        labels = np.array([1.5, 2.5, 2.0])
        loss = mse(logits, labels)
        expected = np.mean((logits - labels) ** 2)
        assert loss == pytest.approx(expected)

    def test_root_forward(self):
        mse = MeanSquaredError()
        logits = np.array([1.0, 2.0, 3.0])
        labels = np.array([1.5, 2.5, 2.0])
        rmse = mse(logits, labels, root=True)
        assert rmse == pytest.approx(np.sqrt(np.mean((logits - labels) ** 2)))

    def test_backward_before_call_raises(self):
        mse = MeanSquaredError()
        with pytest.raises(AssertionError):
            mse.backward()

    def test_backward_gradcheck_mse(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=5)
        labels = rng.normal(size=5)

        mse = MeanSquaredError()
        mse(logits, labels)
        analytical = mse.backward()

        numeric = numerical_grad(lambda x: MeanSquaredError()(x, labels), logits)
        assert np.allclose(analytical, numeric, atol=1e-5)

    def test_backward_gradcheck_rmse(self):
        rng = np.random.default_rng(1)
        logits = rng.normal(size=5)
        labels = rng.normal(size=5)

        mse = MeanSquaredError()
        mse(logits, labels, root=True)
        analytical = mse.backward()

        numeric = numerical_grad(lambda x: MeanSquaredError()(x, labels, root=True), logits)
        assert np.allclose(analytical, numeric, atol=1e-5)

    def test_rmse_zero_error_returns_zero_grad(self):
        mse = MeanSquaredError()
        logits = np.array([1.0, 2.0, 3.0])
        labels = np.array([1.0, 2.0, 3.0])  
        mse(logits, labels, root=True)
        grad = mse.backward()
        assert np.allclose(grad, 0.0)

class TestCrossEntropy:
    def test_softmax_sums_to_one(self):
        ce = CrossEntropy()
        logits = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.7]])
        probs = ce._softmax(logits)
        assert np.allclose(probs.sum(axis=-1), 1.0)

    def test_log_softmax_matches_log_of_softmax(self):
        ce = CrossEntropy()
        logits = np.array([[1.0, 2.0, 3.0]])
        assert np.allclose(ce._log_softmax(logits), np.log(ce._softmax(logits)))

    def test_no_overflow_on_large_logits(self):
        ce = CrossEntropy()
        logits = np.array([[1000.0, 999.0, 0.0]])
        probs = ce._softmax(logits)
        assert np.all(np.isfinite(probs))
        assert np.allclose(probs.sum(axis=-1), 1.0)

    def test_forward_matches_manual_two_class(self):
        ce = CrossEntropy()
        logits = np.array([[2.0, 0.5], [0.1, 1.2]])
        labels = np.array([0, 1])
        loss = ce(logits, labels)

        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        expected = -np.mean(np.log(probs[np.arange(2), labels]))
        assert loss == pytest.approx(expected)

    def test_shape_mismatch_raises(self):
        ce = CrossEntropy()
        logits = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([0, 1, 0])  
        with pytest.raises(ValueError):
            ce(logits, labels)

    def test_backward_before_call_raises(self):
        ce = CrossEntropy()
        with pytest.raises(AssertionError):
            ce.backward()

    def test_backward_gradcheck(self):
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(4, 3))
        labels = np.array([0, 2, 1, 1])

        ce = CrossEntropy()
        ce(logits, labels)
        analytical = ce.backward()

        numeric = numerical_grad(lambda x: CrossEntropy()(x, labels), logits)
        assert np.allclose(analytical, numeric, atol=1e-5)

    def test_confident_correct_prediction_lowers_loss(self):
        ce = CrossEntropy()
        confident_logits = np.array([[10.0, 0.0, 0.0]])
        unsure_logits = np.array([[0.1, 0.0, 0.0]])
        labels = np.array([0])
        assert ce(confident_logits, labels) < ce(unsure_logits, labels)

class TestBCEWithLogits:
    def test_forward_matches_manual(self):
        bce = BCEWithLogits()
        logits = np.array([0.5, -1.0, 2.0])
        labels = np.array([1.0, 0.0, 1.0])
        loss = bce(logits, labels)

        probs = 1 / (1 + np.exp(-logits))
        expected = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        assert loss == pytest.approx(expected)

    def test_shape_mismatch_raises(self):
        bce = BCEWithLogits()
        with pytest.raises(AssertionError):
            bce(np.array([1.0, 2.0]), np.array([1.0]))

    def test_no_nan_on_extreme_logits(self):
        bce = BCEWithLogits()
        logits = np.array([1000.0, -1000.0])
        labels = np.array([1.0, 0.0])
        loss = bce(logits, labels)
        assert np.isfinite(loss)

    def test_backward_before_call_raises(self):
        bce = BCEWithLogits()
        with pytest.raises(AssertionError):
            bce.backward()

    def test_backward_gradcheck(self):
        rng = np.random.default_rng(3)
        logits = rng.normal(size=6)
        labels = (rng.uniform(size=6) > 0.5).astype(np.float64)

        bce = BCEWithLogits()
        bce(logits, labels)
        analytical = bce.backward()

        numeric = numerical_grad(lambda x: BCEWithLogits()(x, labels), logits)
        assert np.allclose(analytical, numeric, atol=1e-5)

class TestCriterionBase:
    def test_all_criterions_are_criterion_subclass(self):
        for cls in [MeanSquaredError, CrossEntropy, BCEWithLogits]:
            assert issubclass(cls, Criterion)

    def test_cache_resets_are_independent_across_instances(self):
        mse_a = MeanSquaredError()
        mse_b = MeanSquaredError()
        mse_a(np.array([1.0]), np.array([2.0]))
        assert mse_b.CACHE is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))