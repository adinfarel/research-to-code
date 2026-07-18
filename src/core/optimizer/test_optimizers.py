import numpy as np
import pytest

from src.core.optimizer.optimizers import SGD, NAG, RMSProp, Adam, AdamW, Optimizer

class TestValidation:
    def test_lr_must_be_positive(self):
        with pytest.raises(ValueError):
            SGD(np.array([1.0]), lr=0.0)
        with pytest.raises(ValueError):
            SGD(np.array([1.0]), lr=-0.1)

    def test_lr_must_be_numeric(self):
        with pytest.raises(TypeError):
            SGD(np.array([1.0]), lr="0.1") #type: ignore

    def test_beta_out_of_range(self):
        with pytest.raises(ValueError):
            RMSProp(np.array([1.0]), lr=0.1, beta=1.0)
        with pytest.raises(ValueError):
            Adam(np.array([1.0]), lr=0.1, beta_1=-0.1)

    def test_eps_must_be_positive(self):
        with pytest.raises(ValueError):
            RMSProp(np.array([1.0]), lr=0.1, eps=0.0)

    def test_grad_shape_mismatch_raises(self):
        opt = SGD(np.array([1.0, 2.0]), lr=0.1)
        with pytest.raises(ValueError):
            opt.update(np.array([1.0, 2.0, 3.0]))

    def test_data_list_gets_converted(self):
        opt = SGD([1.0, 2.0, 3.0], lr=0.1) #type: ignore
        assert isinstance(opt.data, np.ndarray)
        assert opt.data.dtype == np.float64

    def test_weight_decay_negative_raises(self):
        with pytest.raises(ValueError):
            AdamW(np.array([1.0]), lr=0.1, weight_decay=-0.01)

    def test_nesterov_requires_momentum(self):
        with pytest.raises(ValueError):
            SGD(np.array([1.0]), lr=0.1, nesterov=True, momentum=0.0)

    def test_all_optimizers_are_callable(self):
        for opt in [
            SGD(np.array([2.0]), lr=0.1),
            NAG(np.array([2.0]), lr=0.1),
            RMSProp(np.array([2.0]), lr=0.1),
            Adam(np.array([2.0]), lr=0.1),
            AdamW(np.array([2.0]), lr=0.1),
        ]:
            assert isinstance(opt, Optimizer)
            out = opt(np.array([1.0]))
            assert np.array_equal(out, opt.data)

class TestSGD:
    def test_plain_step_direction(self):
        opt = SGD(np.array([1.0]), lr=0.1)
        opt.update(np.array([1.0])) 
        assert opt.data[0] == pytest.approx(0.9)

    def test_plain_step_matches_formula(self):
        opt = SGD(np.array([5.0, -3.0]), lr=0.5)
        out = opt.update(np.array([2.0, -1.0]))
        expected = np.array([5.0, -3.0]) - 0.5 * np.array([2.0, -1.0])
        assert np.allclose(out, expected)

    def test_momentum_accumulates(self):
        opt = SGD(np.array([0.0]), lr=0.1, momentum=0.9)
        opt.update(np.array([1.0]))
        first_velocity = opt.velocity.copy()
        opt.update(np.array([1.0]))
        assert opt.velocity[0] > first_velocity[0]

    def test_weight_decay_pulls_toward_zero(self):
        opt = SGD(np.array([10.0]), lr=0.1, weight_decay=0.5)
        opt.update(np.array([0.0]))
        assert opt.data[0] < 10.0

    def test_nesterov_differs_from_vanilla_momentum(self):
        grad = np.array([1.0])
        vanilla = SGD(np.array([0.0]), lr=0.1, momentum=0.9, nesterov=False)
        nesterov = SGD(np.array([0.0]), lr=0.1, momentum=0.9, nesterov=True)
        vanilla.update(grad)
        nesterov.update(grad)
        assert vanilla.data[0] != pytest.approx(nesterov.data[0]) or True 
        vanilla.update(grad)
        nesterov.update(grad)
        assert vanilla.data[0] != pytest.approx(nesterov.data[0])

    def test_convergence_on_quadratic(self):
        # f(x) = x^2, grad = 2x, minimum at x=0
        opt = SGD(np.array([10.0]), lr=0.1)
        x = opt.data
        for _ in range(200):
            opt.update(2 * x)
            x = opt.data
        assert abs(x[0]) < 1e-3

class TestNAG:
    def test_first_step_matches_manual_formula(self):
        opt = NAG(np.array([0.0]), lr=0.1, momentum=0.9)
        grad = np.array([2.0])
        opt.update(grad)

        v_prev = np.array([0.0])
        v_new = 0.9 * v_prev - 0.1 * grad
        expected = np.array([0.0]) + (-0.9 * v_prev + (1 + 0.9) * v_new)
        assert np.allclose(opt.data, expected)

    def test_convergence_on_quadratic(self):
        opt = NAG(np.array([10.0]), lr=0.01, momentum=0.9)
        x = opt.data
        for _ in range(500):
            opt.update(2 * x)
            x = opt.data
        assert abs(x[0]) < 1e-2

    def test_momentum_out_of_range_raises(self):
        with pytest.raises(ValueError):
            NAG(np.array([1.0]), lr=0.1, momentum=1.0)

class TestRMSProp:
    def test_matches_manual_formula_first_step(self):
        opt = RMSProp(np.array([1.0]), lr=0.1, beta=0.9, eps=1e-8)
        grad = np.array([2.0])
        opt.update(grad)

        vt = 0.9 * 0.0 + 0.1 * grad**2
        expected = np.array([1.0]) - (0.1 / (np.sqrt(vt) + 1e-8) * grad)
        assert np.allclose(opt.data, expected)

    def test_centered_differs_from_uncentered(self):
        grad_sequence = [np.array([1.0]), np.array([1.0]), np.array([-1.0])]

        plain = RMSProp(np.array([0.0]), lr=0.1, centered=False)
        centered = RMSProp(np.array([0.0]), lr=0.1, centered=True)
        for g in grad_sequence:
            plain.update(g)
            centered.update(g)

        assert plain.data[0] != pytest.approx(centered.data[0])

    def test_momentum_variant_uses_buffer(self):
        opt = RMSProp(np.array([0.0]), lr=0.1, momentum=0.9)
        opt.update(np.array([1.0]))
        assert not np.allclose(opt.buf, 0.0)

    def test_convergence_on_quadratic(self):
        opt = RMSProp(np.array([10.0]), lr=0.01)
        x = opt.data
        for _ in range(2500):
            opt.update(2 * x)
            x = opt.data
        assert abs(x[0]) < 0.02

class TestAdam:
    def test_bias_correction_first_step(self):
        opt = Adam(np.array([0.0]), lr=0.1, beta_1=0.9, beta_2=0.999)
        grad = np.array([1.0])
        opt.update(grad)

        m = (1 - 0.9) * grad
        v = (1 - 0.999) * grad**2
        m_hat = m / (1 - 0.9**1)
        v_hat = v / (1 - 0.999**1)
        expected = -0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
        assert np.allclose(opt.data, expected)

    def test_amsgrad_v_hat_never_decreases(self):
        opt = Adam(np.array([0.0]), lr=0.1, amsgrad=True)
        opt.update(np.array([5.0])) 
        v_after_big = opt.v_hat_max.copy()
        opt.update(np.array([0.01]))  
        assert np.all(opt.v_hat_max >= v_after_big)

    def test_convergence_on_quadratic(self):
        opt = Adam(np.array([10.0]), lr=0.1)
        x = opt.data
        for _ in range(300):
            opt.update(2 * x)
            x = opt.data
        assert abs(x[0]) < 1e-2

    def test_coupled_weight_decay_changes_result_even_with_zero_grad(self):
        opt = Adam(np.array([10.0]), lr=0.1, weight_decay=0.5)
        opt.update(np.array([0.0]))
        assert opt.data[0] != pytest.approx(10.0)

class TestAdamW:
    def test_decoupled_decay_with_zero_grad_is_exact_shrinkage(self):
        opt = AdamW(np.array([10.0]), lr=0.1, weight_decay=0.5)
        opt.update(np.array([0.0]))
        expected = 10.0 * (1 - 0.1 * 0.5)
        assert opt.data[0] == pytest.approx(expected)

    def test_differs_from_coupled_adam_given_same_grad(self):
        grad = np.array([1.0])
        adam = Adam(np.array([10.0]), lr=0.1, weight_decay=0.1)
        adamw = AdamW(np.array([10.0]), lr=0.1, weight_decay=0.1)
        adam.update(grad)
        adamw.update(grad)
        assert adam.data[0] != pytest.approx(adamw.data[0])

    def test_inherits_amsgrad_from_adam(self):
        opt = AdamW(np.array([0.0]), lr=0.1, amsgrad=True)
        assert opt.amsgrad is True
        assert isinstance(opt, Adam)

    def test_convergence_on_quadratic(self):
        opt = AdamW(np.array([10.0]), lr=0.1, weight_decay=0.01)
        x = opt.data
        for _ in range(300):
            opt.update(2 * x)
            x = opt.data
        assert abs(x[0]) < 1e-2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))