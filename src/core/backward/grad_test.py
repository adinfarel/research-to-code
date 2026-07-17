import math
import pytest
 
from src.core.backward.micrograd import Value, no_grad
 
def numerical_grad(f, x: float, h: float = 1e-6) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)
 
 
def gradcheck(f, x0: float, atol: float = 1e-4) -> None:
    x = Value(x0)
    y = f(x)
    y.backward()
    analytical = x.grad
 
    numeric = numerical_grad(lambda v: f(Value(v)).data, x0)
 
    assert math.isclose(analytical, numeric, abs_tol=atol), (
        f"gradcheck failed at x={x0}: analytical={analytical}, "
        f"numerical={numeric}"
    )
 

class TestForward:
    def test_add(self):
        assert (Value(2.0) + Value(3.0)).data == 5.0
 
    def test_mul(self):
        assert (Value(2.0) * Value(3.0)).data == 6.0
 
    def test_sub(self):
        assert (Value(5.0) - Value(3.0)).data == 2.0
 
    def test_div(self):
        assert (Value(6.0) / Value(2.0)).data == 3.0
 
    def test_pow(self):
        assert (Value(3.0) ** 2).data == 9.0
 
    def test_tanh_bounds(self):
        assert -1.0 <= Value(50.0).tanh().data <= 1.0
        assert -1.0 <= Value(-50.0).tanh().data <= 1.0
        assert -1.0 < Value(3.0).tanh().data < 1.0
 
    def test_tanh_no_overflow(self):
        Value(1000.0).tanh() 
 
    def test_relu_negative(self):
        assert Value(-5.0).relu().data == 0.0
 
    def test_relu_positive(self):
        assert Value(5.0).relu().data == 5.0
 
    def test_exp(self):
        assert math.isclose(Value(1.0).exp().data, math.e)
 
    def test_radd_rmul(self):
        assert (2 + Value(3.0)).data == 5.0
        assert (2 * Value(3.0)).data == 6.0
 
    def test_rsub(self):
        assert (5 - Value(3.0)).data == 2.0
 
class TestBackwardKnownValues:
    def test_sub_grad(self):
        a, b = Value(2.0), Value(3.0)
        c = a - b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == -1.0
 
    def test_pow_grad(self):
        d = Value(3.0)
        e = d ** 2
        e.backward()
        assert e.data == 9.0
        assert d.grad == 6.0  # d/dx x^2 = 2x = 2*3 = 6
 
    def test_tanh_grad_saturated(self):
        r = Value(-10.0)
        t = r.tanh()
        t.backward()
        assert t.grad == 1.0
        assert r.grad == pytest.approx(0.0, abs=1e-7)
 
    def test_relu_grad_negative_input(self):
        l = Value(-0.1)
        out = l.relu()
        out.backward()
        assert l.grad == 0.0
 
    def test_relu_grad_positive_input(self):
        l = Value(0.5)
        out = l.relu()
        out.backward()
        assert l.grad == 1.0
 
    def test_quadratic_f(self):
        # f(x) = 3x^2 - 4x + 5, f'(x) = 6x - 4
        x = Value(2.0)
        y = 3 * x ** 2 - 4 * x + 5
        y.backward()
        assert y.data == 9.0
        assert x.grad == 8.0  # 6*2 - 4 = 8
 
class TestDiamondGraph:
    def test_variable_used_twice_in_add(self):
        a = Value(3.0)
        b = a + a  
        b.backward()
        assert b.data == 6.0
        assert a.grad == 2.0  # d(a+a)/da = 2
 
    def test_diamond_graph(self):
        #     a
        #    / \
        #   b   c   (b = a*2, c = a+1)
        #    \ /
        #     d = b*c
        a = Value(2.0)
        b = a * 2
        c = a + 1
        d = b * c
        d.backward()
        # d = (2a)(a+1) = 2a^2 + 2a -> d'(a) = 4a + 2 = 4*2+2 = 10
        assert a.grad == 10.0
 
    def test_multiple_backward_calls_reset_grad(self):
        a = Value(3.0)
        y1 = a * 2
        y1.backward()
        first_grad = a.grad
        y2 = a * 2
        y2.backward()
        assert a.grad == first_grad  
 
class TestNoGrad:
    def test_requires_grad_off_inside_no_grad(self):
        x = Value(2.0)
        with no_grad():
            y = x * 3
        assert y.requires_grad is False
        assert y._grad_fn is None
 
    def test_grad_enabled_restored_after_context(self):
        x = Value(2.0)
        with no_grad():
            _ = x * 3
        y = x * 3
        assert y.requires_grad is True
 
    def test_backward_noop_when_no_grad(self):
        with no_grad():
            x = Value(2.0)
            y = x * 3
        y.backward()
        assert y.grad == 1.0
 
class TestUtilities:
    def test_detach_breaks_graph(self):
        x = Value(2.0)
        y = x * 3
        z = y.detach()
        assert z.data == y.data
        assert z._prev == ()
        assert z.requires_grad is False
 
    def test_zero_grad(self):
        x = Value(2.0)
        x.grad = 5.0
        x.zero_grad()
        assert x.grad == 0.0
 
class TestGradCheck:
    @pytest.mark.parametrize("x0", [-3.0, -0.5, 0.1, 1.0, 2.7])
    def test_add_const(self, x0):
        gradcheck(lambda x: x + 5, x0)
 
    @pytest.mark.parametrize("x0", [-3.0, -0.5, 0.1, 1.0, 2.7])
    def test_mul_const(self, x0):
        gradcheck(lambda x: x * 4, x0)
 
    @pytest.mark.parametrize("x0", [0.5, 1.0, 2.0, 3.5])
    def test_pow(self, x0):
        gradcheck(lambda x: x ** 3, x0)
 
    @pytest.mark.parametrize("x0", [-2.0, -0.5, 0.5, 2.0])
    def test_tanh(self, x0):
        gradcheck(lambda x: x.tanh(), x0)
 
    @pytest.mark.parametrize("x0", [-2.0, 0.5, 2.0])
    def test_relu_away_from_kink(self, x0):
        gradcheck(lambda x: x.relu(), x0)
 
    @pytest.mark.parametrize("x0", [-1.0, 0.0, 1.0, 2.0])
    def test_exp(self, x0):
        gradcheck(lambda x: x.exp(), x0)
 
    @pytest.mark.parametrize("x0", [0.5, 1.0, 2.0, 3.0])
    def test_composite_quadratic(self, x0):
        gradcheck(lambda x: 3 * x ** 2 - 4 * x + 5, x0)
 
    @pytest.mark.parametrize("x0", [-1.0, 0.5, 1.5])
    def test_composite_tanh_mul(self, x0):
        gradcheck(lambda x: (x * 2 + 1).tanh(), x0)
 
 
if __name__ == "__main__":
    import sys
 
    sys.exit(pytest.main([__file__, "-v"]))