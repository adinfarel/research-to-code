'''
testing whether the function running correctly
'''

import math

from src.core.micrograd.micrograd import Value

def assert_close(a, b, tol=1e-6):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), f"{a}, {b}"
    
def test_add_backward():
    a = Value(2.0)
    b = Value(3.0)
    
    c = a + b
    c.backward()
    
    assert_close(c.data, 5.0)
    assert_close(a.grad, 1.0)
    assert_close(b.grad, 1.0)

def test_mul_backward():
    a = Value(2.0)
    b = Value(3.0)
    
    c = a * b
    c.backward()
    
    assert_close(c.data, 6.0)
    assert_close(a.grad, 3.0)
    assert_close(b.grad, 2.0)


def test_sub_backward():
    a = Value(2.0)
    b = Value(3.0)

    c = a - b
    c.backward()

    assert_close(c.data, -1.0)
    assert_close(a.grad, 1.0)
    assert_close(b.grad, -1.0)


def test_reverse_sub_backward():
    a = Value(2.0)

    c = 10.0 - a
    c.backward()

    assert_close(c.data, 8.0)
    assert_close(a.grad, -1.0)


def test_div_backward():
    a = Value(6.0)
    b = Value(3.0)

    c = a / b
    c.backward()

    assert_close(c.data, 2.0)
    assert_close(a.grad, 1.0 / 3.0)
    assert_close(b.grad, -6.0 / 9.0)


def test_pow_backward():
    x = Value(3.0)

    y = x ** 2
    y.backward()

    assert_close(y.data, 9.0)
    assert_close(x.grad, 6.0)

def test_polynomial_backward():
    x = Value(2.0)
    
    y = 3 * x**2 - 4 * x + 5
    y.backward()
    
    # y = 3x^2 - 4x + 5
    # dy/dx = 6x - 4
    # at x = 2 -> 8
    assert_close(y.data, 9.0)
    assert_close(x.grad, 8.0)

def test_branching_gradient_accumulates():
    a = Value(2.0)

    b = a + a
    b.backward()

    assert_close(b.data, 4.0)
    assert_close(a.grad, 2.0)

def test_reuse_node_gradient_accumulates():
    a = Value(3.0)

    b = a * a
    b.backward()

    assert_close(b.data, 9.0)
    assert_close(a.grad, 6.0)

def test_tanh_backward():
    x = Value(0.5)

    y = x.tanh()
    y.backward()

    expected = 1.0 - math.tanh(0.5) ** 2

    assert_close(y.data, math.tanh(0.5))
    assert_close(x.grad, expected)

def test_relu_positive_backward():
    x = Value(2.0)

    y = x.relu()
    y.backward()

    assert_close(y.data, 2.0)
    assert_close(x.grad, 1.0)
    
def test_relu_negative_backward():
    x = Value(-2.0)

    y = x.relu()
    y.backward()

    assert_close(y.data, 0.0)
    assert_close(x.grad, 0.0)

def test_exp_backward():
    x = Value(2.0)

    y = x.exp()
    y.backward()

    assert_close(y.data, math.exp(2.0))
    assert_close(x.grad, math.exp(2.0))

def test_composed_expression_backward():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    d = a * b + c
    e = d.relu()
    f = e * 2.0
    f.backward()
    
    # d = 2 * -3 + 10 = 4
    # e = relu(4) = 4
    # f = 4 * 2 = 8
    #
    # f = 1.0
    # e = 2.0 * 1.0 = 2.0
    # d = 2.0 * 1.0 = 2.0
    # a = 2.0 * -3.0 = -6.0
    # b = 2.0 * 2.0 = 4.0
    # c = 2.0 * 1.0 = 2.0
    
    assert_close(f.data, 8.0)
    assert_close(a.grad, -6.0)
    assert_close(b.grad, 4.0)
    assert_close(c.grad, 2.0)