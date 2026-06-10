'''
Build atom implementations of autograd in mini-version >.<
'''

import math
import numpy as np

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out   = Value(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out   = Value(self.data * other.data, (self, other), "*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int and float pow for now"
        out   = Value(self.data**other, (self,), f"**{other}")
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward
        return out
    
    def relu(self):
        x = self.data
        out = Value(max(0, x), (self,), "relu")
        
        def _backward():
            if out.data:
                self.grad += 1.0 * out.grad
            else:
                self.grad += 0.0 * out.grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")
        
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def backward(self):
        self.grad = 1.0
        topo = []
        visited = set()
        
        def _build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    _build_topo(child)
                topo.append(node)
        
        _build_topo(self)
        for node in reversed(topo):
            node._backward()
    
def f(x):
    return 3*x**2 - 4*x + 5

if __name__ == "__main__":
    a = Value(2)
    b = Value(3)
    c = a - b
    c.grad = 1.0
    c._backward()
    print(c, c.grad)
    print(a.grad, b.grad)

    d = Value(3)
    e = d ** 2
    e.grad = 1.0
    e._backward()
    print(e, e.grad, d, d.grad)

    r = Value(-1)
    tanh = r.tanh()
    tanh.grad = 1.0
    tanh._backward()
    print(tanh, tanh.grad, r, r.grad)

    l = Value(-0.1)
    relu = l.relu()
    relu.grad = 1.0
    relu._backward()
    print(relu, relu.grad, l, l.grad)
