'''
AutoGrad micro version implementation
'''

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

Number = Union[int, float]

_GRAD_ENABLED = True

class Context:
    
    __slots__ = ("saved_values", "adin")
    
    def __init__(self) -> None:
        self.saved_values: tuple = ()
    
    def save_for_backward(self, *values: Number) -> None:
        self.saved_values = values

class Function:
    
    @staticmethod
    def forward(ctx: Context, *args: Number) -> None:
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx: Context, grad_output: float) -> tuple:
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *raw_inputs: Union["Value", Number]) -> "Value":
        inputs = [v if isinstance(v, Value) else Value(v) for v in raw_inputs]
        
        ctx = Context()
        raw_data = [v.data for v in inputs]
        out_data = cls.forward(ctx, *raw_data)
        
        requires_grad = _GRAD_ENABLED and any(v.requires_grad for v in inputs)
        
        out = Value(
            out_data, #type: ignore
            _children=tuple(inputs),
            _op=cls.__name__.lower(),
            requires_grad=requires_grad,
        )
        
        if requires_grad:
            out._grad_fn = cls
            out._ctx = ctx
        
        return out

class Value:
    
    def __init__(
        self,
        data: Number,
        _children: Sequence["Value"] = (),
        _op: str = "",
        requires_grad: bool = True
    ) -> None:
        self.data = data
        self.grad = 0.0
        self._prev: tuple = tuple(_children)
        self._op = _op
        self.requires_grad = requires_grad

        self._grad_fn: Optional[type] = None
        self._ctx: Optional[Context] = None
    
    def __repr__(self) -> str:
        grad_fn_repr = f", grad_fn={self._grad_fn.__name__}" if self._grad_fn else ""
        return f"Value(data={self.data}{grad_fn_repr})"
    
    def __add__(self, other: Union["Value", Number]) -> "Value":
        return Add.apply(self, other)
    
    def __radd__(self, other: Number) -> "Value":
        return self + other
    
    def __mul__(self, other: Union["Value", Number]) -> "Value":
        return Mul.apply(self, other)
    
    def __rmul__(self, other: Number) -> "Value":
        return self * other
    
    def __neg__(self) -> "Value":
        return self * -1
    
    def __sub__(self, other: Union["Value", Number]) -> "Value":
        return self + (-other if isinstance(other, Value) else -other)
    
    def __rsub__(self, other: Number) -> "Value":
        return other + (-self)
    
    def __pow__(self, other: Number) -> "Value":
        assert isinstance(other, (int, float)), "only support int and float"
        return Pow.apply(self, other)
    
    def __truediv__(self, other: Union["Value", Number]) -> "Value":
        return self * other ** -1
        
    
    def tanh(self) -> "Value":
        return Tanh.apply(self)
    
    def relu(self) -> "Value":
        return ReLU.apply(self)
    
    def exp(self) -> "Value":
        return Exp.apply(self)

    def zero_grad(self) -> None:
        self.grad = 0.0
    
    def detach(self):
        return Value(self.data, requires_grad=False)
    
    def backward(self) -> None:
        topo: list["Value"] = []
        visited: set = set()
        
        def _build_topo(node: "Value") -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    _build_topo(child)
                topo.append(node)
        
        _build_topo(self)
        
        for node in topo:
            node.grad = 0.0
        self.grad = 1.0
        
        for node in reversed(topo):
            if node._grad_fn is None:
                continue
            grads = node._grad_fn.backward(node._ctx, node.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for child, grad in zip(node._prev, grads):
                if grad is not None and child.requires_grad:
                    child.grad += grad

class no_grad:
    
    def __enter__(self) -> "no_grad":
        global _GRAD_ENABLED
        self._prev_stats = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self
    
    def __exit__(self, *exc_info) -> None:
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev_stats

class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: Number, b: Number) -> Number:
        return a + b
    
    @staticmethod
    def backward(ctx: Context, grad_output: float) -> tuple:
        # d(a+b)/da = 1
        # d((a+h+b) - (a+b))/d((a+h) - a)
        # d(h)/d(h) = 1
        return grad_output, grad_output

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Number, b: Number) -> Number:
        ctx.save_for_backward(a, b)
        return a * b
    
    @staticmethod
    def backward(ctx: Context, grad_output: float) -> tuple:
        a, b = ctx.saved_values
        # d(a*b)/da = b, d(a*b)/db = a
        return grad_output * b, grad_output * a   

class Pow(Function):
    @staticmethod
    def forward(ctx: Context, base: Number, exponent: Number) -> Number:
        ctx.save_for_backward(base, exponent)
        return base ** exponent
    
    @staticmethod
    def backward(ctx: Context, grad_output: float) -> tuple:
        base, exponent = ctx.saved_values
        # d(base^n)/dbase = n * base^(n-1)
        return grad_output * exponent * (base ** (exponent - 1)), None
    
class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, x: Number) -> Number:
        t = math.tanh(x)
        ctx.save_for_backward(t)
        return t
    
    @staticmethod
    def backward(ctx: Context, grad_output: float) -> tuple:
        (t,) = ctx.saved_values
        return grad_output * (1 - t ** 2),

class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, x: Number) -> Number:
        ctx.save_for_backward(x)
        return max(0.0, x)
 
    @staticmethod
    def backward(ctx: Context, grad_output: float) -> tuple:
        (x,) = ctx.saved_values
        return (grad_output * (1.0 if x > 0 else 0.0),)

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, x: Number) -> Number:
        out = math.exp(x)
        ctx.save_for_backward(out)
        return out
 
    @staticmethod
    def backward(ctx: Context, grad_output: float) -> tuple:
        (out,) = ctx.saved_values
        return (grad_output * out,)