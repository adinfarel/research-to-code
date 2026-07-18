'''
Build atom implemetations of Adam, RMSProp, SGD, AdamW

NOTE: here some intuition
    # if get +grad it means direction of grad is decreasing, must -weight loss decrease
    # if get -grad it means direction of grad is increasing, must +weight loss decrease
    # so, changing x towards opposite of sign grad will decrease loss to minimum loss
    # i hope yall understand intuition from me >.<
'''

from __future__ import annotations

import numpy as np
from abc import abstractmethod, ABC

class Optimizer(ABC):
    
    def __init__(self, data: np.ndarray, lr: float | int):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
 
        if not isinstance(lr, (int, float)):
            raise TypeError(f"lr must be float, got {lr}")
 
        if lr <= 0:
            raise ValueError("lr must be positive")
        
        self.data = np.asarray(data, dtype=np.float64)
        self.lr = float(lr)
    
    @staticmethod
    def _check_beta(beta: float, name: str = "beta") -> float:
        if not 0 <= beta < 1:
            raise ValueError(f"{name} must between 0 and 1, got {beta}")
        return float(beta)
    
    @staticmethod
    def _check_eps(eps: float) -> float:
        if eps <= 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        return float(eps)
    
    @staticmethod
    def _check_weight_decay(weight_decay: float) -> float:
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        return float(weight_decay)
    
    @staticmethod
    def _check_momentum(momentum: float) -> float:
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be non-negative, got {momentum}")
        return float(momentum)
    
    def _prepare_grad(self, grad: np.ndarray) -> np.ndarray:
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != self.data.shape:
            raise ValueError("grad shape and data shape mismatch")
        return grad
    
    @abstractmethod
    def update(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def __call__(self, grad: np.ndarray) -> np.ndarray:
        return self.update(grad)
    
    def _repr_params(self) -> dict:
        return {"lr": self.lr}

class SGD(Optimizer):
    
    def __init__(
        self,
        data: np.ndarray,
        lr: float | int,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        super().__init__(data, lr)
        
        self.momentum = self._check_momentum(momentum)
        
        if not 0 <= dampening < 1:
            raise ValueError(f"dampening must between 0 and 1, got {dampening}")
 
        if nesterov and (self.momentum <= 0 or dampening != 0):
            raise ValueError("nesterov requires momentum > 0 and dampening == 0")
        
        self.dampening = float(dampening)
        self.weight_decay = self._check_weight_decay(weight_decay)
        self.nesterov = bool(nesterov)
        
        self.velocity = np.zeros_like(self.data)
        self.t = 0
    
    def _repr_params(self) -> dict:
        return {"lr": self.lr, "momentum": self.momentum, "nesterov": self.nesterov}
    
    def update(self, grad: np.ndarray) -> np.ndarray:
        grad = self._prepare_grad(grad)
        
        if self.weight_decay != 0:
            grad = grad + self.weight_decay * self.data # L2 coupled gradient
        
        if self.momentum != 0:
            if self.t == 0:
                self.velocity = grad.copy()
            else:
                self.velocity = self.momentum * self.velocity + (1 - self.dampening) * grad
            
            if self.nesterov:
                # lookahead: use grad now
                grad = grad + self.momentum * self.velocity
            else:
                grad = self.velocity
        
        self.t += 1
        self.data += -self.lr * grad # moving data (weight or parameter) opposite to grad so change 
        # weight towards opposite grad can decrease loss 
        return self.data

class NAG(Optimizer):
    
    def __init__(
        self,
        data: np.ndarray,
        lr: float | int,
        momentum: float = 0.9
    ):
        super().__init__(data, lr)
        
        self.momentum = self._check_momentum(momentum)
        self.velocity = np.zeros_like(self.data)
    
    def _repr_params(self) -> dict:
        return {"lr": self.lr, "momentum": self.momentum}
    
    def update(self, grad: np.ndarray) -> np.ndarray:
        grad = self._prepare_grad(grad)
        
        v_prev = self.velocity.copy()
        self.velocity = self.momentum * self.velocity - self.lr * grad
        
        self.data += -self.momentum * v_prev + (1 + self.momentum) * self.velocity
        return self.data

class RMSProp(Optimizer):
    
    def __init__(
        self,
        data: np.ndarray,
        lr: float | int,
        eps: float = 1e-8,
        beta: float = 0.9,
        momentum: float = 0.0,
        centered: bool = False,
    ):
        super().__init__(data, lr)
        
        self.eps = self._check_eps(eps)
        self.beta = self._check_beta(beta)
        self.momentum = self._check_momentum(momentum)
        self.centered = bool(centered)
        
        self.acm_grad = np.zeros_like(self.data)
        self.grad_avg = np.zeros_like(self.data)
        self.buf = np.zeros_like(self.data)
    
    def _repr_params(self) -> dict:
        return {"lr": self.lr, "beta": self.beta, "centered": self.centered}
    
    def update(self, grad: np.ndarray):
        grad = self._prepare_grad(grad)
        
        vt = (self.beta * self.acm_grad) + (1 - self.beta) * grad ** 2
        self.acm_grad = vt # just store acum new grad, not like AdaGrad that (+=) increase constantly from early
        
        if self.centered:
            self.grad_avg = self.beta * self.grad_avg + (1 - self.beta) * grad
            avg = vt - self.grad_avg ** 2
        else:
            avg = vt
        
        denom = np.sqrt(avg) + self.eps
        
        if self.momentum != 0:
            self.buf = self.momentum * self.buf + grad / denom
            self.data += -self.lr * self.buf
        else:
            self.data += -(self.lr / denom * grad)
        
        return self.data

class Adam(Optimizer):
    
    def __init__(
        self,
        data: np.ndarray,
        lr: int | float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        super().__init__(data, lr)
        
        self.beta_1 = self._check_beta(beta_1, "beta_1")
        self.beta_2 = self._check_beta(beta_2, "beta_2")
        self.eps = self._check_eps(eps)
        self.weight_decay = self._check_weight_decay(weight_decay)
        self.amsgrad = bool(amsgrad)
        
        self.momentum = np.zeros_like(self.data) # m
        self.acm_grad = np.zeros_like(self.data) # v
        self.v_hat_max = np.zeros_like(self.data) # use if amsgrad=True
        self.t = 0
    
    def _repr_params(self) -> dict:
        return {"lr": self.lr, "beta_1": self.beta_1, "beta_2": self.beta_2, "amsgrad": self.amsgrad}
    
    def update(self, grad: np.ndarray) -> np.ndarray:
        grad = self._prepare_grad(grad)
        
        if self.weight_decay != 0:
            grad = grad + self.weight_decay * self.data # L2 coupled
        
        self.t += 1
        
        self.momentum = self.beta_1 * self.momentum + (1 - self.beta_1) * grad
        self.acm_grad = self.beta_2 * self.acm_grad + (1 - self.beta_2) * grad ** 2
        
        m_hat = self.momentum / (1 - self.beta_1**self.t)
        v_hat = self.acm_grad / (1 - self.beta_2**self.t)
        
        if self.amsgrad:
            self.v_hat_max = np.maximum(self.v_hat_max, v_hat)
            v_hat = self.v_hat_max
        
        self.data -= (self.lr * m_hat) / (np.sqrt(v_hat) + self.eps)
        return self.data

class AdamW(Adam):
    
    def __init__(
        self,
        data: np.ndarray,
        lr: int | float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ):
        super().__init__(data, lr, beta_1, beta_2, eps, weight_decay=0.0, amsgrad=amsgrad)
        self.decoupled_weight_decay = self._check_weight_decay(weight_decay)
    
    def _repr_params(self) -> dict:
        params = super()._repr_params()
        params["weight_decay"] = self.decoupled_weight_decay
        return params
    
    def update(self, grad: np.ndarray) -> np.ndarray:
        if self.decoupled_weight_decay != 0:
            self.data -= self.lr * self.decoupled_weight_decay * self.data
        return super().update(grad)