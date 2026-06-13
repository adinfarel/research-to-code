'''
Build atom implemetations of Adam, RMSProp, SGD, AdamW

NOTE: here some intuition
    # if get +grad it means direction of grad is decreasing, must -weight loss decrease
    # if get -grad it means direction of grad is increasing, must +weight loss decrease
    # so, changing x towards opposite of sign grad will decrease loss to minimum loss
    # i hope yall understand intuition from me >.<
'''

import numpy as np

class SGD:
    
    def __init__(self, data: np.ndarray, lr: float | int):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if not isinstance(lr, (int, float)):
            raise TypeError(f"lr must be float, got {lr}")

        if lr <= 0:
            raise ValueError("lr must be positive")
            
        self.data = np.asarray(data, dtype=np.float64)
        self.lr = float(lr)
        
    def update(self, grad: np.ndarray):
        assert grad.shape == self.data.shape, "grad shape and data shape mismatch"
        
        self.data += -self.lr * grad # moving data (weight or parameter) opposite to grad so change weight towards opposite grad can decrease loss
        return self.data

class RMSProp:
    
    def __init__(self, data: np.ndarray, lr: float | int, eps: float = 1e-8, beta: float = 0.9):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if not isinstance(lr, (int, float)):
            raise TypeError(f"lr must be float, got {lr}")

        if lr <= 0:
            raise ValueError("lr must be positive")
        
        if not 0 <= beta < 1:
            raise ValueError(f"beta must between 0 and 1, got {beta}")
        
        if eps <= 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        
        self.data = np.asarray(data, dtype=np.float64)
        self.lr = float(lr)
        self.eps = float(eps)
        self.beta = float(beta)
        self.acm_grad = np.zeros_like(self.data)
        
    def update(self, grad: np.ndarray):
        grad = np.asarray(grad, dtype=np.float64)
        
        if grad.shape != self.data.shape:
            raise ValueError(f"grad shape and data shape is mismatch")
        
        vt = (self.beta * self.acm_grad) + (1 - self.beta) * grad**2
        self.acm_grad = vt # just store acum new grad, not like AdaGrad that (+=) increase constantly from early
        
        self.data += - (self.lr / (np.sqrt(vt) + self.eps) * grad)
        return self.data

class Adam:
    
    def __init__(self, data: np.ndarray, lr: int | float, beta_1: float = 0.9, beta_2: float = 0.999, eps: float = 1e-8):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if not isinstance(lr, (int, float)):
            raise TypeError(f"lr must be float, got {lr}")

        if lr <= 0:
            raise ValueError("lr must be positive")
        
        if not 0 <= beta_1 < 1:
            raise ValueError(f"beta must between 0 and 1, got {beta_1}")
        
        if not 0 <= beta_2 < 1:
            raise ValueError(f"beta must between 0 and 1, got {beta_2}")
        
        if eps <= 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        
        self.data = np.asarray(data, dtype=np.float64)
        self.lr = float(lr)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.eps = float(eps)
        
        self.momentum = np.zeros_like(self.data) # m
        self.acm_grad = np.zeros_like(self.data) # v
        self.t = 0
        
    def update(self, grad: np.ndarray):
        grad = np.asarray(grad, dtype=np.float64)
        
        if grad.shape != self.data.shape:
            raise ValueError(
                f"grad shape and data shape is mismatch."
            )
        
        self.t += 1
        
        self.momentum = self.beta_1 * self.momentum + (1 - self.beta_1) * grad
        self.acm_grad = self.beta_2 * self.acm_grad + (1 - self.beta_2) * grad**2
        
        m_hat = self.momentum / (1 - self.beta_1**self.t)
        v_hat = self.acm_grad / (1 - self.beta_2**self.t)
        
        self.data -= (self.lr * m_hat) / (np.sqrt(v_hat) + self.eps)
        return self.data