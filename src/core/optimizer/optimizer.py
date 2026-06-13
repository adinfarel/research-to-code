'''
Build atom implemetations of Adam, RMSProp, SGD, AdamW

NOTE: here some intuition
    # if get +grad it means direct of grad is decreasing, must -weight loss decrease
    # if get -grad it means direct of grad is increasing, must +weight loss decrease
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
