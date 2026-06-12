'''
Build MSE atom implementations
'''

import numpy as np

class MeanSquaredError:
    
    CACHE = None
    
    def __call__(self, logits: np.ndarray, labels: np.ndarray, root: bool = False):
        # root?
        loss = np.mean((logits - labels)**2)
        self.CACHE = (root, logits, labels, loss)
        
        if root:
            return np.sqrt(loss)
        
        return loss
    
    def backward(self):
        assert self.CACHE is not None, 'call MeanSquaredError(logits, labels) first before backward'
        root, logits, labels, mse = self.CACHE
        N = logits.size
        
        grad = 2.0 * (logits - labels) / N
        
        if root:
            rmse = np.sqrt(mse)
            
            if rmse == 0:
                # RMSE gradient is undefined at exactly zero error.
                # Return zero is a practical convention
                return np.zeros_like(logits)

            grad = grad / (2.0 * rmse)
        
        return grad