'''
Build atom implementations of Binary Cross Entropy (BCE)
'''

import numpy as np

class BCEWithLogits:
    
    CACHE = None # for save logits and labels
    
    def _sigmoid(self, logits: np.ndarray):
        probs = 1 / (1 + np.exp(-logits))
        return probs
    
    def __call__(self, logits: np.ndarray, labels: np.ndarray, eps: float = 1e-12):
        
        assert logits.shape == labels.shape, "shape labels and logits must be match"
        
        
        probs = self._sigmoid(logits)
        probs = np.clip(probs, eps, 1 - eps)
        self.CACHE = (probs, labels)
        
        loss = -np.mean((labels * np.log(probs)) + ((1 - labels) * np.log(1 - probs)))
        return loss
    
    def backward(self):
        assert self.CACHE is not None, "call BinaryCrossEntropy(logits, labels) first before backward"
        probs, labels = self.CACHE
        grads = probs - labels
        
        grads /= probs.size
        
        return grads