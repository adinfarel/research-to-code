'''
Build atom implementations of cross entropy (Softmax + NLL)
'''

import numpy as np

class CrossEntropy:
    
    CACHE = None
    
    def _log_softmax(self, logits: np.ndarray, axis: int = -1):
        stable_logits = np.max(logits, axis=axis, keepdims=True) # (N, 1)
        logits = logits - stable_logits # prevent overflow value while get into exp ops
        counts = np.exp(logits) # get virtual count each value
        logsumexp = np.log(counts.sum(axis=axis, keepdims=True)) # normalize onto range 0 - 1
        return logits - logsumexp

    def _softmax(self, logits: np.ndarray, axis: int = -1):
        log_probs = self._log_softmax(logits, axis)
        return np.exp(log_probs)

    def _nll(self, logits: np.ndarray, labels: np.ndarray, axis: int = -1):
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(f"logits and labels shape mismatch, got logits {logits.shape[0]} != {labels.shape[0]}")
        
        log_probs = self._log_softmax(logits=logits, axis=axis)
        num_inputs = labels.shape[0]
        negative_log_likelihood = -np.mean(log_probs[np.arange(num_inputs), labels])
        self.CACHE = (np.exp(log_probs), labels) # save cache for backward
        return negative_log_likelihood
    
    def __call__(self, logits: np.ndarray, labels: np.ndarray, axis: int = -1):
        return self._nll(logits=logits, labels=labels, axis=axis)
    
    def backward(self):
        assert self.CACHE is not None, "call CrossEntropy(logits, labels) first before backward."
        
        probs, labels = self.CACHE
        N = probs.shape[0]
        
        grad = probs.copy()
        grad[np.arange(N), labels] -= 1.0
        grad /= N
        
        return grad