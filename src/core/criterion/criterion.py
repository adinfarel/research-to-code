'''
Build Criterion BCE, CE, and MSE
'''

from abc import abstractmethod, ABC
import numpy as np

class Criterion(ABC):
    CACHE = None
    
    @abstractmethod
    def __call__(self, logits: np.ndarray, labels: np.ndarray, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def backward(self):
        raise NotImplementedError
    
    def _check_cache(self):
        assert self.CACHE is not None, (
            f"call {self.__class__.__name__}(logits, labels) first before backward"
        )

class MeanSquaredError(Criterion):
 
    def __call__(self, logits: np.ndarray, labels: np.ndarray, root: bool = False):
        # root?
        loss = np.mean((logits - labels)**2)
        self.CACHE = (root, logits, labels, loss)
 
        if root:
            return np.sqrt(loss)
 
        return loss

    def backward(self):
        self._check_cache()
        root, logits, labels, mse = self.CACHE
        N = logits.size
 
        grad = 2.0 * (logits - labels) / N
 
        if root:
            rmse = np.sqrt(mse)
 
            if rmse == 0:
                # RMSE gradient is undefined at exactly zero error.
                # return zero is a practical convention
                return np.zeros_like(logits)
 
            grad = grad / (2.0 * rmse)
 
        return grad

class CrossEntropy(Criterion):
 
    def _log_softmax(self, logits: np.ndarray, axis: int = -1):
        stable_logits = np.max(logits, axis=axis, keepdims=True)  # (N, 1)
        logits = logits - stable_logits  # prevent overflow value while get into exp ops
        counts = np.exp(logits)  # get virtual count each value
        logsumexp = np.log(counts.sum(axis=axis, keepdims=True))  # normalize onto range 0 - 1
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
        self.CACHE = (np.exp(log_probs), labels)  # save cache for backward
        return negative_log_likelihood
 
    def __call__(self, logits: np.ndarray, labels: np.ndarray, axis: int = -1):
        return self._nll(logits=logits, labels=labels, axis=axis)
 
    def backward(self):
        self._check_cache()
 
        probs, labels = self.CACHE
        N = probs.shape[0]
 
        grad = probs.copy()
        grad[np.arange(N), labels] -= 1.0
        grad /= N
 
        return grad
 
 
class BCEWithLogits(Criterion):
 
    def _sigmoid(self, logits: np.ndarray):
        positive = logits >= 0
        probs = np.empty_like(logits, dtype=np.float64)
        probs[positive] = 1 / (1 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probs[~positive] = exp_logits / (1 + exp_logits)
        return probs
 
    def __call__(self, logits: np.ndarray, labels: np.ndarray, eps: float = 1e-12):
 
        assert logits.shape == labels.shape, "shape labels and logits must be match"
 
        probs = self._sigmoid(logits)
        probs = np.clip(probs, eps, 1 - eps)
        self.CACHE = (probs, labels)
 
        loss = -np.mean((labels * np.log(probs)) + ((1 - labels) * np.log(1 - probs)))
        return loss
 
    def backward(self):
        self._check_cache()
        probs, labels = self.CACHE
        grads = probs - labels
 
        grads /= probs.size
 
        return grads