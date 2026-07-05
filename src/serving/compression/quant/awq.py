'''
Build AWQ implementation
'''

import numpy as np

from src.serving.compression.quant.quantization import Quantization

class AWQuant:
    def __init__(self, group_size: int = 128, bits: int = 4):
        r'''
        INTUITION:
        AWQ (Activation-aware quantization) discover that accuracy LLM-int4
        determine by 1% salient weights (Weight Important)
        
        Salient weight that's not mean weight which have large MAGNITUDE but rather salient weight
        paired with Input activation that have large magnitude, which is if Error quantize large for
        this weight can affect results in the end because, Input activation acting as loudspeaker
        (Error Amplication) that messed up output layer.
        
        SO, instead of doing normal quant, or Mixed-Precision between weight which salient weight keep
        in high precision and left of their weight in low precision but hardware GPU can compute diff type data
        in one tensor, so based on paper their discover that we need new paramater which call Scaling Factor
        The idea is, we multiply weight with scale to prevent weight get rounding error and divide x with
        scale to get small magnitude but in the math this is exact with
        Y = XW equal to Y = (X / s) x (w x s) because variabel s can cancel-out
        '''
        self.group_size = group_size
        self.bits = bits
        self.quantizer = Quantization()
    
    def _quantize_weight_groupwise(self, W: np.ndarray, bits: int, group_size: int):
        '''
        Formula:
        Delta   = max(abs(x)) / (2^(b-1)-1)
        Q(w)    = Delta * Round(w / Delta)
        '''
        out_features, in_features = W.shape # [B, T]
        num_groups = in_features // group_size
        
        W_reshaped = W.reshape(out_features, num_groups, group_size)
        
        qmax = (2 ** (bits - 1) - 1)
        qmin = -qmax
        
        delta = np.max(np.abs(W_reshaped), axis=-1, keepdims=True) / qmax
        delta = np.where(delta == 0, 1e-8, delta)
        
        W_q = delta * np.round(W_reshaped / delta)
        W_q = np.clip(W_q, qmin * delta, qmax * delta)
        
        return W_q.reshape(out_features, in_features)
    
    def fit_and_quantize(self, W: np.ndarray, X: np.ndarray):
        out_features, in_features = W.shape
        
        S_X = np.max(np.abs(X), axis=(0, 1))
        
        Y_true = np.matmul(X, W.T)
        
        best_loss = float("inf")
        best_alpha = 0.0
        
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for alpha in alphas:
            s = np.power(S_X, alpha)
            s = np.where(s == 0, 1e-8, s)
            
            W_scaled = W * s
            X_scaled = X / s
            
            W_quant_sim = self._quantize_weight_groupwise(W_scaled, self.bits, self.group_size)
            Y_simulated = np.matmul(X_scaled, W_quant_sim.T)
            
            loss = np.mean((Y_simulated - Y_true) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha
        
        best_s = np.power(S_X, best_alpha)
        best_s = np.where(best_s == 0, 1e-8, best_s)
        
        W_final_scaled = W * best_s
        W_final_quantized = self._quantize_weight_groupwise(W_final_scaled, self.bits, self.group_size)
        
        W_awq_processed = W_final_quantized / best_s
        
        return W_awq_processed, best_s, best_alpha