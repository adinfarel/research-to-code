'''
Build GPTQ (Generalized Post-Training Quantization)

INTUITION:
    GPTQ doing quantization column by column at weight matrix
    if we dealing with quantization we also dealing with rounding error
    
    error make drops accuracy model, if this error we propagate to next layer
    then the error can be accumulate and make model getting worse if error become larga
    Intuively, error at linear layer 1 can accumulate with error at layer 2
    for example:
    X_error = X @ W_quant (Layer One)
    X_error_2 = X_error @ W_quant (Layer Two)
    and so on..., this is what i mean earlier, error accumulate from naive quant make model accuracy drop
    
    GPTQ have approach instead of let it error affect next layer we want this error can be distributed
    because GPTQ work in column level (usually Group-Wise), error that we get in column 0 we can distributed
    as a correction for next column, and so on
    
    but why GPTQ know how exactly correction for each column just see distributed error from prev column?
    GPTQ use approach Hessian, which mean GPTQ see how much loss increase when we get this error so instead of
    minimial error we prefer to check increas loss when we get error
    
    but why use Hessian? generally we know in Taylor series
    First-order it is Gradient which mean Slope (Linear)
    Second-order it is Hessian which mean Curveture (Quadratic)
    
    so we know GPTQ is method after we pretrain model and we know cause we dealing with pre-train model
    we assume that loss already at minimum, so use gradient in this case can give we more information
    Hessian how faster loss increase if we change a bit weight, if the minimum is flat curve then changing weight
    a bit not affect the loss, but if we dealing with minimum curve, change weight a bit can affect the loss
    loss become large 
'''

import numpy as np

class GPTQ:
    
    def __init__(self, bits: int = 4):
        self.bits = bits
        
    def fit_and_quantize(self, W: np.ndarray, X: np.ndarray):
        C_out, C_in = W.shape
        assert C_out == C_in
        C = C_in
        
        W_updated = W.copy().astype(np.float32)
        
        # GPTQ not really measure real hessian
        # GPTQ approximate loss towards weight
        # so, H = (2 / N) * (X_flat.T @ X_flat)
        X_flat = X.reshape(-1, C)
        
        N = X_flat.shape[0]
        H = (2.0 / N) * np.dot(X_flat.T, X_flat) + np.eye(C) * 1e-4
        
        H_inv = np.linalg.inv(H)
        
        qmax = 2 ** (self.bits - 1) - 1
        qmin = -qmax
        
        for i in range(C):
            W_original = W_updated[:, i]
            
            alpha = np.max(np.abs(W_original))
            scale = alpha / qmax if alpha > 0 else 1.0
            
            W_column_q = scale * np.round(W_original / scale) # Quantized and Dequant
            W_column_q = np.clip(W_column_q, qmin * scale, qmax * scale)
            
            error = W_original - W_column_q
            
            W_updated[:, i] = W_column_q
            
            if i < C - 1:
                h_inv_row = H_inv[i, i+1:]
                h_inv_self = H_inv[i, i]

                # if h_inv_self != 0:
                #     direction = h_inv_row / h_inv_self
                # else:
                #     direction = h_inv_row
                    
                direction = h_inv_row / h_self_adjustment if (h_self_adjustment := h_inv_self) != 0 else h_inv_row
                
                W_updated[:, i+1:] -= np.outer(error, direction)
            
        return W_updated