'''
Build function that can backward in a way linear
'''

import numpy as np

def linear_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray):
    assert x.shape[1] == w.shape[0]
    
    # x -> (N x M)
    # w -> (M x D)
    # b -> (D,)
    xw = x.dot(w)
    z  = xw + b # (N x D)
    cache = (x, w, b)
    
    return z, cache

def linear_backward(dout: np.ndarray, cache: tuple[np.ndarray, np.ndarray, np.ndarray]):
    x, w, b = cache
    
    # shape dx must be matchup with x
    # dx -> (N x M) == x -> (N x M)
    #
    # shape dw must be matchup with w
    # dw -> (M x D) == w -> (M x D)
    #
    # shape db must be matchup with b
    # db -> (D,) == b -> (D,)
    #
    # dout shape == z shape (N x D)
    #
    # cache to save value and shape input x, w, b
    
    # for formula gradient matrix-multiplication is similiar to scalar-multiplication
    # it is swap multiplier x * w -> dx = w -> dw = x
    dx = dout.dot(w.T) # -> dout(N x D) @ w.T(D x M) = dx(N x M) == x(N x M)
    
    dw = x.T.dot(dout) # -> x.T(M x N) @ dout(N x D) = dw(M x D) == w(M x D)
    
    db = np.sum(dout, axis=0) # why axis=0? cause effect of broadcasting b(D,) replicate as many as dimension xw (output)
    # b(D,) broadcast to b(N x D), replicate D == N times
    # so that's way we sum axis=0 to squash dimension of O which is N to 1 if keepdim=True
    
    return dx, dw, db

def relu_forward(z: np.ndarray):
    # keep value of z if z is positive else 0
    cache = z
    z = np.maximum(z, 0.0)
    
    return z, cache

def relu_backward(dout: np.ndarray, cache: np.ndarray):
    # cause we use max operations value z that positive mapping to 1 and value < 0 mapping 0
    # not weird, cause relu function disable value < 0 without tolerance like GELU, SILU, etc
    z = cache
    dz = np.where(z > 0, 1, 0)
    
    dz = dout * dz
    return dz