'''
Implement causal mask (copy code at transformers/attention/scaled_dot_product.py)
'''

import numpy as np

def _softmax(matrix: np.ndarray, axis: int = -1):
    shifted = matrix - np.max(matrix, axis=-1, keepdims=True)
    counts = np.exp(shifted)
    counts = counts / np.sum(counts, axis=-1, keepdims=True)
    return counts

def _matrix_multiplication(mat1: np.ndarray, mat2: np.ndarray):
    if mat1.shape[1] != mat2.shape[0]:
        raise ValueError(
            f"shape mat1 and mat2 is mismatch"
        )
    
    m = mat1.shape[0]
    n = mat2.shape[1]
    
    results = np.zeros(shape=(m, n))
    
    for i in range(m):
        for j in range(n):
            res = 0.0
            
            for t in range(mat1.shape[1]):
                # print(f"{mat1[i][t]} * {mat2[t][j]}")
                res += mat1[i][t] * mat2[t][j]
                
            results[i][j] = res
    
    return results

def _causal_mask(matrix: np.ndarray):
    token_length = matrix.shape[-1]
    triu = np.triu(np.ones((token_length, token_length), dtype=bool), k=1)
    mask = np.where(triu == 1, float('-inf'), matrix)
    return mask

def scaled_dot_product(Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal_mask: bool = False):
    affinity = _matrix_multiplication(mat1=Q, mat2=K.T)
    affinity = affinity / np.sqrt(K.shape[-1])
    
    if causal_mask:
        affinity = _causal_mask(affinity)
    
    weight = _softmax(affinity, axis=-1)
    
    out = _matrix_multiplication(mat1=weight, mat2=V)
    
    return out

if __name__ == "__main__":
    mat1 = np.random.randn(3, 3)
    mat2 = np.random.randn(3, 3)
    dummy = _matrix_multiplication(mat1, mat2.T)
    
    mask = _causal_mask(dummy)
    
    softmax = _softmax(mask)
    
    print(f"Tringular Upper Masking:\n{mask}")
    print(f"Effect of softmax:\n{softmax}")