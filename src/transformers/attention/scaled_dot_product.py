'''
Build scaled dot product attention atom implementations

Here an intuition:
    Attention mechanism is method which token can see each other, consists 3 matrix:
    Q : what am i looking for?
    K : what information do i contain?
    V : what information should i send if selected, like "if u want me, then here i am"

Mechanism:
    matrix Q matrix-multiplication with matrix K and divide by square root dimension matrix Q/K
     --> Q @ K / sqrt(d_k)
        Q -> Matrix Query
        K -> Matrix Key
        d_k -> dimension of this attention (usually head_dim)
    
    then result of Q@K masking with causal mask if dealing with Decoder
     --> Causal Mask: Token dont see future (token in next index just see token prev curr token)
    
    softmax to get probability each token, if res dot-product between Qi * Ki large, 
    then probs them will be large, vice versa...
    
    then last step weight (Q@K) @ V to get real value each token

NOTE:
if there is misinformation or miss-intuition, dont hesitate to tell me >.<, i'm open minded
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
    
    # Q = [
    #     [1.0, 2.0],
    #     [3.0, 4.0],
    # ]
    #
    # K.T? = [
    #     [1.0, 3.0],
    #     [2.0, 4.0],
    # ]
    #
    # Q @ K
    # Q1 * K1 = [1.0, 2.0] * [1.0, 2.0]
    # Q1 * K2 = [1.0, 2.0] * [3.0, 4.0]
    # Q2 * K1 = [3.0, 4.0] * [1.0, 2.0]
    # Q2 * K2 = [3.0, 4.0] * [3.0, 4.0]
    
    results = np.zeros(shape=(m, n))
    
    for i in range(m):
        for j in range(n):
            res = 0.0
            
            for t in range(mat1.shape[1]):
                # print(f"{mat1[i][t]} * {mat2[t][j]}")
                res += mat1[i][t] * mat2[t][j]
                
            results[i][j] = res
    
    return results


def scaled_dot_product(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    
    # Q @ K
    # Exam:
    #   -> (2, 3) @ (3, 2).T = (2, 2)
    affinity = _matrix_multiplication(mat1=Q, mat2=K.T)
    affinity = affinity / np.sqrt(K.shape[-1]) # usually last shape are dimension of head
    
    # TODO: real practice apply causal mask
    # triu = np.triu(token_length, token_length)
    # Exam:
    #   -> [
    #     [False, True, True],
    #     [False, False, True],
    #     [False, False, False]
    #    ]
    # affinity.masked_fill(triu, float('-inf'))

    weight = _softmax(affinity, axis=-1)
    
    # TODO: real practice apply dropout before matmul with V
    out = _matrix_multiplication(mat1=weight, mat2=V)
    
    return out
    
if __name__ == "__main__":
    Q = np.array([
        [1.0, 3.0],
        [4.0, 7.0],
    ])
    
    K = np.array([
        [2.0, 5.0],
        [6.0, 3.0],
    ])
    
    res = _matrix_multiplication(Q, K.T)
    print("\nResult :\n", res)