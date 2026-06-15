'''
Build MHA atom level

NOTE: this is anti-mainstream implementation, cause i am dealing with 4D hell no

Exact same with intuition from 'scaled_dot_product.py __docs__'
but the main is not matrix again but tensor

Here some explanation:
    -> each tensor contain 4D, consist is (Batch, Head, Seq_Len, Embed)
        Exam:
            Q @ K.T
            --> (B, H, T, C) @ (B, H, C, T)
            --> (B, H, T, T)
            
            Wei @ V
            --> (B, H, T, T) @ (B, H, T, C)
            --> (B, H, T, C)
    
    that's the whole main idea behind this, not scary at all if we understand under the hood

NOTE: if there mistake from intuition or implementation, just call me >.<
'''

import numpy as np

class MultiHeadAttn:
    
    def __init__(self, embed_dim: int, n_head: int):
        
        assert embed_dim % n_head == 0, f"embed_dim and n_head must be divisble"
        
        self.emb_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head # cause we checked through assert
        
        self.query = np.random.randn(embed_dim, embed_dim)
        self.key = np.random.randn(embed_dim, embed_dim)
        self.value = np.random.randn(embed_dim, embed_dim)
        self.proj = np.random.randn(embed_dim, embed_dim)

    def __call__(self, X: np.ndarray, causal_mask: bool = True):
        B, T, C = X.shape
        
        Q = X @ self.query
        K = X @ self.key
        V = X @ self.value
        
        # Reshape 3D to 4D
        # --> (B, T, C) -> (B, T, n_h, h_s)
        #
        # NOTE: some intuition below ^_^
        # in attention we dealing with many head, why?
        # cause each head can collect diff information
        # each head can be ask how good A, how good B
        # and each head have head_size that representation each token
        # after matmul operations, merge again each head to C
        Q = Q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3) # (B, T, n_h, h_s) -> (B, n_h, T, h_s)
        K = K.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        
        K_T = K.transpose(0, 1, 3, 2) # (B, n_h, T, h_s) -> (B, n_h, h_s, T), so that can matmul with Q
        
        affinity = self._four_dimension_matmul(Q, K_T)
        
        if causal_mask:
            masking = np.triu(np.ones((T, T), dtype=bool), k=1)
            affinity = np.where(masking == 1, float('-inf'), affinity)
        
        weight = self._softmax(affinity, axis=-1)
        
        # TODO: should've apply dropout after softmax, for simplification im not apply that
        # maybe in Flash Attention will apply
        
        out = self._four_dimension_matmul(weight, V)
        
        # Back again into shape formerly
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return out @ self.proj  # --> keep return shape corresponding first shape, and for keep residual stream so that can re-write into that
    
    def _softmax(self, tensor: np.ndarray, axis: int = -1):
        shifted = tensor - np.max(tensor, axis=axis, keepdims=True)
        counts = np.exp(shifted)
        counts /= np.sum(counts, axis=axis, keepdims=True)
        return counts
    
    def _scaled_dot_product(self, tensor1: np.ndarray, tensor2: np.ndarray):
        if tensor1.shape[1] != tensor2.shape[0]:
            raise ValueError(
                f"dimension {tensor1.shape} can not matrix-multiplication with dimension {tensor2.shape}"
            )
        
        m = tensor1.shape[0]
        n = tensor2.shape[1]
        k = tensor1.shape[1]
        
        results = np.zeros(shape=(m, n))
        
        for i in range(m):
            for j in range(n):
                res = 0.0
                
                for t in range(k):
                    res += tensor1[i][t] * tensor2[t][j]
                
                results[i][j] = res
        
        return results
    
    def _four_dimension_matmul(self, tensor1: np.ndarray, tensor2: np.ndarray):
        if tensor1.shape[0] != tensor2.shape[0]:
            raise ValueError(
                f"batch mismatch {tensor1.shape[0]} != {tensor2.shape[0]}"
            )
        
        if tensor1.shape[1] != tensor2.shape[1]:
            raise ValueError(
                f"heads mismatch {tensor1.shape[1]} != {tensor2.shape[1]}"
            )
        
        batch = tensor1.shape[0]
        heads = tensor1.shape[1]
        
        out_shape = (batch, heads, tensor1.shape[-2], tensor2.shape[-1])
        results = np.zeros(shape=(out_shape))
        
        for b in range(batch):
            for h in range(heads):
                tens1 = tensor1[b][h]
                tens2 = tensor2[b][h]
                results[b][h] = self._scaled_dot_product(tens1, tens2)
        
        return results