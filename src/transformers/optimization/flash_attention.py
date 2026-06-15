'''
Build flash attention atom level

Here intuition:
 --> FA is the mechanism tiling attention which is can reduce memory access from O(Nd + N^2) to (N^2 d^2 M^-1) M is mean memory size of SRAM
 
 Naive Attention:
    --> Every calculate attention Q @ K until out
        the computer load and save result to HBM which this is time-consuming just because of memory
        
        Example:
            -> Step 1:
                Calculate Q @ K, computer must load parameter from HBM and save back into HBM
            -> Step 2:
                Softmax(affinity), computer load and save back into HBM
            
            See, just from 2 step computer has been loaded and saved parameter 4 times, this is memory-bound
        
        Solution:
            This is why FA exists to solve that problem
            
            The main idea: idea behind FA is because SRAM can contain small parameter, the trick is use tilling-mechanism
            instead of calculate full attention (dense) better calculate (Example) Qi @ Ki it's mean just one tile for each iterate
            so the computer can store result into SRAM instead of HBM
            NOTE: why not store full attention into SRAM? cause SRAM only can save small parameter because memory that SRAM have too small
                  for keep all parameter
            
            Limitations:
            How about softmax that need denominator all token length -> Solve: use online softmax with tracking Mi (running max) and Li (running sum)

NOTE: if there's mistake or misinformation from my intuition, dont hesitate to call me, that's it >.<
'''

import numpy as np

class MultiHeadAttn:
    
    def __init__(self, embed_dim: int, n_head: int):
            assert embed_dim % n_head == 0, "embed_dim and n_head must be divisible"
            
            self.emb_dim = embed_dim
            self.n_head = n_head
            self.head_dim = embed_dim // n_head
            
            self.query = np.random.randn(embed_dim, embed_dim) * 0.02
            self.key = np.random.randn(embed_dim, embed_dim) * 0.02
            self.value = np.random.randn(embed_dim, embed_dim) * 0.02
            self.proj = np.random.randn(embed_dim, embed_dim) * 0.02
    
    def _flash_attn_2d(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal_mask: bool = False, B_r: int = 2, B_c: int = 2):
        T, d = Q.shape
        
        # m: Running Max (prevent overflow)
        # l: Running Sum (denominator softmax)
        # O: Running Output (Accumulation Matrix)
        O = np.zeros_like(Q)
        m = np.full((T, 1), float('-inf'))
        l = np.zeros((T, 1))
        
        scale = 1.0 / np.sqrt(d)
        
        for j in range(0, T, B_c):
            K_j = K[j:j+B_c] # (Bc, d)
            V_j = V[j:j+B_c] # (Bc, d)
            
            for i in range(0, T, B_r):
                Q_i = Q[i:i+B_r] # (Br, d)
                
                S_ij = (Q_i @ K_j.T) * scale
                
                if causal_mask:
                    row_indices = np.arange(i, i + S_ij.shape[0])[:, None]
                    col_indices = np.arange(j, j + S_ij.shape[1])
                    masking = row_indices < col_indices
                    S_ij = np.where(masking, float('-inf'), S_ij)
                
                m_ij = np.max(S_ij, axis=-1, keepdims=True)
                P_ij = np.exp(S_ij - m_ij)
                l_ij = np.sum(P_ij, axis=-1, keepdims=True)
                
                m_old = m[i:i+B_r]
                l_old = l[i:i+B_r]
                
                m_new = np.maximum(m_old, m_ij)
                
                alpha = np.exp(m_old - m_new)
                beta = np.exp(m_ij - m_new)
                
                l_new = (alpha * l_old) + (beta + l_ij)
                
                O_old = O[i:i+B_r]
                safe_l_new = np.where(l_new == 0, 1.0, l_new)
                
                O[i:i+B_r] = ((O_old * l_old * alpha) * (P_ij @ V_j * beta)) / safe_l_new
                
                m[i:i+B_r] = m_new
                l[i:i+B_r] = l_new
        
        return O

    def __call__(self, X: np.ndarray, causal_mask: bool = True, use_flash: bool = True):
        B, T, C = X.shape
        
        Q_scaled = X @ self.query
        K_scaled = X @ self.key
        V_scaled = X @ self.value
        
        Q = Q_scaled.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        K = K_scaled.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        V = V_scaled.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        
        out_tensor = np.zeros_like(Q)
        
        if use_flash:
            for b in range(B):
                for h in range(self.n_head):
                    out_tensor[b][h] = self._flash_attn_2d(Q[b][h], K[b][h], V[b][h], causal_mask=causal_mask)
        
        else:
            K_T = K.transpose(0, 1, 3, 2)
            affinity = (Q @ K_T) / np.sqrt(self.head_dim)
            if causal_mask:
                masking = np.triu(np.ones((T, T), dtype=bool), k=1)
                affinity = np.where(masking, float('-inf'), affinity)
            weight = np.exp(affinity - np.max(affinity, axis=-1, keepdims=True))
            weight /= np.sum(weight, axis=-1, keepdims=True)
            out_tensor = weight @ V
        
        out = out_tensor.transpose(0, 2, 1, 3).reshape(B, T, C)
        return out @ self.proj