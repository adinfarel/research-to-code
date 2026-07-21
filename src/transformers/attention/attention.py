'''
Build Attention implementation
'''

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

def _matmul_2d(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    if mat1.shape[-1] != mat2.shape[0]:
                raise ValueError(
            f"shape mismatch: {mat1.shape} tidak bisa matmul dengan {mat2.shape}"
        )
    
    m, k = mat1.shape
    _, n = mat2.shape
    out = np.zeros((m, n), dtype=np.float64)
    
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for t in range(k):
                acc += mat1[i][t] * mat2[t][j]
            out[i][j] = acc
    return out

def _matmul_4d(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
    if tensor1.shape[0] != tensor2.shape[0]:
        raise ValueError(f"batch mismatch {tensor1.shape[0]} != {tensor2.shape[0]}")
    if tensor1.shape[1] != tensor2.shape[1]:
        raise ValueError(f"heads mismatch {tensor1.shape[1]} != {tensor2.shape[1]}")
    
    B, H = tensor1.shape[0], tensor1.shape[1]
    out_shape = (B, H, tensor1.shape[-2], tensor2.shape[-1])
    out = np.zeros(out_shape, dtype=np.float64)
    
    for b in range(B):
        for h in range(H):
            out[b][h] = _matmul_2d(mat1=tensor1[b][h], mat2=tensor2[b][h])
    return out

def _softmax(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = matrix - np.max(matrix, axis=axis, keepdims=True)
    counts  = np.exp(shifted) # stable softmax prevent overflow NaN from exponential
    return counts / np.sum(counts, axis=axis, keepdims=True)

class BaseAttention(ABC):
    
    def __init__(self, embed_dim: int, n_head: int):
        assert embed_dim % n_head == 0, "embed_dim must be divisible with n_head"
        self.emb_dim = embed_dim
        self.n_heads = n_head
        self.head_dim = self.emb_dim // n_head
        self.softmax_scale = 1 / self.head_dim ** 0.5
        
        self._init_projections()
        
        self.k_cache: Optional[np.ndarray] = None
        self.v_cache: Optional[np.ndarray] = None
    
    @abstractmethod
    def _init_projections(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _project_qkv(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def _expand_kv(self, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return K, V
    
    def _update_kv_cache(self, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.k_cache is None:
            self.k_cache, self.v_cache = K, V
        else:
            self.k_cache = np.concatenate([self.k_cache, K], axis=2)
            self.v_cache = np.concatenate([self.v_cache, V], axis=2) #type: ignore
        return self.k_cache, self.v_cache #type: ignore
    
    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
    
    def _apply_causal_mask(self, affinity: np.ndarray, T_q: int, T_kv: int) -> np.ndarray:
        offset = T_kv - T_q
        
        q_positions = np.arange(T_q).reshape(-1, 1) + offset
        kv_positions = np.arange(T_kv).reshape(1, -1)
        mask = kv_positions > q_positions
        
        return np.where(mask[None, None, :, :], float("-inf"), affinity)
    
    def _scaled_dot_product(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        causal_mask: bool
    ):
        T_q = Q.shape[2]
        T_kv = K.shape[2]
        
        K_T = K.transpose(0, 1, 3, 2)
        affinity = _matmul_4d(Q, K_T)
        affinity = affinity * self.softmax_scale
        
        if causal_mask:
            affinity = self._apply_causal_mask(affinity, T_q, T_kv)
        
        weight = _softmax(affinity, axis=-1)
        
        out = _matmul_4d(weight, V)
        return out
    
    def __call__(
        self,
        X: np.ndarray,
        causal_mask: bool = True,
        use_cache: bool = False,
    ) -> np.ndarray:
        B, T, C = X.shape
        
        Q, K, V = self._project_qkv(X)
        
        if use_cache:
            K, V = self._update_kv_cache(K, V)
        
        K, V = self._expand_kv(K, V)
        
        out = self._scaled_dot_product(
            Q,
            K,
            V,
            causal_mask
        )
        
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        out_proj = np.zeros((B, T, C))

        for b in range(B):
            out_proj[b] = _matmul_2d(out[b], self.proj) #type: ignore

        return out_proj

class MultiHeadAttention(BaseAttention):
    
    def _init_projections(self) -> None:
        self.query = np.random.randn(self.emb_dim, self.emb_dim)
        self.key = np.random.randn(self.emb_dim, self.emb_dim)
        self.value = np.random.randn(self.emb_dim, self.emb_dim)
        self.proj = np.random.randn(self.emb_dim, self.emb_dim)
    
    def _project_qkv(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B, T, C = X.shape
        
        Q, K, V = np.zeros((B, T, C)), np.zeros((B, T, C)), np.zeros((B, T, C))
        for b in range(B):
            Q[b] = _matmul_2d(X[b], self.query)
            K[b] = _matmul_2d(X[b], self.key)
            V[b] = _matmul_2d(X[b], self.value)
    
        Q = Q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        return Q, K, V

class MultiQueryAttention(BaseAttention):
    
    def _init_projections(self) -> None:
        self.query = np.random.randn(self.emb_dim, self.emb_dim)
        self.key = np.random.randn(self.emb_dim, self.head_dim)
        self.value = np.random.randn(self.emb_dim, self.head_dim)
        self.proj = np.random.randn(self.emb_dim, self.emb_dim)
    
    def _project_qkv(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B, T, C = X.shape
        
        Q, K, V = np.zeros((B, T, C)), np.zeros((B, T, self.head_dim)), np.zeros((B, T, self.head_dim))
        for b in range(B):
            Q[b] = _matmul_2d(X[b], self.query)
            K[b] = _matmul_2d(X[b], self.key)
            V[b] = _matmul_2d(X[b], self.value)
        
        Q = Q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, 1, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, 1, self.head_dim).transpose(0, 2, 1, 3)
        
        return Q, K, V
    
    def _expand_kv(self, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        B, heads, T, head_dim = K.shape
        
        K = np.broadcast_to(K, shape=(B, self.n_heads * heads, T, head_dim))
        V = np.broadcast_to(V, shape=(B, self.n_heads * heads, T, head_dim))
        
        return K, V

class GroupQueryAttention(BaseAttention):
    
    def __init__(self, embed_dim: int, n_head: int, n_kv_head: int):
        assert n_head % n_kv_head == 0, "n_head must be divisible with n_kv_head"
        
        self.n_kv_head = n_kv_head
        self.num_queries_each_kv_head = n_head // n_kv_head
        
        super().__init__(embed_dim=embed_dim, n_head=n_head)
    
    def _init_projections(self) -> None:
        self.query = np.random.randn(self.emb_dim, self.emb_dim)
        self.key = np.random.randn(self.emb_dim, self.n_kv_head * self.head_dim)
        self.value = np.random.randn(self.emb_dim, self.n_kv_head * self.head_dim)
        self.proj = np.random.randn(self.emb_dim, self.emb_dim)
    
    def _project_qkv(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B, T, C = X.shape
        
        Q, K, V = np.zeros((B, T, C)), \
            np.zeros((B, T, self.n_kv_head * self.head_dim)), \
                np.zeros((B, T, self.n_kv_head * self.head_dim))
                
        for b in range(B):
            Q[b] = _matmul_2d(X[b], self.query)
            K[b] = _matmul_2d(X[b], self.key)
            V[b] = _matmul_2d(X[b], self.value)
        
        Q = Q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)

        return Q, K, V

    def _expand_kv(self, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        K = K.repeat(self.num_queries_each_kv_head, axis=1)
        V = V.repeat(self.num_queries_each_kv_head, axis=1)
        
        return K, V

class MultiLatentAttention(BaseAttention):
    
    def __init__(self, embed_dim: int, n_head: int, 
                 q_latent_size: int = 1536, kv_latent_size: int = 512):
        self.q_latent_size = q_latent_size
        self.kv_latent_size = kv_latent_size
        
        super().__init__(embed_dim=embed_dim, n_head=n_head)
        
    def _init_projections(self) -> None:
        self.q_down_proj = np.random.randn(self.emb_dim, self.q_latent_size)
        self.q_up_proj = np.random.randn(self.q_latent_size, self.emb_dim)
        
        self.kv_down_proj = np.random.randn(self.emb_dim, self.kv_latent_size)
        self.k_up_proj = np.random.randn(self.kv_latent_size, self.emb_dim)
        self.v_up_proj = np.random.randn(self.kv_latent_size, self.emb_dim)
        
        self.proj = np.random.randn(self.emb_dim, self.emb_dim)
    
    def _project_qkv(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B, T, C = X.shape
        
        Q_down, KV_latent = np.zeros((B, T, self.q_latent_size)), \
            np.zeros((B, T, self.kv_latent_size))
            
        for b in range(B):
            Q_down[b] = _matmul_2d(X[b], self.q_down_proj)
            KV_latent[b] = _matmul_2d(X[b], self.kv_down_proj)
        
        Q = np.zeros((B, T, self.emb_dim))
        for b in range(B):
            Q[b] = _matmul_2d(Q_down[b], self.q_up_proj)
        
        Q = Q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        KV_latent = KV_latent[:, None, :, :]
        
        return Q, KV_latent, KV_latent
    
    def _expand_kv(self, KV_latent: np.ndarray, _: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        B, _, T, latent_space = KV_latent.shape
        KV_latent = np.squeeze(KV_latent, axis=1)
        
        K, V = np.zeros((B, T, self.emb_dim)), \
            np.zeros((B, T, self.emb_dim))
        
        for b in range(B):
            K[b] = _matmul_2d(KV_latent[b], self.k_up_proj)
            V[b] = _matmul_2d(KV_latent[b], self.v_up_proj)
        
        K = K.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        return K, V