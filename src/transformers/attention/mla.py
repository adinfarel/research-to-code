''''
Build MultiHead-Latent Attention implementation
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiLatentAttention(nn.Module):
    def __init__(self, emb_dim, n_head, head_size,
                 kv_lora_rank = 512, q_lora_rank = 1536):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.head_size = head_size
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        
        self.q_down_proj = nn.Linear(emb_dim, q_lora_rank, bias=False)
        self.q_up_proj = nn.Linear(q_lora_rank, emb_dim, bias=False)
        
        self.kv_down_proj = nn.Linear(emb_dim, self.kv_lora_rank, bias=False)
        
        self.k_up_proj = nn.Linear(kv_lora_rank, emb_dim, bias=False)
        self.v_up_proj = nn.Linear(kv_lora_rank, emb_dim, bias=False)
        
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.softmax_scale = 1.0 / (self.head_size ** 0.5)
        
    def forward(self, x: torch.Tensor, return_cache = False):
        B, T, _ = x.shape
        
        q_latent = self.q_down_proj(x)
        Q = self.q_up_proj(q_latent).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        kv_latent = self.kv_down_proj(x)
        
        K = self.k_up_proj(kv_latent).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        V = self.v_up_proj(kv_latent).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) * self.softmax_scale
        affinity = F.softmax(scores, dim=-1)
        
        out = affinity @ V
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size)
        output = self.out_proj(out) 
        
        if return_cache:
            return output, kv_latent
            
        return output

if __name__ == "__main__":
    B, T, C = 2, 512, 4096  
    model = MultiLatentAttention(emb_dim=C, n_head=32, head_size=128, kv_lora_rank=512, q_lora_rank=1536)
    
    x = torch.randn(B, T, C)
    out, kv_cache = model(x, return_cache=True)
    
    print("=== TRACING PURE MLA SHAPE ===")
    print("1. Input Tensor Shape      :", x.shape)
    print("2. Output Tensor Shape     :", out.shape)
    print("3. KV Cache VRAM:", kv_cache.shape)