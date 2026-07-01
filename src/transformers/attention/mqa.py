'''
Build MultiQueryAttention implementation 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    
    def __init__(self, emb_dim, n_head):
        super().__init__()
        assert emb_dim % n_head == 0
        
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.head_size = emb_dim // n_head
        self.softmax_scale = 1.0 / (self.head_size ** 0.5)
        
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, self.head_size)
        self.value = nn.Linear(emb_dim, self.head_size)
    
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(B, T, self.n_head, self.head_size).transpose(1, 2) 
        K = K.view(B, T, 1, self.head_size).transpose(1, 2) 
        V = V.view(B, T, 1, self.head_size).transpose(1, 2) 

        K_rep = K.repeat_interleave(self.n_head, dim=1).permute(0, 1, 3, 2)
        V_rep = V.repeat_interleave(self.n_head, dim=1)
        
        QK = (Q @ K_rep) * self.softmax_scale
        # In real world apply causal mask before softmax
        affinity = F.softmax(QK, dim=-1)
        # Apply dropout after softmax
        
        out = affinity @ V_rep
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        # Apply projection before return out (to learned mixing information head and keep out same as shape x before and matchy with x for resnet)
        return out