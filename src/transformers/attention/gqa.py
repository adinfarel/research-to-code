 
'''
Build GroupQueryAttention implementation

Intuition:
    This method exists to solve how kv-cache that store in VRAM compressed because memory grow
    linearly each new token attend
    
    So, instead of each one query attend one key, what if each group-query just see one token?
    then this method successfully prevented memory from continuously growing in VRAM
    
    MHA:
    Q: (32, 100), K: (32, 100)
    GQA:
    Q: (group, Q, 100), K: (seq_len/group_size <- repeat interleaved for each group, 100) 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, emb_dim, n_head, n_kv_head):
        super().__init__()
        
        assert emb_dim % n_head == 0
        assert n_head % n_kv_head == 0
        
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.head_size = emb_dim // n_head
        self.n_kv_head = n_kv_head
        self.num_queries_each_kv_head = n_head // n_kv_head
        self.softmax_scale = 1.0 / (self.head_size ** 0.5)
        
        print(f"Embed: {emb_dim}, N_head: {n_head}, HeadSize: {self.head_size}\nKV Head: {n_kv_head}, Query size for one KV: {self.num_queries_each_kv_head}")
        
        self.query = nn.Linear(emb_dim, n_head * self.head_size, bias=False)
        self.key = nn.Linear(emb_dim, n_kv_head * self.head_size, bias=False)
        self.value = nn.Linear(emb_dim, n_kv_head * self.head_size, bias=False)
    
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape 
        
        Q = self.query(x) 
        K = self.key(x) 
        V = self.value(x) 
        
        Q = Q.view(B, T, self.n_head, self.head_size).transpose(1, 2) 
        K = K.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2) 
        V = V.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2) 
        
        K_rep = K.repeat_interleave(self.num_queries_each_kv_head, dim=1).permute(0, 1, 3, 2)
        V_rep = V.repeat_interleave(self.num_queries_each_kv_head, dim=1)
        
        # return (Q, K, V, K_rep, V_rep)
        
        QK = (Q @ K_rep) * self.softmax_scale
        # In real world apply causal mask before softmax
        affinity = F.softmax(QK, dim=-1)
        # Apply dropout after softmax
        
        out = affinity @ V_rep
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        # Apply projection before return out (to learned mixing information head and keep out same as shape x before and matchy with x for resnet)
        return out
        
if __name__ == "__main__":
    torch.manual_seed(seed=42)
    x = torch.randint(low=0, high=20, size=(4, 2))
    print(x)
    
    emb_table = nn.Embedding(20, 16)
    emb_x = emb_table(x)
    print(emb_x.shape)
    
    gqa = GroupQueryAttention(16, 4, 2)
    x_gqa = gqa(emb_x)
    
    print(x_gqa[0].shape)
    print(x_gqa[1].shape)
    print(x_gqa[2].shape)
    print(x_gqa[3].shape)
    print(x_gqa[4].shape)