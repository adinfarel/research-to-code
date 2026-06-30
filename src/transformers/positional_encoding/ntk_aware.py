'''
Build NTK-Aware atom implementation
'''

import torch
import torch.nn as nn
 
class NTKAwareRoPE(nn.Module):
    def __init__(self, emb_dim, base=10000.0, orig_max_pos=2048,
                 target_max_pos=8096):
        super().__init__()
        self.dim = emb_dim
        self.orig_max_pos = orig_max_pos
        self.target_max_pos = target_max_pos
        
        scale_factor = target_max_pos / orig_max_pos
        
        base_scale = base * (scale_factor ** (self.dim / (self.dim - 2)))
        
        inv_freq = 1.0 / (base_scale ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor):
        T, C = x.shape[-2], x.shape[-1]

        l = torch.arange(T, dtype=self.inv_freq.dtype, device=x.device)  # type: ignore
        theta = torch.einsum('i,j->ij', l, self.inv_freq)
        hat_theta = torch.cat([theta, theta], dim=-1)
        sin = torch.sin(hat_theta)
        cos = torch.cos(hat_theta)
        xu, xd = x[..., : C // 2], x[..., C // 2:]
        hatx = torch.cat([-xd, xu], dim=-1)
        return x * cos + hatx * sin