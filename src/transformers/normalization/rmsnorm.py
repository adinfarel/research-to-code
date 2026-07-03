'''
Build RMSNorm (Root-Mean-Squared Normalization) implementation
'''

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Parameter Gamma (Gain) learnable
        self.weight = nn.Parameter(torch.ones(emb_dim))
    
    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        
        x_f32 = x.float()
        
        # x * 1 / sqrt(mean(x**2) + eps)
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(variance + self.eps)
        
        x_norm = (x_f32 * rms).to(orig_dtype)
        
        return x_norm * self.weight