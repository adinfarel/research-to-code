'''
Build Rotary Positional Encoding atom implementations

Intuition:
    Rope is positional encodoing that use math trigonometry, rotary each token based on position token in sentence
    Why ROPE exists?
        ROPE answer regular positional encoding which worse for extrapolation and too bad to learning position each token based on static position
        LPE cannot extrapolate sequence length, so if model meet new token position for example model learn just position 0 - 20 token
        so if meet token 21 model confuse and getting catasthropic, then if model meet token 1 'meet' with token 3 'her' if 
        in one batch model meet 2 token like that in the same range (with difference 1 token), if 2 token that attend
        in different row model learn again for the same token but in different position
        for example token 1 and token 2 in sentence 1 in a position early sentence if model meet again same token
        but in a position last of sentence model have to learn again about representation position that token
    
    so ROPE solve that problem, ROPE use 2D rotate mechanism which is, no matter (for example) 2 token exist in last or early sentence
    as long as that token always close to each other model can rotate wheel model as many as in position that token
    if result dot product between 2 token large then it'is always large, so with the exist of a rope model can quickly
    convergen because model know difference between token just same no matter that token in a position
    
    maybe we wondering, whether rope quickly forget tokens that are far away eventhough that token have large affinity?
    this is a beauty of ROPE, model can still learn about that how far difference to each other
    because rope have 2 columns, columns fast and slow, 
    columns slow save that token have large affinity but far distance
    so there is some tolerance for token with wide gap
'''

import torch
import torch.nn as nn

class RotaryPositionalEncoding(nn.Module):
    
    def __init__(self, emb_dim, base=10000.0) -> None:
        super().__init__()
        self.dim = emb_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor):
        T, C = x.shape[-2], x.shape[-1]
        
        l = torch.arange(T, dtype=self.inv_freq.dtype, device=x.device) #type: ignore
        theta = torch.einsum('i,j->ij', l, self.inv_freq)
        hat_theta = torch.cat([theta, theta], dim=-1)
        sin = torch.sin(hat_theta)
        cos = torch.cos(hat_theta)
        xu, xd = x[..., : C // 2], x[..., C // 2 :]
        hatx = torch.cat([-xd, xu], dim=-1)
        return x * cos + hatx * sin