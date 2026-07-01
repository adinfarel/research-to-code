'''
Build YaRN (Yet Another RoPE extensioN) Scaling
'''

import torch
import torch.nn as nn
import math

class YaRNRoPE(nn.Module):
    def __init__(self, emb_dim, base=10000.0, orig_max_position=2048,
                target_max_position=8192, beta_fast=32, beta_slow=1,
                mscale=1.0, mscale_all_dim=0.0) -> None:
        super().__init__()
        self.dim = emb_dim
        self.orig_max_position = orig_max_position
        self.target_max_position = target_max_position
        scale_factor = target_max_position / orig_max_position
        
        pos_freqs = base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs # (high-freq treatment)
        inv_freq_interpolation = 1.0 / (scale_factor * pos_freqs) # (low-freq treatment)
        
        low = self._find_correction_dim(beta_fast)
        high = self._find_correction_dim(beta_slow)
        low = max(low, 0)
        high = min(high, self.dim // 2 - 1)
        
        ramp = self._linear_ramp(low, high, self.dim // 2)
        inv_freq_mask = 1.0 - ramp  # 1 = full extrapolation, 0 = full interpolation

        inv_freq = (inv_freq_interpolation * (1 - inv_freq_mask) +
                    inv_freq_extrapolation * inv_freq_mask)

        self.register_buffer("inv_freq", inv_freq)
        
        # attention temperature scaling (mscale) -- menyesuaikan
        # magnitude sin/cos biar dot product attention tetap stabil
        if mscale_all_dim:
            self.mscale = float(self._get_mscale(scale_factor, mscale_all_dim))
        else:
            self.mscale = float(self._get_mscale(scale_factor, mscale))
            
    def _find_correction_dim(self, num_rotations):
        return (self.dim * math.log(self.orig_max_position / (num_rotations * 2 * math.pi))) / (2 * math.log(10000.0))

    def _linear_ramp(self, low, high, total_dims):
        if low == high:
            high += 0.001
        
        idx = torch.arange(total_dims, dtype=torch.float32)
        ramp = (idx - low) / (high - low)
        return torch.clamp(ramp, 0, 1)

    def _get_mscale(self, scale, mscale_factor=1.0):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale_factor * math.log(scale) + 1.0
    
    def forward(self, x:torch.Tensor):
        T, C = x.shape[-2], x.shape[-1]

        l = torch.arange(T, dtype=self.inv_freq.dtype, device=x.device)  # type: ignore
        theta = torch.einsum('i,j->ij', l, self.inv_freq)
        hat_theta = torch.cat([theta, theta], dim=-1)
        
        sin = torch.sin(hat_theta) * self.mscale
        cos = torch.cos(hat_theta) * self.mscale
        
        xu, xd = x[..., : C // 2], x[..., C // 2 :]
        hatx = torch.cat([-xd, xu], dim=-1)
        return x * cos + hatx * sin