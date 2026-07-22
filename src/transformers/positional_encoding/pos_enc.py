'''
Build Positional Encoding implementation
'''

from __future__ import annotations

import math
import torch
import torch.nn as nn
from abc import abstractmethod, ABC

class PositionalEncoding(nn.Module):
    
    def __init__(self, emb_dim: int, max_position: int = 2048, base: float = 10000.0):
        super().__init__()
        assert emb_dim % 2 == 0, "emb_dim must be even for pairing sin/cos"
        
        self.dim = emb_dim
        self.max_position = max_position
        self.base = float(base)
        
        position = torch.arange(max_position, dtype=torch.float32).unsqueeze(1) # (max_position, 1)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, emb_dim, 2, dtype=torch.float32) / emb_dim))
        # position = (max_position, 1)
        # inv_freq = (inv_freq,)
        # angles   = (max_position, inv_freq)
        angles = position * inv_freq
        
        pe = torch.zeros(max_position, emb_dim)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-2]
        assert T <= self.max_position, f"sequence length {T} exceeds max_position {self.max_position}"
        
        return x + self.pe[:T] #type: ignore
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim}, max_position={self.max_position})"

class ROPEBase(nn.Module, ABC):
    def __init__(self, emb_dim: int, base: float = 10000.0):
        super().__init__()
        assert emb_dim % 2 == 0, "emb_dim must be even for pairing rotary dims"
        
        self.dim = emb_dim
        self.base = float(base)
        
        # for subclass that need orig_max_pos like YaRN, NTK, etc.
        # must set attribut before super().__init__(...)
        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq)
        self.mscale = self._compute_mscale()
    
    @abstractmethod
    def _compute_inv_freq(self) -> torch.Tensor:
        raise NotImplementedError
    
    def _compute_mscale(self) -> float:
        return 1.0
    
    @staticmethod
    def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, C = x.shape[-2], x.shape[-1]
        
        l = torch.arange(T, dtype=self.inv_freq.dtype, device=x.device) #type: ignore
        theta = torch.einsum("i,j->ij", l, self.inv_freq) # (T, dim/2)
        
        hat_theta = theta.repeat_interleave(2, dim=-1) # (T, dim)
        
        sin = torch.sin(hat_theta) * self.mscale
        cos = torch.cos(hat_theta) * self.mscale
        
        return x * cos + self._rotate_interleaved(x) * sin

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim}, base={self.base})"

class ROPE(ROPEBase):
    
    def _compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))

class NTKAwareROPE(ROPEBase):
    
    def __init__(self, emb_dim, base=10000.0, orig_max_pos=2048, target_max_pos=8096):
        self.orig_max_pos = orig_max_pos
        self.target_max_pos = target_max_pos
        super().__init__(emb_dim=emb_dim, base=base)
    
    def _compute_inv_freq(self) -> torch.Tensor:
        scale_factor = self.target_max_pos / self.orig_max_pos
        base_scale   = self.base  * (scale_factor ** (self.dim / (self.dim - 2)))
        return 1.0 / (base_scale ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(dim={self.dim}, base={self.base}, "
                f"orig_max_pos={self.orig_max_pos}, target_max_pos={self.target_max_pos})")

class YaRNROPE(ROPEBase):
    
    def __init__(self, emb_dim, base=10000.0, orig_max_position=2048,
                 target_max_position=8192, beta_fast=32, beta_slow=1,
                 mscale=1.0, mscale_all_dim=0.0):
        self.orig_max_position = orig_max_position
        self.target_max_position = target_max_position
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self._mscale_arg = mscale
        self._mscale_all_dim_arg = mscale_all_dim
        self.scale_factor = target_max_position / orig_max_position
        super().__init__(emb_dim, base)
    
    def _compute_inv_freq(self) -> torch.Tensor:
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale_factor * pos_freqs)
        
        low = self._find_correction_dim(self.beta_fast)
        high = self._find_correction_dim(self.beta_slow)
        low = max(low, 0)
        high = min(high, self.dim // 2 - 1)
        
        ramp = self._linear_ramp(low, high, self.dim // 2)
        inv_freq_mask = 1.0 - ramp
        
        inv_freq = (inv_freq_interpolation * (1 - inv_freq_mask) + 
                    inv_freq_extrapolation * inv_freq_mask)
        
        return inv_freq

    def _compute_mscale(self) -> float:
        if self._mscale_all_dim_arg:
            return float(self._get_mscale(self.scale_factor, self._mscale_all_dim_arg))
        return float(self._get_mscale(self.scale_factor, self._mscale_arg))
    
    def _find_correction_dim(self, num_rotations):
        return (self.dim * math.log(self.orig_max_position / (num_rotations * 2 * math.pi))) / (2 * math.log(self.base))
    
    def _linear_ramp(self, lw, hi, total_dims):
        if lw == hi:
            hi += 0.001
            
        idx = torch.arange(total_dims, dtype=torch.float32)
        ramp = (idx - lw) / (hi - lw)
        return torch.clamp(ramp, 0, 1)
    
    def _get_mscale(self, scale, mscale_factor=1.0):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale_factor * math.log(scale) + 1.0
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(dim={self.dim}, base={self.base}, "
                f"orig_max_position={self.orig_max_position}, "
                f"target_max_position={self.target_max_position}, mscale={self.mscale:.4f})")