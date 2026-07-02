'''
Build DyT (Dynamic Tanh) implementation
'''

import torch
import torch.nn as nn

class DynamicErf(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        # Parameter Scale (Alpha)
        self.alpha = nn.Parameter(torch.ones(embedding_dim))
        
        # Parameter Shifting (Shift / Bias)
        self.shift = nn.Parameter(torch.zeros(embedding_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # formula: erf(alpha * x + shift)
        return torch.erf(self.alpha * x + self.shift)