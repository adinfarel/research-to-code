'''
Build DyT (Dynamic Tanh) implementation
'''

import torch
import torch.nn as nn

class DynamicTanh(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # Alpha
        self.alpha = nn.Parameter(torch.ones(emb_dim))
    
    def forward(self, x: torch.Tensor):
        return torch.tanh(self.alpha * x)
    
