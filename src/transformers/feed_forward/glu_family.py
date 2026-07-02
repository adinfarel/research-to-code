'''
Build GLU-Family from scratch

Why we need feed forward?
    In ML/DL we need non-linearity feature but why?
    cause if in model just contain linear matrix then model can do trick or can be call "Linear Collapse"
    because linearity property in algebra is associative,
    example:
    A x (B x C) is same as (A x B) x C so model can do trick before matmul with X
    model to do matmul first W_absorb = A x (B x C) then Act = W_absorb x X
    but if we put in non-linearity function in equation
    A x (ReLU(B) x C) see we can apply associative for that chain linear algebra becomes disconnected that's why we need non-linearity
    and also we dont want to model can dealing with linearity information cause if we think in 2d graph and we pull
    straight line (which is linear), we just can get information of data around straight line, we cant get information
    if data located at the very bottom line or very top line so we need non-linearity
    
    Oke, in regular FFWD we just expand and clipping the data if x < 0 and squash
    but in SwiGLU we adding mechanism gating
    so insted of force clipping data to zero, what if we seperate become 3 matrix
    Gated Matrix: to calculate x importance
    Value Matrix: to search new representation feature
    Down_Proj: to squash again so that compatible with residual connection and force model to transform non-linearity high dimension to lower
    dimension (bottleneck effect) so we can get combination non-linearity feature
    
    If x those that have entered gate and exited, if x not clipping to zero then while multiply with value (element-wise)
    we can get amplication of x, vice versa if x clipping to zero we elimanate not important feature
    then down projection
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class GLUFamily(nn.Module):
    def __init__(self, emb_dim, func_act="silu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.func_act = func_act
        self.hid_dim = int(emb_dim * (8/3)) # it's same as emb_dim * 4 == emb_dim * (8/3)
        
        self.value = nn.Linear(emb_dim, self.hid_dim, bias=False)
        self.gated = nn.Linear(emb_dim, self.hid_dim, bias=False)
        self.down_p = nn.Linear(self.hid_dim, emb_dim, bias=False)
    
    def forward(self, x: torch.Tensor):
        value = self.value(x) # get new representation feature
        gated = self.gated(x) # filter feature importance
        
        if self.func_act == "silu":
            gated = F.silu(gated)
        elif self.func_act == "gelu":
            gated = F.gelu(gated)
        elif self.func_act == "relu":
            gated = F.relu(gated)
        else:
            raise ValueError(f"Unknown function activation: {str(self.func_act)!r}")
        
        hidden = value * gated
        
        return self.down_p(hidden) # sqaush again to lower dimension so that compatible to next layer