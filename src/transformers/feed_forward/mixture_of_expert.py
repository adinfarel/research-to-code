'''
Build MoE (Mixture of Expert) implementation
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformers.feed_forward.glu_family import GLUFamily

class MixtureOfExpert(nn.Module):
    def __init__(self, emb_dim, num_experts, top_k, func_act="silu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = nn.Linear(emb_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [GLUFamily(emb_dim=emb_dim, func_act=func_act) for _ in range(num_experts)]
        )
        
        # self.noise_linear = nn.Linear(emb_dim, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        
        x_flat = x.view(B * T, C)
        N = x_flat.shape[0]
        
        # THIS Noisy Top-K Gating 
        # to prevent model only pick smart expert instead of expert else
        # adding noise make model general to pick expert without bias to one expert
        # clean_logits = self.router(x_flat)
        # noise_scale_logits = self.noise_linear(x_flat)
        # noise_scale = F.softplus(noise_scale_logits)
        
        # raw_noise = torch.randn_like(clean_logits)
        
        # router_logits = clean_logits + (raw_noise * noise_scale)
        
        router_logits = self.router(x_flat)
        
        weights, expert_ids = torch.topk(router_logits, self.top_k, dim=-1)
        
        gates = F.softmax(weights, dim=-1)
        
        out_flat = torch.zeros_like(x_flat)
        
        for idx in range(self.num_experts):
            token_indices, k_indices = torch.where(expert_ids == idx)
            
            if len(token_indices) > 0:
                expert_input = x_flat[token_indices]
                
                expert_output = self.experts[idx](expert_input)
                
                gate_weight = gates[token_indices, k_indices].unsqueeze(-1)
                
                out_flat.index_add_(0, token_indices, gate_weight * expert_output)
        
        return out_flat.view(B, T, C)