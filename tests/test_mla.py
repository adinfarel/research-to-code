'''
Testing MultiLatentAttention implementation
'''

import torch

from src.transformers.attention.mla import MultiLatentAttention


def test_mla_output_shape():
    B, T, C, n_head, head_size = 2, 5, 16, 4, 4
    x = torch.randn(B, T, C)

    mla = MultiLatentAttention(emb_dim=C, n_head=n_head, head_size=head_size)
    out = mla(x)

    assert out.shape == (B, T, C)


def test_mla_kv_cache_size_reduction():
    B, T, C, n_head, head_size = 2, 5, 32, 4, 8
    kv_lora_rank = 4 
    
    x = torch.randn(B, T, C)
    mla = MultiLatentAttention(emb_dim=C, n_head=n_head, head_size=head_size, kv_lora_rank=kv_lora_rank)
    
    out, kv_cache = mla(x, return_cache=True)

    assert kv_cache.shape == (B, T, kv_lora_rank)

    mha_equivalent_single_cache = n_head * head_size
    assert kv_cache.shape[-1] < mha_equivalent_single_cache


def test_mla_param_count_reflects_low_rank_compression():
    C, n_head, head_size = 32, 4, 8
    kv_lora_rank = 8
    
    mla = MultiLatentAttention(emb_dim=C, n_head=n_head, head_size=head_size, kv_lora_rank=kv_lora_rank)
    
    kv_down_params = sum(p.numel() for p in mla.kv_down_proj.parameters())
    
    expected_down_proj_size = C * kv_lora_rank
    assert kv_down_params <= expected_down_proj_size + kv_lora_rank


def test_mla_decompress_recovers_head_dimensions_internally():
    B, T, C, n_head, head_size = 1, 3, 16, 4, 4
    kv_lora_rank = 4
    
    x = torch.randn(B, T, C)
    mla = MultiLatentAttention(emb_dim=C, n_head=n_head, head_size=head_size, kv_lora_rank=kv_lora_rank)

    kv_latent = mla.kv_down_proj(x)
    assert kv_latent.shape == (B, T, kv_lora_rank)
    
    K_recovered = mla.k_up_proj(kv_latent).view(B, T, n_head, head_size).transpose(1, 2)
    V_recovered = mla.v_up_proj(kv_latent).view(B, T, n_head, head_size).transpose(1, 2)
    
    assert K_recovered.shape == (B, n_head, T, head_size)
    assert V_recovered.shape == (B, n_head, T, head_size)