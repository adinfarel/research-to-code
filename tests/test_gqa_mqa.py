'''
Testing MultiQueryAttention and GroupQueryAttention implementations
'''

import torch

from src.transformers.attention.mqa import MultiQueryAttention
from src.transformers.attention.gqa import GroupQueryAttention


def test_mqa_output_shape():
    B, T, C, n_head = 2, 5, 16, 4
    x = torch.randn(B, T, C)

    mqa = MultiQueryAttention(emb_dim=C, n_head=n_head)
    out = mqa(x)

    assert out.shape == (B, T, C)


def test_mqa_kv_param_count_smaller_than_mha_equivalent():
    # Core motivation check: MQA's K/V projection should NOT scale
    # with n_head, unlike a standard MHA K/V projection would.
    C, n_head = 16, 4
    mqa = MultiQueryAttention(emb_dim=C, n_head=n_head)

    head_size = C // n_head
    k_params = sum(p.numel() for p in mqa.key.parameters())
    v_params = sum(p.numel() for p in mqa.value.parameters())

    # K/V projection output dim should be head_size (single head worth),
    # NOT n_head * head_size (which would be the MHA-equivalent size)
    expected_single_head_out = head_size * C  # weight matrix size (no bias)
    assert k_params <= expected_single_head_out + head_size  # +bias tolerance
    assert v_params <= expected_single_head_out + head_size


def test_gqa_output_shape():
    B, T, C, n_head, n_kv_head = 2, 5, 16, 4, 2
    x = torch.randn(B, T, C)

    gqa = GroupQueryAttention(emb_dim=C, n_head=n_head, n_kv_head=n_kv_head)
    out = gqa(x)

    assert out.shape == (B, T, C)


def test_gqa_equals_mha_when_n_kv_head_equals_n_head():
    # Sanity check: GQA with n_kv_head == n_head should behave like
    # standard MHA (no sharing happening, num_queries_each_kv_head == 1)
    B, T, C, n_head = 1, 3, 16, 4
    x = torch.randn(B, T, C)

    gqa = GroupQueryAttention(emb_dim=C, n_head=n_head, n_kv_head=n_head)

    assert gqa.num_queries_each_kv_head == 1
    out = gqa(x)
    assert out.shape == (B, T, C)


def test_gqa_equals_mqa_when_n_kv_head_is_one():
    # Sanity check: GQA with n_kv_head=1 should be structurally
    # equivalent to MQA (all heads share single K/V)
    B, T, C, n_head = 1, 3, 16, 4
    x = torch.randn(B, T, C)

    gqa = GroupQueryAttention(emb_dim=C, n_head=n_head, n_kv_head=1)

    assert gqa.num_queries_each_kv_head == n_head
    out = gqa(x)
    assert out.shape == (B, T, C)


def test_gqa_kv_cache_size_reduction():
    # Verify the actual memory-saving claim: GQA's K/V cache size
    # (per token) should be proportional to n_kv_head, not n_head.
    C, n_head, n_kv_head = 16, 8, 2
    head_size = C // n_head

    gqa = GroupQueryAttention(emb_dim=C, n_head=n_head, n_kv_head=n_kv_head)

    x = torch.randn(1, 1, C)  # single token
    K = gqa.key(x)
    V = gqa.value(x)

    # K/V cache size per token should be n_kv_head * head_size,
    # NOT n_head * head_size (which is what standard MHA would store)
    assert K.shape[-1] == n_kv_head * head_size
    assert V.shape[-1] == n_kv_head * head_size
    assert K.shape[-1] < n_head * head_size  # confirms actual reduction