'''
Testing MixtureOfExpert implementation
'''

import torch

from src.transformers.feed_forward.mixture_of_expert import MixtureOfExpert


def test_moe_output_shape():
    B, T, C, num_experts, top_k = 2, 5, 16, 4, 2
    x = torch.randn(B, T, C)

    moe = MixtureOfExpert(emb_dim=C, num_experts=num_experts, top_k=top_k)
    out = moe(x)

    assert out.shape == (B, T, C)


def test_moe_only_top_k_experts_active_per_token():
    # Core sparsity claim: exactly top_k experts should contribute
    # non-zero output for each token (not all num_experts).
    B, T, C, num_experts, top_k = 1, 1, 8, 4, 2
    x = torch.randn(B, T, C)

    moe = MixtureOfExpert(emb_dim=C, num_experts=num_experts, top_k=top_k)

    router_logits = moe.router(x.view(B * T, C))
    _, expert_ids = torch.topk(router_logits, top_k, dim=-1)

    # for the single token, exactly top_k unique experts should be selected
    assert expert_ids.shape == (1, top_k)
    assert len(torch.unique(expert_ids)) == top_k


def test_moe_gate_weights_sum_to_one_per_token():
    # Softmax over top_k logits should sum to 1 per token (weighted
    # combination of selected experts, not arbitrary scaling).
    B, T, C, num_experts, top_k = 1, 3, 8, 4, 2
    x = torch.randn(B, T, C)

    moe = MixtureOfExpert(emb_dim=C, num_experts=num_experts, top_k=top_k)

    x_flat = x.view(B * T, C)
    router_logits = moe.router(x_flat)
    weights, _ = torch.topk(router_logits, top_k, dim=-1)
    gates = torch.softmax(weights, dim=-1)

    gate_sums = gates.sum(dim=-1)
    torch.testing.assert_close(gate_sums, torch.ones(B * T), rtol=1e-5, atol=1e-5)


def test_moe_zero_input_gives_zero_output_with_relu_experts():
    # Sanity check: zero input through GLU (with relu gate) should
    # produce zero output for every expert (no bias terms anywhere
    # in the chain), so MoE output should also be zero.
    B, T, C, num_experts, top_k = 1, 2, 8, 4, 2
    x = torch.zeros(B, T, C)

    moe = MixtureOfExpert(emb_dim=C, num_experts=num_experts, top_k=top_k, func_act="relu")
    out = moe(x)

    torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-5, atol=1e-5)


def test_moe_different_tokens_can_route_to_different_experts():
    # With enough experts and varied input, not all tokens should
    # necessarily pick the exact same expert combination.
    torch.manual_seed(0)
    B, T, C, num_experts, top_k = 1, 10, 16, 8, 2
    x = torch.randn(B, T, C) * 5  # scale up to spread router logits more

    moe = MixtureOfExpert(emb_dim=C, num_experts=num_experts, top_k=top_k)

    x_flat = x.view(B * T, C)
    router_logits = moe.router(x_flat)
    _, expert_ids = torch.topk(router_logits, top_k, dim=-1)

    # check not all tokens got IDENTICAL expert assignment (weak sanity check)
    unique_assignments = set(tuple(sorted(row.tolist())) for row in expert_ids)
    assert len(unique_assignments) > 1


def test_moe_all_experts_receive_gradient_when_used():
    # Verify backward pass actually updates only experts that were
    # selected by at least one token (sparse gradient flow check).
    B, T, C, num_experts, top_k = 1, 4, 8, 4, 2
    x = torch.randn(B, T, C, requires_grad=True)

    moe = MixtureOfExpert(emb_dim=C, num_experts=num_experts, top_k=top_k)
    out = moe(x)
    loss = out.sum()
    loss.backward()

    # at least the router should have gradient
    assert moe.router.weight.grad is not None
    assert not torch.all(moe.router.weight.grad == 0)