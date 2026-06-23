'''
Testing flash attention triton whether running correctly
'''

import torch
from src.triton.flash_attention_triton import TritonAttention

def test_op():
    Q = (
        torch.empty(
            (2, 2, 8, 4), device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (2, 2, 8, 4), device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (2, 2, 8, 4), device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (4**0.5)
    dO = torch.randn_like(Q)
    
    MASK = torch.tril(torch.ones((8, 8), device="cuda"))
    P = torch.matmul(Q, K.transpose(2,3)) * softmax_scale
    causal = True
    if causal:
        P[:, :, MASK == 0] = float("-inf")
        # OR
        # P = P.masked_fill(MASK[:SEQ_LEN, :SEQ_LEN] == 0, float("-inf"))
    P = torch.softmax(P.float(), dim=-1).half()
    
    # Naive imp
    ref_D = torch.matmul(P, V)
    ref_D.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None #type: ignore
    ref_dK, K.grad = K.grad.clone(), None #type: ignore
    ref_dQ, Q.grad = Q.grad.clone(), None #type: ignore
    
    # Triton imp
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None #type: ignore
    tri_dK, K.grad = K.grad.clone(), None #type: ignore
    tri_dQ, Q.grad = Q.grad.clone(), None #type: ignore
    
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_D, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)