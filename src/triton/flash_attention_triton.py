'''
Build flash attention but in triton term (Low-Level GPU)
'''

import torch

import triton
import triton.language as tl

@triton.jit
def _attn_fwd(
    Q, 
    K,
    V,
    softmax_scale,
    O,
    M,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr, 
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    
    block_index_q = tl.program_id(axis=0)
    
    index_batch_head = tl.program_id(axis=1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    
    qkv_offsets = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offsets, # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0)
    )
    
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offsets, # K[index_batch, index_head, :, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_K_dim, stride_K_seq), # Trick transpose without .transpose(), but manipulate strides (How many pointer move)
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1)
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offsets, # V[index_batch, index_head, :, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0)
    )
    
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_offsets, # O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0)
    )
    
    return

class TritonAttention(torch.autograd.function): #type: ignore
    
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        
        # pre-allocate
        O = torch.empty_like(Q)
        stage = 3 if causal else 1
        
        # Number of parallel programs: (BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q)
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1, # Z in the CUDA launch grid
        )

        M = torch.empty(
            (BATCH_SIZE, HEAD_DIM, SEQ_LEN), device=Q.device, dtype=torch.float32
        )
        
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride[0],
            stride_Q_head=Q.stride[1],
            stride_Q_seq=Q.stride[2],
            stride_Q_dim=Q.stride[3],
            stride_K_batch=K.stride[0],
            stride_K_head=K.stride[1],
            stride_K_seq=K.stride[2],
            stride_K_dim=K.stride[3],
            stride_V_batch=V.stride[0],
            stride_V_head=V.stride[1],
            stride_V_seq=V.stride[2],
            stride_V_dim=V.stride[3],
            stride_O_batch=O.stride[0], #type: ignore
            stride_O_head=O.stride[1], #type: ignore
            stride_O_seq=O.stride[2], #type: ignore
            stride_O_dim=O.stride[3], #type: ignore
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage
        )
        
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O
        
def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)
    
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2,3)) * softmax_scale
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
    tri_out = TritonAttention(Q, K, V, causal, softmax_scale).half()
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