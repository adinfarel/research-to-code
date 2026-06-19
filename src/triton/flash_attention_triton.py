'''
Build flash attention but in triton term (Low-Level GPU)
'''

from numpy import dtype
import torch

import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    O_block,
    m_i,
    l_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    
    # IF CAUSAL:
    #
    #     [[Diag_Block, Mask_Block, Mask_Block],
    #      [No_Mask_Block, Diag_Block, Mask_Block],
    #      [No_Mask_Block, No_Mask_Block, Diag_Block]]
    
    if STAGE == 1:
        # from 0 to the left of diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 3:
        # used only for the block which there is transition between non-masked and masked keys
        # diagonal block
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # only if use non-causal attn
        lo, hi = 0, SEQ_LEN
    
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # let the compiler know that start_kv is a multiple of BLOCK_SIZE_KV
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        
        # -- compute qk --
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)
        
        if STAGE == 2:
            mask = offs_q[:, None] >= [start_kv + offs_kv[None, :]]
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # compute exponential each value (element-wise) that stable by - maximum value each row query
        P_block = tl.math.exp(QK_block)
        
        # compute sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)
        
        # correction factor for prev l_i
        alpha = tl.math.exp(m_i - m_ij)
        
        # compute l_i with correction factor
        l_i = l_i * alpha + l_ij # in flash_attention pytorch l_ij * beta, but this time no
        
        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        
        # compute output: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block) # O_block += P_block @ V_block
        
        m_i = m_ij
         
        # move to the next block
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, K_block_ptr))
    
    return O_block, l_i, m_i

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
    
    # offset_q: all token q based on block_id
    offs_q = block_index_q * BLOCK_SIZE_Q * tl.arange(0, BLOCK_SIZE_Q)
    # offset_kv: the all token k and v of this block
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    # m_i: running max for each row query (max, 1)
    m_i = tl.zeros((BLOCK_SIZE_Q), dtype=tl.float32) - float("-inf")
    # l_i: running sum for each row query (as a denominator for normalize each value in one row query)
    l_i = tl.zeros((BLOCK_SIZE_Q), dtype=tl.float32) + 1.0
    # output: place for store accumulation from operations
    O_block = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)
    # O_block = (xi - m_i) / l_i
    
    # Stage: 3 if causal else 1
    Q_block = tl.load(Q_block_ptr)
    
    if STAGE == 1 or STAGE == 3:
        # Runs with non-causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN
        )
    
    if STAGE == 3:
        # Runs with causal attention, which is can't see info the next token cause masking
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN
        )
        
    m_i += tl.math.log(
        l_i
    ) # needed to compute logsumexp for the backward pass
    
    # normalize block
    O_block = O_block / l_i[:, None]
    
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))

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