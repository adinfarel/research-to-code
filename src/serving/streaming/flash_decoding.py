'''
Build flash decoding in terms GPU
'''

from pickle import REDUCE

from numpy import dtype
import triton
import triton.language as tl

@triton.jit
def _fd_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    SEQ_LEN_KV: tl.constexpr,
    split_start,
    split_end,
):
    lo = split_start
    hi = split_end
    
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        
        # load K tile
        K_block = tl.load(K_block_ptr)
        # compute QK^T
        QK_block = tl.dot(Q_block, K_block)
        QK_block = QK_block * softmax_scale
        
        # online softmax
        m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1),)
        QK_block -= m_ij[:, None] # stable softmax to prevent overflow exp
        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, axis=1)
        alpha = tl.math.exp(m_i - m_ij) # correction factor
        l_i = l_i * alpha + l_ij
        
        # load V tile
        V_block = tl.load(V_block_ptr)
        
        P_block = P_block.to(tl.float16)
        
        # running output
        O_block = O_block * alpha[:, None]
        
        O_block = tl.dot(
            P_block,
            V_block,
            O_block
        ) # O_block += P_block @ V_block
        
        m_i = m_ij
        
        # shift pointer to the next block KV
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
    
    return O_block, l_i, m_i

@triton.jit
def _fd_fwd(
    Q,
    K,
    V,
    softmax_scale,
    Partial_O,
    Partial_M,
    Partial_L,
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
    stride_O_splits,
    stride_O_batch,
    stride_O_head,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN_Q: tl.constexpr,
    SEQ_LEN_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    NUM_SPLITS: tl.constexpr
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    
    split_idx = tl.program_id(0)
    
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    
    split_size = tl.cdiv(
        SEQ_LEN_KV,
        NUM_SPLITS
    )
    
    split_start = split_idx * split_size
    split_end   = tl.minimum(
        split_size + split_start,
        SEQ_LEN_KV
    )
    
    q_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    k_offset = (
        index_batch.to(tl.int64) * stride_K_batch
        + index_head.to(tl.int64) * stride_K_head
    )

    v_offset = (
        index_batch.to(tl.int64) * stride_V_batch
        + index_head.to(tl.int64) * stride_V_head
    )
    
    partial_offset = (
        split_idx * stride_O_splits + 
        index_batch * stride_O_batch +
        index_head * stride_O_head
    )
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(SEQ_LEN_Q, HEAD_DIM), # (1, HEAD_DIM)
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(0, 0),
        order=(1,0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
    )
    
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset, # K[index_batch, index_head, :, :]
        shape=(HEAD_DIM, SEQ_LEN_KV),
        strides=(stride_K_dim, stride_K_seq), # Trick transpose without .transpose(), but manipulate strides (How many pointer move)
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1)
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset, # V[index_batch, index_head, :, :]
        shape=(SEQ_LEN_KV, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0)
    )
    
    stride_O_seq = stride_O_dim * HEAD_DIM
    
    Partial_O_ptr = tl.make_block_ptr(
        base=Partial_O + partial_offset, # O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
        shape=(SEQ_LEN_Q, HEAD_DIM), # (1, HEAD_DIM)
        strides=(stride_O_seq, stride_O_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0)
    )
    
    # m_i: running max for each row query (max, 1)
    m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32) - float("-inf")
    # l_i: running sum for each row query (as a denominator for normalize each value in one row query)
    l_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32) + 1.0
    # output: place for store accumulation from operations
    O_block = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)
    # O_block = (xi - m_i) / l_i
    
    # load query
    Q_block = tl.load(Q_block_ptr, boundary_check=(0,))
    
    O_block, l_i, m_i = _fd_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr,
        V_block_ptr,
        softmax_scale,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        SEQ_LEN_KV,
        split_start,
        split_end
    )
    
    partial_m_offset = (
        split_idx * BATCH_SIZE * NUM_HEADS + 
        index_batch_head
    )
    
    tl.store(Partial_L + partial_m_offset, l_i)
    tl.store(Partial_M + partial_m_offset, m_i)
    tl.store(Partial_O_ptr, O_block.to(Partial_O.type.element_ty))

@triton.jit
def _fd_reduce(
    Partial_O,
    Partial_M,
    Partial_L,
    O,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    index_batch_head = tl.program_id(0)
    
    global_m = -float("inf")
    for i in range(NUM_SPLITS):
        offset = i * BATCH_SIZE * NUM_HEADS + index_batch_head
        m_val = tl.load(Partial_M + offset)
        global_m = tl.maximum(global_m, m_val)
    
    l_acc = 0.0
    o_acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    
    dim_offsets = tl.arange(0, HEAD_DIM)
    
    for i in range(NUM_SPLITS):
        offset = i * BATCH_SIZE * NUM_HEADS + index_batch_head
        m_val = tl.load(Partial_M + offset)
        l_val = tl.load(Partial_L + offset)
        
        correction = tl.math.exp(m_val - global_m)
        l_acc += l_val * correction
        
        o_base = offset * HEAD_DIM
        o_val = tl.load(Partial_O + o_base + dim_offsets)
        o_acc += o_val * correction
    
    o_final = o_acc / l_acc
    out_base = index_batch_head * HEAD_DIM
    tl.store(O + out_base + dim_offsets, o_final)

import torch

def flash_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    softmax_scale: float | None = None, 
):
    assert Q.is_cuda
    assert K.is_cuda
    assert V.is_cuda
    
    BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM = Q.shape
    _, _, SEQ_LEN_KV, _ = K.shape
    
    assert SEQ_LEN_Q == 1
    
    assert K.shape == (
        BATCH_SIZE,
        NUM_HEADS,
        SEQ_LEN_KV,
        HEAD_DIM,
    )

    assert V.shape == K.shape
    NUM_SPLITS = 8
    
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** -0.5
        
    Partial_O = torch.empty(
        (
            NUM_SPLITS,
            BATCH_SIZE,
            NUM_HEADS,
            HEAD_DIM,
        ),
        device=Q.device,
        dtype=torch.float32
    )
    Partial_M = torch.empty(
        (
            NUM_SPLITS,
            BATCH_SIZE,
            NUM_HEADS,
        ),
        device=Q.device,
        dtype=torch.float32,
    )
    Partial_L = torch.empty(
        (
            NUM_SPLITS,
            BATCH_SIZE,
            NUM_HEADS,
        ),
        device=Q.device,
        dtype=torch.float32,
    )

    BLOCK_SIZE_Q = 1
    
    grid = (
        NUM_SPLITS,
        BATCH_SIZE * NUM_HEADS,
    )

    _fd_fwd[grid](
        Q,
        K,
        V,
        softmax_scale,
        Partial_O,
        Partial_M,
        Partial_L,
        # Q stride
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        # K stride
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        # V stride
        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        # Partial_O stride
        Partial_O.stride(0),
        Partial_O.stride(1),
        Partial_O.stride(2),
        Partial_O.stride(3),
        BATCH_SIZE=BATCH_SIZE,
        NUM_HEADS=NUM_HEADS, #type: ignore
        SEQ_LEN_Q=SEQ_LEN_Q, #type: ignore
        SEQ_LEN_KV=SEQ_LEN_KV, #type: ignore
        HEAD_DIM=HEAD_DIM, #type: ignore
        BLOCK_SIZE_Q=BLOCK_SIZE_Q, #type: ignore
        BLOCK_SIZE_KV=128, #type: ignore
        NUM_SPLITS=NUM_SPLITS, #type: ignore
    )
    
    O = torch.empty(
        (BATCH_SIZE, NUM_HEADS, HEAD_DIM),
        device=Q.device, dtype=Q.dtype
    )
    
    reduce_grid = (BATCH_SIZE * NUM_HEADS,)

    _fd_reduce[reduce_grid](
        Partial_L,
        Partial_M,
        Partial_O,
        O,
        BATCH_SIZE=BATCH_SIZE, #type: ignore
        NUM_HEADS=NUM_HEADS, #type: ignore
        HEAD_DIM=HEAD_DIM, #type: ignore
        NUM_SPLITS=NUM_SPLITS, #type: ignore
    )
    
    return O.view(BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM)