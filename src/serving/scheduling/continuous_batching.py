'''
Build Continuous Batching implementation

INTUITION:
    If we dealing with real user, we can get 1 prompt req if we generate 1 prompt for 1 result
    for one time, it's take time too long for dealing many request, and make the lower throughput
    because GPU happy if we dealing many data in 1 instruction (SIMD) why we not try in one batch?
    the result is the throughput become higher because GPU was exists for many task
    
    Continuous Batching, one of method for dealing with many request in real user system
    how this solve problem of Static Batching, we know SB have problem that if launch 1 batch
    and compute it, we finding that we must waiting as a whole request finish, we must wait
    all request in batch finish which mean one req that have long sequence hit EOS, this is worse
    this is make 1 req have finished become idle
    for example
    [t1, t2, t3, t4, .., .., .., ..]
    [t1, t2, t3, t4, t5, t6, t7, EOS]
    [t1, t2, t3, t4, t5, .., .., ..]
    See req 0 and 2 idle position and have to wait req 1 hit token EOS, so this is wasted potency of GPU
    which can dealing parallel input and this is not efficient, and we cannot predict how longer 1 batch it takes because if 1 req finish
    we can release the result to user and we cant inject new request to batch
    
    CB solve this problem instead of let it GPU in idle position we release 1 req have finished and 
    inject 1 new req so nothing idle GPU and we utilize GPU to the maximum
    
    How flow CB
    maybe we think CB logic like this
    (B, T)
    [--->]
    [--->]
    [--->]
    if 1 req done we replace row level and inject new req as a new row in the batch
    i also think like that until i know that it's not like that:
    CB running like this:
    we receive one batch and we flatten the batch so,
    (B, T) --> (B * T)
    maybe we wondering, if we flatten how we exactly know each token for each request then cu_seqlens solve this
    problem cu_seqlens save address memory for each token in request so we not wrong address
    for example
    [I am (Req 0), Heaven (Req 1), Hell (Req 1), Good (Req 2)]
    cu_seqlens:
    [0, 1, 3, 4]
    this mean req 1 [0:1], req 2 [1:3], req 3 [3:4]
    but why req 2 have 2 token, because req 2 is prefill phase so we can merge decode phase and prefill phase
    in 1D array, but trade-off this process decode request for this iterate make slower rather than 
    1D arrah full 1 decode phase
    flow flatten batch if come in to trasnformers model
    we have (B, T) --flatten--> (B * T)
    Embedding: (B * T, C)
    Linear   : (B * T, C), notice if we dealing in linear layer we can compute together by parallel 
    because linear layer token-wise, because this process can do by parallel we just need launce kernel once
    instead of launch kernel for each request, and this is save memory bound (usually kernel GEMM for matmul)
    Attention: (B * T, C) this is unique, we cannot compute by parallel because attention is sequence-wise
    that's means we must split req by address memory (cu_seqlens useful for this case) and compute req
    by KV-cache the request itself because kv cache req 1 cannot attend query req 2
    so each quary must attend to the kv cache request query own

NOTE: This is just implementation by Pytorch Naive doesn't representation real case
'''

import torch
import torch.nn.functional as F

class ContinuousBatching:
    
    def __init__(self, embed_dim=64, n_heads=4):
        assert embed_dim % n_heads == 0
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.softmax_scale = 1 / (self.head_dim ** 0.5)
        
        self.W_qkv = torch.randn(embed_dim, 3 * embed_dim)
        self.W_out = torch.randn(embed_dim, embed_dim)
        
        self.active_request = {}
        self.kv_caches = {}
    
    def inject_request(self, req_id, prompt_tokens):
        self.active_request[req_id] = {
            "tokens": prompt_tokens,
            "phase": "prefill"
        }
        self.kv_caches[req_id] = {"K": None, "V": None}
        print(f"[INJECT] Request '{req_id}' put in bring {prompt_tokens.size(0)} token (Prefill).")
    
    def step(self):
        if not self.active_request:
            print(f"SERVER IDLE, NOT HAVE REQUEST")
            return

        req_ids = list(self.active_request.keys())
        
        flattened_tokens_list = []
        cu_seqlens = [0]
        curr_offset = 0
        
        
        for r_id in req_ids:
            req = self.active_request[r_id]
            flattened_tokens_list.append(req['tokens'])
            
            num_tokens = req['tokens'].size(0)
            curr_offset += num_tokens
            cu_seqlens.append(curr_offset)
        
        X_batch_2d = torch.cat(flattened_tokens_list, dim=0) # List[Tensor] --> flatten Tensor([])
        
        print(f"\n[NEW ITERASI] Process total {X_batch_2d.size(0)} from token {len(req_ids)} request.")
        print(f"Metadata cu_seqlens: {cu_seqlens}")
        
        qkv = torch.matmul(X_batch_2d, self.W_qkv)
        
        Q, K, V = torch.chunk(qkv, 3, dim=-1)
        
        Q = Q.view(-1, self.n_heads, self.head_dim)
        K = K.view(-1, self.n_heads, self.head_dim)
        V = V.view(-1, self.n_heads, self.head_dim)
        
        attention_outputs = []
        
        for i, r_id in enumerate(req_ids):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i+1]
            
            q_user = Q[start_idx:end_idx]
            k_user = K[start_idx:end_idx]
            v_user = V[start_idx:end_idx]
            
            if self.active_request[r_id]["phase"] == "prefill":
                self.kv_caches[r_id]["K"] = k_user
                self.kv_caches[r_id]["V"] = v_user
            else:
                self.kv_caches[r_id]["K"] = torch.cat([self.kv_caches[r_id]["K"], k_user], dim=0)
                self.kv_caches[r_id]["V"] = torch.cat([self.kv_caches[r_id]["V"], v_user], dim=0)
            
            k_history = self.kv_caches[r_id]["K"]
            v_history = self.kv_caches[r_id]["V"]
            
            q_u = q_user.transpose(0, 1)
            k_h = k_history.transpose(0, 1)
            v_h = v_history.transpose(0, 1)
            
            scores = torch.matmul(q_u, k_h.transpose(-2, -1)) * self.softmax_scale
            attn_weights = F.softmax(scores, dim=-1)
            
            context_user = torch.matmul(attn_weights, v_h)
            context_user = context_user.transpose(0, 1).contiguous().view(-1, self.embed_dim)
            
            attention_outputs.append(context_user)
        
        O_batch_2d = torch.cat(attention_outputs, dim=0)
        
        final_output = torch.matmul(O_batch_2d, self.W_out)
        
        # UPDATE STATUS (SIMULATION)
        new_active_requests = {}
        
        for i, r_id in enumerate(req_ids):
            if r_id == "Req_A" and self.active_request[r_id]["phase"] == "decode":
                print(f"[RELEASE] Request '{r_id}' has finished/hit EOS. Remove from Batch!")
                del self.kv_caches[r_id]
                continue
            
            dummy_next_token = torch.randn(1, self.embed_dim)
            new_active_requests[r_id] = {
                "tokens": dummy_next_token,
                "phase": "decode"
            }
            print(f"[UPDATE] Request '{r_id}' success generate 1 token. Status: DECODE.")
        
        self.active_request = new_active_requests

if __name__ == "__main__":
    server = ContinuousBatching()
    
    print("--- ITERATIONS 1 ---")
    server.inject_request("Req_A", torch.randn(2, 64))
    server.inject_request("Req_B", torch.randn(4, 64))
    
    server.step()
    
    print("\n--- ITERATION 2 ---")
    server.inject_request("Req_C", torch.randn(3, 64))
    
    server.step()
    
    print("\n--- ITERASI 3 ---")
    server.step()