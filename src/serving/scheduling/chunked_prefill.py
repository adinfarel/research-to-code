'''
Build Chunked Prefill implementation
'''

import torch
import torch.nn.functional as F

class ChunkedPrefillBatching:
    
    def __init__(self, embed_dim=64, chunk_size=2):
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        
        self.W_qkv = torch.randn(embed_dim, 3 * embed_dim)
        
        self.active_requests = {}
        self.waiting_prefill_queue = {}
        self.kv_caches = {}
    
    def inject_request(self, req_id, all_prompt_tokens):
        self.kv_caches[req_id] = {"K": None, "V": None}
        
        if all_prompt_tokens.size(0) > self.chunk_size:
            self.active_requests[req_id] = {
                "tokens": all_prompt_tokens[:self.chunk_size],
                "phase": "chunked_prefill",
            }
        
            self.waiting_prefill_queue[req_id] = all_prompt_tokens[self.chunk_size:]
            print(f"[INJECT & CHUNK] '{req_id}' take {all_prompt_tokens.size(0)} token. "
                  f"Into it {self.chunk_size} first token. Remain {all_prompt_tokens.size(0) - self.chunk_size} queue token.")
        
        else:
            self.active_requests[req_id] = {
                "tokens": all_prompt_tokens,
                "phase": "prefill"
            }
            print(f"[INJECT] '{req_id}' take {all_prompt_tokens.size(0)} token. Directly put in full prefill.")
    
    def step(self):
        if not self.active_requests:
            print(f"SERVER IDLE, NOT HAVE REQUEST")
            return

        req_ids = list(self.active_requests.keys())
        
        flattened_tokens_list = []
        cu_seqlens = [0]
        curr_offset = 0
        
        for r_id in req_ids:
            req = self.active_requests[r_id]
            flattened_tokens_list.append(req['tokens'])
            
            num_tokens = req['tokens'].size(0)
            curr_offset += num_tokens
            cu_seqlens.append(curr_offset)
        
        X_batch_2d = torch.cat(flattened_tokens_list, dim=0)
        print(f"\n[ITERATION] Processing {X_batch_2d.size(0)} token global. cu_seqlens: {cu_seqlens}")
        
        qkv = torch.matmul(X_batch_2d, self.W_qkv)
        Q, K, V = torch.chunk(qkv, chunks=3, dim=-1)
        
        for i, r_id in enumerate(req_ids):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i+1]
            
            k_user = K[start_idx:end_idx]
            v_user = V[start_idx:end_idx]
            
            if self.kv_caches[r_id]["K"] is None:
                self.kv_caches[r_id]["K"] = k_user
                self.kv_caches[r_id]["V"] = v_user
            else:
                self.kv_caches[r_id]["K"] = torch.cat([self.kv_caches[r_id]["K"], k_user], dim=0)
                self.kv_caches[r_id]["V"] = torch.cat([self.kv_caches[r_id]["V"], v_user], dim=0)
        
        next_active_requests = {}
        
        for r_id in req_ids:
            current_phase = self.active_requests[r_id]["phase"]
            
            if current_phase == "chunked_prefill":
                prompt_left = self.waiting_prefill_queue[r_id]
                
                if prompt_left.size(0) > self.chunk_size:
                    next_active_requests[r_id] = {
                        "tokens": prompt_left[:self.chunk_size],
                        "phase": "chunked_prefill",
                    }
                    self.waiting_prefill_queue[r_id] = prompt_left[self.chunk_size:]
                    print(f"[CHUNKING] '{r_id}' continue chunked {self.chunk_size} token more.")
                
                else:
                    next_active_requests[r_id] = {
                        "tokens": prompt_left,
                        "phase": "chunked_prefill_last" 
                    }
                    del self.waiting_prefill_queue[r_id]
                    print(f"[CHUNKING] '{r_id}' take remaining {prompt_left.size(0)} last token from queue prefill.")
            
            elif current_phase in ["prefill", "chunked_prefill_last", "decode"]:
                dummy_next_token = torch.randn(1, self.embed_dim)
                next_active_requests[r_id] = {
                    "tokens": dummy_next_token,
                    "phase": "decode"
                }
                print(f"[DECODE] '{r_id}' running in phase Decode.")
                
        self.active_requests = next_active_requests

if __name__ == "__main__":
    server = ChunkedPrefillBatching(embed_dim=64, chunk_size=2)
    
    print("--- ITERASI 1 (Req A Decoding, Req B New Inject) ---")
    server.active_requests["Req_A"] = {"tokens": torch.randn(1, 64), "phase": "decode"}
    server.kv_caches["Req_A"] = {"K": torch.randn(1, 64), "V":torch.randn(1, 64)}
    server.inject_request("Req_B", torch.randn(5, 64))
    
    print("\n--- ITERASI 2 (Second Chunked Req B) ---")
    server.step()

    print("\n--- ITERASI 3 (Last Chunked Req B) ---")
    server.step()
    
    print("\n--- ITERASI 4 (All Request Decode Phase) ---")
    server.step()