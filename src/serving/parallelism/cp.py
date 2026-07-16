'''
Build Context Parallelism simulation implementation
'''

import torch

class DummyGPU:
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        # {id: tensor}
        self.k_cache = torch.Tensor()
        self.v_cache = torch.Tensor()
        self.q_cache = torch.Tensor()
        self.current_k_buffer = torch.Tensor()
        self.current_v_buffer = torch.Tensor()
        self.output_list = []
    
    def __repr__(self) -> str:
        return (f"[GPU-{self.gpu_id}] - K Cache {self.k_cache.shape} & V Cache {self.v_cache.shape}")

class CP:
    
    def __init__(self, num_gpus: int, emb_dim: int):
        self.num_gpus = num_gpus
        self.gpus = [DummyGPU(gpu_id=i) for i in range(num_gpus)]
        
        self.query = torch.randn(emb_dim, emb_dim)
        self.key = torch.randn(emb_dim, emb_dim)
        self.value = torch.randn(emb_dim, emb_dim)
    
    def forward(self, x: torch.Tensor):
        _, T, _ = x.shape # (B, T, C)
        assert T % self.num_gpus == 0, f'sequence length must be divisible with {self.num_gpus} num_gpus'
        
        timestamp = self.num_gpus
        chunk_size = T // self.num_gpus
        
        micro_seq = [
            x[:, i:i+chunk_size, :]
            for i in range(0, T, chunk_size)
        ]
        
        print(f"--- TIMESTEP 0: Fill KV Cache Local and Computation Early ---")
        for gpu in self.gpus:
            micro_x = micro_seq[gpu.gpu_id]
            Q, K, V = torch.matmul(micro_x, self.query), torch.matmul(micro_x, self.key), torch.matmul(micro_x, self.value)
            gpu.k_cache = K
            gpu.v_cache = V
            gpu.q_cache = Q
            # Naive attention, in practical case we masking with causal mask
            # Do softmax, dropout
            attn_score = torch.matmul(Q, K.transpose(-2, -1))
            output     = torch.matmul(attn_score, V)
            gpu.output_list.append(output)
            print(f"    [GPU-{gpu.gpu_id}] Finished compute token local itself")
        
        
        print("\n--- TIMESTEP 1 - FINISHED: Cycle Ring Attention ---")
        for gpu in self.gpus:
            gpu.current_k_buffer = gpu.k_cache.clone()
            gpu.current_v_buffer = gpu.v_cache.clone()
        
        # 3 gpu
        for step in range(1, self.num_gpus):
            print(f"\n[Cycle Ring Step-{step}]")
            
            next_k_buffers = [None] * self.num_gpus
            next_v_buffers = [None] * self.num_gpus
            
            for i in range(self.num_gpus):
                next_gpu_idx = (i + 1) % self.num_gpus
                next_k_buffers[next_gpu_idx] = self.gpus[i].current_k_buffer #type: ignore
                next_v_buffers[next_gpu_idx] = self.gpus[i].current_v_buffer #type: ignore
                # note: add print in here for debugging
            
            for i in range(self.num_gpus):
                self.gpus[i].current_k_buffer = next_k_buffers[i] #type: ignore
                self.gpus[i].current_v_buffer = next_v_buffers[i] #type: ignore

            for gpu in self.gpus:
                attn_score = torch.matmul(gpu.q_cache, gpu.current_k_buffer.transpose(-2, -1))
                output = torch.matmul(attn_score, gpu.current_v_buffer)
                gpu.output_list.append(output)
                print(f"    [GPU-{gpu.gpu_id}] Compute Q-Local with KV-Buffer")
                
        print("\n--- LAST PHASE: Accumulate Result (SUM) ---")
        result_each_gpus = {}
        for gpu in self.gpus:
            final_output = sum(gpu.output_list)
            result_each_gpus[gpu.gpu_id] = final_output
            print(f"  [GPU-{gpu.gpu_id}] Final Output Shape: {tuple(final_output.shape)}") #type: ignore

        return result_each_gpus
        
        # # Ring Attention
        # gpu_now = 0
        # while timestamp >= 0:
        #     K_now = self.gpus[gpu_now].k_cache
        #     V_now = self.gpus[gpu_now].v_cache
            
        #     for i, gpu in enumerate(self.gpus):
        #         if gpu_now == gpu.gpu_id:
        #             continue # naive skip
            
        #         attn_score = torch.matmul(gpu.q_cache, K_now.transpose(-2, -1))
        #         output     = torch.matmul(attn_score, V_now) 
        #         gpu.output[gpu_now] = output
            
        #     gpu_now += 1
        #     timestamp -= 1
        
        # result_each_gpus = {}
        # for gpu in self.gpus:
        #     list_tensor = [gpu.output[k] for k in sorted(gpu.output.keys())]
        #     merged_tensor = torch.stack(list_tensor)
        #     result_each_gpus[gpu.gpu_id] = merged_tensor

        # return result_each_gpus

if __name__ == "__main__":
    x = torch.randn(1, 8, 4)
    cp = CP(num_gpus=4, emb_dim=4)
    res = cp.forward(x)