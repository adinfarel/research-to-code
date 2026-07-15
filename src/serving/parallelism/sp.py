'''
Build Sequence Parallelism 
'''

import torch
import torch.nn as nn

class DummyGPU:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        
        self.layer_norm = nn.LayerNorm(8)
        self.w_col_shard = None
    
    def __repr__(self) -> str:
        return f"GPU-{self.gpu_id} | LayerNorm Local Active"

class SP:
    def __init__(self, num_gpus: int, hidden_dim: int = 8):
        self.num_gpus = num_gpus
        self.gpus = [DummyGPU(gpu_id=i) for i in range(num_gpus)]
        self.hidden_dim = hidden_dim
        
        full_w_col = torch.randn(hidden_dim, hidden_dim)
        col_shard_size = hidden_dim // num_gpus
        
        for i, gpu in enumerate(self.gpus):
            gpu.w_col_shard = full_w_col[:, i * col_shard_size : (i + 1) * col_shard_size] #type: ignore
    
    def run(self, global_sequence_data: torch.Tensor):
        seq_len = global_sequence_data.shape[1]
        assert seq_len % self.num_gpus == 0, "seq_len must be divisble with num_gpus" 
        
        print(f"[SP INIT] Global Input Sequence Length: {seq_len} token.")
        
        chunk_size = seq_len // self.num_gpus
        sp_shards = [
            global_sequence_data[:, i * chunk_size : (i + 1) * chunk_size]
            for i in range(self.num_gpus)
        ]
        
        print("\n--- STAGE 1: LayerNorm (Sequence Parallel) ---")
        ln_outputs = []
        for i, gpu in enumerate(self.gpus):
            local_seq = sp_shards[i]
            local_out = gpu.layer_norm(local_seq)
            ln_outputs.append(local_out)
            print(f"  [GPU-{gpu.gpu_id}] Calculating LayerNorm for token [{i*chunk_size} s/d {(i+1)*chunk_size-1}] "
                  f"| Activation Shape: {tuple(local_out.shape)}")
        
        print(f"\n--- STAGE 2: All-Gather (Dimension Sequence) ---")
        gathered_seq = torch.cat(ln_outputs, dim=0)
        print(f"  [All-Gather Finished] Token merge back fully.")
        print(f"  Input Shape for Linear (TP): {tuple(gathered_seq.shape)}")
        
        print("\n--- STAGE 3: Column Linear (Tensor Parallel) ---")
        col_outputs = []
        for gpu in self.gpus:
            x_flat = gathered_seq.squeeze(0)
            y_local = torch.matmul(x_flat, gpu.w_col_shard) #type: ignore
            col_outputs.append(y_local.unsqueeze(0))
            print(f"  [GPU-{gpu.gpu_id}] Compute Linear-Col -> Output Shape: {tuple(col_outputs[-1].shape)}")

        print("\n--- STAGE 4: Reduce-Scatter (Back into Shard Sequence) ---")
        
        reduced_total = sum(col_outputs)
        
        final_sp_shards = []
        scatter_chunk = seq_len // self.num_gpus
        for i, gpu in enumerate(self.gpus):
            gpu_final_shard = reduced_total[i * scatter_chunk : (i + 1) * scatter_chunk] #type: ignore
            final_sp_shards.append(gpu_final_shard)
            print(f"  [GPU-{gpu.gpu_id}] Accept again shard result token Reduce-Scatter -> Shape: {tuple(gpu_final_shard.shape)}")
            
        print(f"\n--- DONE. Sequence Parallel ---")
        return final_sp_shards

if __name__ == "__main__":
    global_data = torch.randn(1, 8, 8)
    
    sp = SP(num_gpus=2, hidden_dim=8)
    sp.run(global_data)