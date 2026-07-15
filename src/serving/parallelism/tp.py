'''
Build Tensor Parallelism (TP) simulation implementation
'''

import torch
import torch.nn as nn

class DummyGPU:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.w_col_shard = None
        self.w_row_shard = None
    
    def __repr__(self):
        return (f"GPU-{self.gpu_id} | "
                f"Col Shard Shape: {tuple(self.w_col_shard.shape)} | " #type: ignore
                f"Row Shard Shape: {tuple(self.w_row_shard.shape)}") #type: ignore

class TP:
    def __init__(self, in_features: int, out_features: int, num_gpus: int):
        assert in_features % num_gpus == 0, "in_features must be divisible with num_gpus"
        assert out_features % num_gpus == 0, "out_features must be divisible with num_gpus"
        self.num_gpus = num_gpus
        self.gpus = [DummyGPU(gpu_id=i) for i in range(num_gpus)]
        
        full_w_col = torch.randn(in_features, out_features)
        full_w_row = torch.randn(out_features, in_features)
        
        print(f"[TP INIT] Slicing Model Vertical to {num_gpus} GPU:")
        print(f"  Full Linear-1 (Col): {tuple(full_w_col.shape)}")
        print(f"  Full Linear-2 (Row): {tuple(full_w_row.shape)}\n")
        
        col_shard_size = out_features // num_gpus
        row_shard_size = out_features // num_gpus
        
        for i, gpu in enumerate(self.gpus):
            gpu.w_col_shard = full_w_col[:, i * col_shard_size : (i + 1) * col_shard_size] #type: ignore
            gpu.w_row_shard = full_w_row[i * row_shard_size : (i + 1) * row_shard_size, :] #type: ignore
            print(f"    {gpu}")
    
    def run(self, x: torch.Tensor):
        print(f"\n--- FORWARD PASS (Tensor Parallel: Col -> Row Combo) ---")
        print(f"Input X (Full) put in all GPU. Shape: {tuple(x.shape)}")
        
        print("\n[STEP 1: Column Parallel]")
        outputs_col = []
        for gpu in self.gpus:
            y_local = torch.matmul(x, gpu.w_col_shard) #type: ignore
            outputs_col.append(y_local)
            print(f"    [GPU-{gpu.gpu_id}] Compute local col product -> Output Shape: {tuple(y_local.shape)}")
        
        print(f"\n[STEP 2: Row Parallel]")
        outputs_row = []
        for i, gpu in enumerate(self.gpus):
            y_in = outputs_col[i]
            z_local = torch.matmul(y_in, gpu.w_row_shard) #type: ignore
            outputs_row.append(z_local)
            print(f"    [GPU-{gpu.gpu_id}] Compute local row product -> Output Shape: {tuple(z_local.shape)}")
        
        print("\n[STEP 3: All Reduce (Sum) Last Layer]")
        final_output = sum(outputs_row)
        print(f"  [All-Reduce Finished] Output successfully merging and synchronize.")
        print(f"  Final Output Shape: {tuple(final_output.shape)}") #type: ignore

if __name__ == "__main__":
    x = torch.randn(4, 8) 

    tp = TP(in_features=8, out_features=4, num_gpus=2)
    tp.run(x)