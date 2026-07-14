'''
Build FSDP (Fully Sharded Data Parallelism) simulation implementations
'''

import copy

from numpy import full

class DummyGPU:
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.param_shard = {}
        self.grad_shard = {}
        self.optimize_state_shard = {}
    
    def __repr__(self):
        return f"GPU-{self.gpu_id}(shard_size={len(self.param_shard)})"
    
class FSDP:
    
    def __init__(self, model_params: dict, num_gpus: int):
        self.num_gpus = num_gpus
        self.gpus = [DummyGPU(gpu_id=i) for i in range(num_gpus)]
        
        param_names = list(model_params.keys())
        for i, name in enumerate(param_names):
            ow_gpu = self.gpus[i % self.num_gpus]
            ow_gpu.param_shard[name] = model_params[name]
            ow_gpu.optimize_state_shard[name] = 0.0
        
        print(f"[FSDP INIT] {num_gpus} GPU, {len(param_names)} params "
              f"SHARD (not full-copy)")
        
        for gpu in self.gpus:
            print(f"    {gpu}: handle {list(gpu.param_shard.keys())}")
    
    def all_gather_params(self):
        print(f"\n[ALL-GATHER] Gathering all shard temp for compute...")
        full_params = {}
        for gpu in self.gpus:
            full_params.update(gpu.param_shard)
            
        print(f"[ALL-GATHER] Full params temp assembly: {full_params}")
        return full_params

    def forward_backward(self, full_params: dict, micro_batch_grad: dict):
        print(f"    Compute forward-backward use full params (temp)...")
        return micro_batch_grad
    
    def reduce_scatter_gradients(self, full_grad_per_gpu: list):
        print(f"[REDUCE-SCATTER] Average gradient between GPU, "
              f"DIRECTLY broken down per shard (not BC full)")
        
        param_names = full_grad_per_gpu[0].keys() 
        avg_grads = {}
        for name in param_names:
            total = sum(g[name] for g in full_grad_per_gpu)
            avg_grads[name] = total / self.num_gpus
        
        for gpu in self.gpus:
            gpu.grad_shard = {
                name: avg_grads[name]
                for name in gpu.param_shard
            }
            print(f"    [GPU-{gpu.gpu_id}] only save grad shard: {gpu.grad_shard}")
        
        return avg_grads

    def optimizer_step(self, lr: float = 0.1):
        for gpu in self.gpus:
            for name in gpu.param_shard:
                gpu.param_shard[name] -= lr * gpu.grad_shard[name]
        
        print(f"[OPTIMIZER-STEP] Each GPU update shard itself")
        for gpu in self.gpus:
            print(f"    {gpu}: {gpu.param_shard}")
    
    def free_full_param(self, full_param: dict):
        del full_param
        print(f"[FREE] Full params temprorary drops from memory GPU")