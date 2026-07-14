'''
Build DDP (Distributed Data Parallel) simulation implementations
'''

import copy

class DummyGPU:
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.params = {}
        self.grads = {}
        self.optimizer_state = {}
    
    def __repr__(self):
        return f"GPU-{self.gpu_id}(params={len(self.params)}, mem_used={self._estimate_mem()})"
    
    def _estimate_mem(self):
        return len(self.params) + len(self.grads) + len(self.optimizer_state)

class DDP:
    
    def __init__(self, model_params: dict, num_gpus: int = 4) -> None:
        self.num_gpus = num_gpus
        self.gpus = [DummyGPU(gpu_id=i) for i in range(num_gpus)]
    
        for gpu in self.gpus:
            gpu.params = copy.deepcopy(model_params)
            gpu.optimizer_state = {k: 0.0 for k in model_params}
        
        print(f"[DDP INIT] {num_gpus} each GPU get FULL COPY model "
              f"({len(model_params)} params).")
    
    def forward_backward(self, gpu_id: int, micro_batch_grad: dict):
        gpu = self.gpus[gpu_id]
        gpu.grads = copy.deepcopy(micro_batch_grad)
        print(f"[GPU-{gpu_id}] Forward-backward finished (independent, not waiting other GPU). "
              f"Local grad: {gpu.grads}")
    
    def all_reduce_gradients(self):
        print(f"\n[ALL-REDUCE] Gathering gradient from all CPU...")
        
        param_names = self.gpus[0].grads.keys()
        averaged_grads = {}
        
        for name in param_names:
            total = sum(gpu.grads[name] for gpu in self.gpus)
            averaged_grads[name] = total / self.num_gpus
        
        for gpu in self.gpus:
            gpu.grads = copy.deepcopy(averaged_grads)
        
        print(f"[ALL-REDUCE] Finished. All GPU now have gradient "
              f"IDENTICAL: {averaged_grads}")
        return averaged_grads

    def optimizer_step(self, lr: float = 0.1):
        for gpu in self.gpus:
            for name in gpu.params:
                gpu.params[name] -= lr * gpu.grads[name]

        
        print(f"[OPTIMIZER STEP] All GPU update weight. "
              f"GPU-0 params after update: {self.gpus[0].params}")