'''
Build Pipeline Parallelism (PP) simulation implementation
'''

import torch
import torch.nn as nn

class DummyGPU:
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.layers = None
    
    def __repr__(self):
        layer = {}
        
        assert self.layers is not None
        
        for i, lay in enumerate(self.layers):
            layer[i] = lay
        
        return f"GPU-{self.gpu_id}(layers:\n{layer})"
    
    def __call__(self, x: torch.Tensor):
        for layer in self.layers: #type: ignore
            x = layer(x)
        return x

class PP:
    
    def __init__(self, layers: nn.ModuleList, num_gpus: int, micro_batch_size: int = 4):
        assert len(layers) % num_gpus == 0, "for simplicity use symmetrical distribution"
        # NOTE: in practical we dealing with Embedding Layer and LM Head
        # so first GPU and last GPU must be have layers less than other layers
        # but for simulation, just use symmetrical distribution
        
        self.num_gpus = num_gpus
        self.gpus = [DummyGPU(gpu_id=i) for i in range(num_gpus)]
        self.block_layer = len(layers) // num_gpus
        self.micro_batch_size = micro_batch_size
        
        i = 0
        for gpu in self.gpus:
            gpu.layers = layers[i:i+self.block_layer] #type: ignore
            i += self.block_layer
        
        for gpu in self.gpus:
            print(gpu)
    
    def run(self, global_batch: torch.Tensor, loss_fn=None):
        assert global_batch.shape[0] % self.micro_batch_size == 0, "global batch must divisible by micro_batch_size"
        
        microbatches = [
            global_batch[i:i + self.micro_batch_size]
            for i in range(0, global_batch.shape[0], self.micro_batch_size)
        ]
        n_micro = len(microbatches)
        n_stages = self.num_gpus
        
        forward_act = {}
        input_per_stage = {}
        
        print(f"--- FORWARD PASS (staggered, GPipe schedule not 1F1B) ---")
        total_fwd_steps = n_micro + n_stages - 1 # total compute all batch
        
        # GPU = 2
        # batch = 6, size = 2
        # total micro batch = 3
        for t in range(total_fwd_steps):
            print(f"\n--- Timestep {t} ---")
            for stage_idx, gpu in enumerate(self.gpus):
                micro_idx = t - stage_idx
                
                if 0 <= micro_idx < n_micro:
                    if stage_idx == 0:
                        x_in = microbatches[micro_idx]
                    else:
                        x_in = forward_act[(micro_idx, stage_idx - 1)]
                    
                    x_in = x_in.clone().detach().requires_grad_(True)
                    input_per_stage[(micro_idx, stage_idx)] = x_in
                    
                    x_out = gpu(x_in)
                    forward_act[(micro_idx, stage_idx)] = x_out
                    
                    print(f"    [GPU-{stage_idx}] forward microbatch {micro_idx} "
                          f"-> shape {tuple(x_out.shape)}")

                else:
                    print(f"    [GPU-{stage_idx}] IDLE (bubble)")
        
        if loss_fn is None:
            loss_fn = lambda out: out.pow(2).mean()
        
        losses = {
            micro_idx: loss_fn(forward_act[(micro_idx, n_stages - 1)])
            for micro_idx in range(n_micro)
        }
        
        print("\n--- BACKWARD PASS (staggered, reverse order) ---")
        grads_out = {}
        total_bwd_steps = n_micro + n_stages - 1
        
        for t in range(total_bwd_steps):
            print(f"\n--- Timestep {t} ---")
            for stage_idx, gpu in enumerate(self.gpus):
                offset = (n_stages - 1) - stage_idx
                micro_idx = t - offset
                
                if 0 <= micro_idx < n_micro:
                    in_tensor = input_per_stage[(micro_idx, stage_idx)]
                    
                    if stage_idx == n_stages - 1:
                        losses[micro_idx].backward()
                    else:
                        grad_from_next = grads_out[(micro_idx, stage_idx + 1)]
                        forward_act[((micro_idx, stage_idx))].backward(gradient=grad_from_next)
                    
                    grads_out[(micro_idx, stage_idx)] = in_tensor.grad
                    
                    print(f"  [GPU-{stage_idx}] backward microbatch {micro_idx}")
                else:
                    print(f"  [GPU-{stage_idx}] IDLE (bubble)")
        
        total_loss = sum(l.item() for l in losses.values())
        print(f"\n--- DONE. Total loss all microbatches: {total_loss:.4f} ---")
        return losses
    
        # last_layer = False
        # for i in range(0, len(global_batch), self.micro_batch_size):
        #     micro_batch = global_batch[i:i+self.micro_batch_size]
        #     x = micro_batch.clone()
            
        #     for gpu in self.gpus:
        #         x = gpu(x)

        #         if gpu.gpu_id + 1 == self.num_gpus:
        #             last_layer = True
        #             pass

if __name__ == "__main__":
    layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])  # 4 layer, dummy
    pp = PP(layers, num_gpus=2, micro_batch_size=2)

    global_batch = torch.randn(10, 8)  
    pp.run(global_batch)