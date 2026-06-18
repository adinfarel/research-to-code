'''
Build vector_add straight from Kernel

Here my intuition about GPU:
  --> Kernel is small function that running by GPU in parallel
  
  Component GPU, why different with CPU?
  --> CPU have a small core but so smart, cause in 1 core CPU have a branch predictor, reduce latency, can run many task at the same time, etc.
  --> GPU have a many core but is so dumb, if core CPU have 5 GHz+ then GPU just 1,5 GHz+, but GPU can running 1 task in large scale
      so, that's way GPU called SIMD (Single Instruction, Multiple Data)
  
  Hierarchy GPU:
  VRAM -> Large Memory of GPU
  L2 - Cache --> Shared Memory each SM (Streaming Multiprocessor)
  L1 - Cache --> 1 Memory each 1 SM
  Block --> In term of GPU, block has contains many threads, Example: 1024 Thread in 1 Block; and Block can have more than 1
  Core --> Place of thread for do something (Operations)
  Control Unit (CU) -->  Instructor that moving each threads to do task given kernel, 1 SM just have 1 CU so that's way we cannot doing many task in 1 SM
          Problem: Because cant have many task in 1 SM this problem has the name Thread Divergence
          Solution: Must be accept that cant how to do about that, cause if add new CU each Warps so expensive (1 Warps = 32 Threads)
  Threads --> Worker that working in Core
  Registry --> Memory owned Threads
  
  Example:
  Have 10 SM, so have 10 CU and 10 L1 - Cache
  each SM consists 128 Core, then have 2 block, each block 64 threads (64/32 = 2 Warps)
  each Block have 64 core that running at the same time
      
  How actually computer running the code?
  --> CPU compile the code and looking for kernel the right one for this code (Dispatch Kernel), after found kernel, CPU launch kernel with 
      store data into VRAM GPU, then in (assume 1 SM) CU take an instructor from kernel what have to do, and CU informed each thread what have to do
      example add operations (each Threads doing adding in each value), after compute that Threads store to Registry --> L1 - Cache --> L2 Cache --> VRAM
  NOTE: 1 Threads doing 1 Operations (Exam: a + a = 2a)

NOTE: That my intuition, if there is mistake or mis-information, don't hesitate to tell me >.<
'''

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0) # (P_id, 1) 
    
    block_start = pid * BLOCK_SIZE # (P_id, 1) * BLOCK_SIZE (1024)
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # (P_id, 1) + (Block_size,) = (P_id, Block_size)
    # [[P_id=0 --> 1024]
    # [P_id=1 --> 1024], etc]
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Output[i] = x[i] + y[i]
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024) #type: ignore
    
    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    
    output_torch = x + y
    output_triton = add(x + y) #type: ignore
    
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')