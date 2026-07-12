'''
Build StreamingLLM implementation
'''

import torch
import torch.nn as nn

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, emb_dim, base=10000.0):
        super().__init__()
        self.dim = emb_dim
        
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def _rotate_half(self, x:torch.Tensor):
        C = x.shape[-1]
        xu, xd = x[..., : C // 2], x[..., C // 2 :]
        return torch.cat([-xd, xu], dim=-1)
        # [[-3theta0, -3theta1, 3theta0, 3theta1]
        #  [-4theta0, -4theta1, 4theta0, 4theta1]]

    def apply_rotation(self, x: torch.Tensor, angle_rotations: torch.Tensor):
        # angle: [3, 4]
        theta = torch.einsum("...i,j->...ij", angle_rotations.float(), self.inv_freq) # outer i * j = [[i -> j]
        # [3, 4] * [theta0, theta1]
        # [[3theta0, 3theta1]
        #  [4theta0, 4theta1]]
        
        hat_theta = torch.cat([theta, theta], dim=-1)
        # [[3theta0, 3theta1, 3theta0, 3theta1]
        #  [4theta0, 4theta1, 4theta0, 4theta1]]
        
        sin = torch.sin(hat_theta)
        cos = torch.cos(hat_theta)
        return x * cos + self._rotate_half(x) * sin
    
    def forward(self, x:torch.Tensor, custom_positions=None):
        T = x.shape[-2]
        if custom_positions is None:
            l = torch.arange(T, dtype=self.inv_freq.dtype, device=x.device) #type: ignore
        else:
            l = custom_positions.to(dtype=self.inv_freq.dtype, device=x.device)
        
        return self.apply_rotation(x, l)

class StreamingLLM(nn.Module):
    def __init__(self, emb_dim, max_context_len, num_sink_tokens=4):
        super().__init__()
        self.num_sink_tokens = num_sink_tokens
        self.max_context_len = max_context_len
        
        self.rope = RotaryPositionalEncoding(emb_dim)
        
        self.k_cache = None
        self.v_cache = None
        
        self.curr_log_len = 0
    
    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.curr_log_len = 0
    
    def update_cache(self, new_k, new_v):
        B, H, T_new, D = new_k.shape
        
        if self.k_cache is None:
            positions = torch.arange(T_new, device=new_k.device)
            self.k_cache = self.rope(new_k, custom_positions=positions)
            self.v_cache = new_v
            self.curr_log_len = T_new
            return self.k_cache, self.v_cache

        T_current = self.k_cache.shape[2]
        
        if T_current + T_new <= self.max_context_len:
            positions = torch.arange(self.curr_log_len, self.curr_log_len + T_new)
            new_k_rope = self.rope(new_k, custom_positions=positions)
            
            self.k_cache = torch.cat([self.k_cache, new_k_rope], dim=2)
            self.v_cache = torch.cat([self.v_cache, new_v], dim=2) #type: ignore
            self.curr_log_len += T_new
        
        else:
            num_tokens_to_evict = (T_current + T_new) - self.max_context_len
            
            sink_k = self.k_cache[:, :, :self.num_sink_tokens, :]
            sink_v = self.v_cache[:, :, :self.num_sink_tokens, :] #type: ignore
            
            start_idx = self.num_sink_tokens + num_tokens_to_evict
            sliding_k = self.k_cache[:, :, start_idx:, :]
            sliding_v = self.v_cache[:, :, start_idx:, :] #type: ignore
            
            num_survivors = sliding_k.shape[2]
            delta = torch.full(
                (num_survivors, ), -float(num_tokens_to_evict), device=new_k.device
            )
            sliding_k_shifted = self.rope.apply_rotation(sliding_k, delta)
            
            new_start_pos = self.num_sink_tokens + num_survivors
            new_positions = torch.arange(
                new_start_pos, new_start_pos + T_new, device=new_k.device
            )
            new_k_rope = self.rope(new_k, custom_positions=new_positions)
            
            self.k_cache = torch.cat([sink_k, sliding_k_shifted, new_k_rope], dim=2)
            self.v_cache = torch.cat([sink_v, sliding_v, new_v], dim=2)
            self.curr_log_len = new_start_pos + T_new
            
        return self.k_cache, self.v_cache

if __name__ == "__main__":
    # Setup dimensi dummy mirip LLM asli
    batch_size = 1
    num_heads = 4
    head_dim = 64
    max_len = 10       # Kita set kecil agar cepat memicu efek pembatasan konteks
    sink_len = 2       # Kita set 2 token awal sebagai Attention Sink
    
    # Inisialisasi Mesin Manifold Cache
    stream_engine = StreamingLLM(emb_dim=head_dim, max_context_len=max_len, num_sink_tokens=sink_len)
    
    print(f"--- 🚀 MEMULAI SIMULASI STREAMING LLM (Max Context Window = {max_len}) ---")
    
    # LANGKAH 1: Tahap Prefill (Memasukkan 6 token awal sekaligus)
    k_init = torch.randn(batch_size, num_heads, 6, head_dim)
    v_init = torch.randn(batch_size, num_heads, 6, head_dim)
    k_c, v_c = stream_engine.update_cache(k_init, v_init)
    print(f"[Prefill] Masuk 6 token. Ukuran Cache saat ini: {k_c.shape[2]} token. (Aman, Belum Penuh)")
    
    # LANGKAH 2: Masukan bertahap token baru (Append 3 token baru lagi)
    k_new = torch.randn(batch_size, num_heads, 3, head_dim)
    v_new = torch.randn(batch_size, num_heads, 3, head_dim)
    k_c, v_c = stream_engine.update_cache(k_new, v_new)
    print(f"[Append 1] Masuk 3 token lagi. Ukuran Cache saat ini: {k_c.shape[2]} token. (Total 9, Hampir Penuh)")
    
    # LANGKAH 3: CRITICAL POINT! Masuk 4 token baru (Total teoritis 9 + 4 = 13 token)
    # Ini akan memaksa sistem melakukan SLICING membuang 3 token, tapi mempertahankan SINK dan melakukan REMAPPING.
    k_overflow = torch.randn(batch_size, num_heads, 4, head_dim)
    v_overflow = torch.randn(batch_size, num_heads, 4, head_dim)
    k_c, v_c = stream_engine.update_cache(k_overflow, v_overflow)
    
    print("\n--- 🚨 MELEBIHI MAX CONTEXT LENGTH (SLICING ACTION TRIGGERED) ---")
    print(f"[Result] Ukuran akhir KV Cache sukses dikunci di angka: {k_c.shape[2]} token!")
    print(f"-> VRAM tidak bengkak ke angka 13, melainkan dipangkas pas di batas maksimal {max_len}.")
    print(f"-> Di dalamnya, {sink_len} token pertama abadi tidak terbuang sebagai Attention Sink.")
    print("-> Sisa sudut token di tengah sukses di-remapping mengikuti pergeseran koordinat barunya.")