'''
Build KIVI Quantization implementation
'''

import torch

class KIVIVolatilQuant:
    
    def __init__(self, bits: int = 2, residual_window_size: int = 2) -> None:
        self.bits = bits
        self.residual_window_size = residual_window_size
        
        self.qmin = 0
        self.qmax = 2 ** bits - 1
    
    def _quantize_per_channel(self, tensor: torch.Tensor):
        alpha = torch.max(tensor, dim=2, keepdim=True)[0]
        beta = torch.min(tensor, dim=2, keepdim=True)[0]
        
        scale = (alpha - beta) / (self.qmax - self.qmin)
        scale = torch.clamp(scale, min=1e-5)
        
        zero_point = self.qmin - torch.round(beta / scale)
        zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
        
        quantized = torch.round(tensor / scale) + zero_point
        quantized = torch.clamp(quantized, self.qmin, self.qmax).to(torch.uint8)
        
        return quantized, scale, zero_point
    
    def _quantize_per_token(self, tensor: torch.Tensor):
        alpha = torch.max(tensor, dim=3, keepdim=True)[0]
        beta = torch.min(tensor, dim=3, keepdim=True)[0]
        
        scale = (alpha - beta) / (self.qmax - self.qmin)
        scale = torch.clamp(scale, min=1e-5)
        
        zero_point = self.qmin - torch.round(beta / scale)
        zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
        
        quantized = torch.round(tensor / scale) + zero_point
        quantized = torch.clamp(quantized, self.qmin, self.qmax).to(torch.uint8)
        
        return quantized, scale, zero_point

    def compress_kv_cache(self, key_cache: torch.Tensor, value_cache: torch.Tensor):
        seq_len = key_cache.shape[2]
        
        if seq_len <= self.residual_window_size:
            return {
                "is_full_fp16": True,
                "key_raw": key_cache,
                "value_raw": value_cache
            }
        
        split_idx = seq_len - self.residual_window_size
        
        k_past = key_cache[:, :, :split_idx, :]
        v_past = value_cache[:, :, :split_idx, :]
        
        k_recent = key_cache[:, :, split_idx:, :]
        v_recent = value_cache[:, :, split_idx:, :]
        
        k_quant, k_scale, k_zp = self._quantize_per_channel(k_past)
        v_quant, v_scale, v_zp = self._quantize_per_token(v_past)
        
        return {
            "is_full_fp16": False,
            "k_quant": k_quant, "k_scale": k_scale, "k_zp": k_zp,
            "v_quant": v_quant, "v_scale": v_scale, "v_zp": v_zp,
            "k_recent": k_recent,
            "v_recent": v_recent
        }
    
    def dequantize_kv_cache(self, compressed_bundle: dict):
        if compressed_bundle["is_full_fp16"]:
            return compressed_bundle["key_raw"], compressed_bundle["value_raw"]
        
        k_quant, k_scale, k_zp = compressed_bundle["k_quant"], compressed_bundle["k_scale"], compressed_bundle["k_zp"]
        v_quant, v_scale, v_zp = compressed_bundle["v_quant"], compressed_bundle["v_scale"], compressed_bundle["v_zp"]
        k_recent = compressed_bundle["k_recent"]
        v_recent = compressed_bundle["v_recent"]
        
        k_past_dq = k_scale * (k_quant.to(torch.float32) - k_zp)
        v_past_dq = v_scale * (v_quant.to(torch.float32) - v_zp)
        
        k_restored = torch.cat([k_past_dq, k_recent], dim=2)
        v_restored = torch.cat([v_past_dq, v_recent], dim=2)
        
        return k_restored, v_restored