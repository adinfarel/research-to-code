'''
Build TurboQuant implementation
'''

import torch
import torch.nn as nn

class TurboQuant:
    
    def __init__(self, bits: int = 3):
        self.bits = bits
        self.num_slots = 2 ** bits
        
    # POLAR QUANT (ROTATE GEOMETRY)
    def _generate_random_orthogonal_matrix(self, dim: int, device: torch.device):
        X = torch.randn(dim, dim, device=device)
        # QR Decomposition
        Q, R = torch.linalg.qr(X)
        return Q
    
    def polar_transform(self, params: torch.Tensor):
        device = params.device
        
        dim = params.shape[-1]
        
        Q_matrix = self._generate_random_orthogonal_matrix(dim, device)
        
        rotated_params = torch.matmul(params, Q_matrix)
        
        return rotated_params, Q_matrix # save q_matrix for dequant
    
    # LLOYD-MAX QUANTIZER (NON-UNIFORM)
    def lloyd_max_fit(self, params: torch.Tensor, steps: int = 10):
        flat_data = params.flatten()
        min_val, max_val = flat_data.min(), flat_data.max()
        centroids = torch.linspace(min_val, max_val, self.num_slots, device=params.device)
        
        for _ in range(steps):
            # distances = |xi - ci|
            distances = torch.abs(flat_data.unsqueeze(-1) - centroids)
            
            assignments = torch.argmin(distances, dim=-1)
            
            new_centroids = centroids.clone()
            for i in range(self.num_slots):
                mask = (assignments == i)
                if mask.any():
                    new_centroids[i] = flat_data[mask].mean()
            centroids = new_centroids
        
        return centroids
    
    def lloyd_quantize(self, params: torch.Tensor, centroids: torch.Tensor):
        distances = torch.abs(params.unsqueeze(-1) - centroids)
        quantized_indices = torch.argmin(distances, dim=-1)
        return quantized_indices

    # QUANTIZED JOHNSON-LINDESTRAUSS (QJL)
    def qjl_compute_residual_bit(self, q_original: torch.Tensor, k_original: torch.Tensor,
                                 k_quant_dequant: torch.Tensor):
        score_original = torch.matmul(q_original, k_original.transpose(-1, -2))
        
        score_quant = torch.matmul(q_original, k_quant_dequant.transpose(-1, -2))
        
        inner_product_error = score_original - score_quant
        
        residual_bit = torch.sign(inner_product_error)
        
        error_scale = torch.mean(torch.abs(inner_product_error))
        
        return residual_bit, error_scale
    
    def fit_and_compress(self, query: torch.Tensor, key: torch.Tensor):
        rotated_key, Q_matrix = self.polar_transform(key)
        
        centroids = self.lloyd_max_fit(rotated_key, steps=10)
        quantized_indices = self.lloyd_quantize(rotated_key, centroids)
        
        dequant_rotated_key = centroids[quantized_indices]
        k_dequant = torch.matmul(dequant_rotated_key, Q_matrix.t())
        
        residual_bit, error_scale = self.qjl_compute_residual_bit(query, key, k_dequant)
        
        return {
            "quantized_indices": quantized_indices,  
            "centroids": centroids,                
            "Q_matrix": Q_matrix,                  
            "residual_bit": residual_bit,            
            "error_scale": error_scale               
        }
        
    def dequantize_and_compute_attention(self, query: torch.Tensor, compressed_bundle: dict):
        quant_idx = compressed_bundle["quantized_indices"]
        centroids = compressed_bundle["centroids"]
        Q_matrix = compressed_bundle["Q_matrix"]
        residual_bit = compressed_bundle["residual_bit"]
        error_scale = compressed_bundle["error_scale"]
        
        dq_rotated_key = centroids[quant_idx]
        
        k_dq = torch.matmul(dq_rotated_key, Q_matrix.t())
        
        base_attention_score = torch.matmul(query, k_dq.transpose(-1, -2))
        
        fused_corrected_score = base_attention_score + (residual_bit * error_scale)
        
        return fused_corrected_score