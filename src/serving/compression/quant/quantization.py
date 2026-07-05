'''
Build Quantization implementation

INTUITION:
    Motivate why quant exists
    We know in inference we must put in model with their weight to VRAM GPU
    but what if model size is large for example 175GB? this is going to be problem
    if each weight of model have high precision such as FP32 (have 4 bytes) or FP16 (have 2 bytes)
    with 175GB parameter and each parameter have high precision in inference this combination getting worse
    
    so quant solve this problem, quant is technicque compression which reduce memory footprint
    by reduce precision weight of model, so instead of inference with FP32 we doing inference with integer
    such as  INT8, INT4, or even INT 1, INT 0.5 and so on.
    
    Why this is effective? model with 175GB parameter become small size and fits on VRAM GPU
    but what trade-off this method?
    so cause Quant it is mapping number from high precision to lower precision, this process reduce
    accuracy of this model, but why? cause error effect quantization from rounding error and clipping error
    make the early weight is not same with weight after de-quant, at least have different in floating point
    
    Example:
    Early weight    : 3.14159
    Quant to int    : 3 (by rounding)
    After de-quant  : 3.00000
    
    we get error as high as 0.12159 <- this is make model accuracy drops

Quantization
│
├── Mapping Scheme
│   ├── Symmetric
│   ├── Asymmetric
│   └── Non-uniform (NF4, dll.)
│
├── Bit-width
│   ├── INT8
│   ├── INT4
│   └── ...
│
├── Granularity
│   ├── Per-tensor
│   ├── Per-channel
│   └── Per-group
│
└── Quantization Algorithm
    ├── RTN
    ├── GPTQ
    ├── AWQ
    ├── SmoothQuant
    ├── QAT
    └── ...
'''

import numpy as np
from pytest import param

class Quantization:
    
    # ------------------ MAPPING SCHEME -----------------------
    def _clamp(self, params_q: np.ndarray, lower_bound: int, upper_bound: int):
        params_q[params_q < lower_bound] = lower_bound
        params_q[params_q > upper_bound] = upper_bound
        return params_q
    
    def symmetric_quant(self, params: np.ndarray, bits: int):
        r'''
        INTUITION:
        why called sym quant? because zero point unquantized weight same as quantized weight
        this is why called sym quant
        
        Advantages -> symm quant so fast implementation and cheap compute
        Dis...     -> symm quant very rigid, because if we have most distributed data only on one side
        symm quantized very poor, because space in opposite side become waste
        Example
        we have distributed data mostly in range [500, ..., 0, -10]
        see negative sides have a lot waste space 128 - 10 = 118 space waste
        
        INT: 8
        <---------------------------------->
        -300             0              300
           \             |               /
            \            |              /
             \           |             /
            -127         0           127
        <---------------------------------->
        
        see zero point unquantized weight same as quantized weight
        proof: 
        if weight = 0
        r = 0 / s + z (0) = 0
        w = s (r (0) - z (0)) = 0
        
        formula:
            Scale = max(abs(x)) / qmax
            Xq    = clamp(round(x/s), qmin,qmax)
            Xdq   = Scale * Xq 
        '''
        alpha = np.max(np.abs(params))
        qmax  = 2 ** (bits - 1) - 1
        qmin  = -qmax
        
        scale = alpha / qmax # to get range bits -> 2^(bits-1) - 1
        
        # quantized parameters (weight)
        quantized = self._clamp(np.round(params / scale), qmin, qmax)
        return quantized, scale
    
    def asymmetric_quant(self, params: np.ndarray, bits: int):
        r'''
        INTUITION:
        why called asymm quant? because zero point unquant is not the same with quant zero point
        
        Advantages -> We not wasting space because range of asymm dynamic corresponding where the zero point is
        Dis...     -> asymm quant have expensive compute because we need calculate zero point each distributed data
        
        UINT 8:
        <---------------------------------->
        -10              0               500
           \             |               /
            \            |              /
             \           |             /
              0          26          255
        <---------------------------------->
        
        formula:
        Scale      = (Xmax - Xmin) / (Qmax - Qmin)
        zero point = round(-Xmin / Scale) + Qmin
        Xq         = clamp(round(x/s) + zero point, qmin, qmax)
        Xdq        = Scale * (Xq - zero point)
        '''
        alpha   = np.max(params)
        beta    = np.min(params)
        
        qmin    = 0
        qmax    = 2 ** bits - 1
        
        scale   = (alpha - beta) / (qmax - qmin)
        zero_point    = qmin - np.round(beta / scale)
        zero_point    = np.clip(zero_point, qmin, qmax)
        
        quantized = self._clamp(np.round(params / scale) + zero_point, qmin, qmax)
        return quantized, scale, zero_point

    def dequantize(self, params_q: np.ndarray, scale: float, zero_point: float = 0.0):
        return scale * (params_q.astype(np.float32) - zero_point)
    
    # ------------------ Quantization Strategy -----------------------
    def normal_float4(self, params: np.ndarray):
        '''
        INTUITION:
        Normal quant assume that weight of model spreading evenly linearly
        even though weight of model originally always normal distributed (Gaussian)
        with mean = 0 and variance = 1
        [-1, 1]
        
        NF4 Solve this with quantile quantization,
        NF4 have 16 number hardcoded result from curve statistics gaussian from [-1, 1]
        based on paper QLoRA:
        [
            -1.0, -0.6965825, -0.52507305, -0.39491752, -0.28444136, -0.18477343, -0.09105004, 0.0,
            0.07958029, 0.1609302, 0.2461123, 0.33791524, 0.4431016, 0.5710033, 0.7513511, 1.0
        ]
        
        Process of quantization no longer divide with scale and rounding, but rather search
        who of 16 number above that position closer to real weight
        '''
        nf4_values = np.array([
            -1.0, -0.6965825, -0.52507305, -0.39491752, -0.28444136, -0.18477343, -0.09105004, 0.0,
            0.07958029, 0.1609302, 0.2461123, 0.33791524, 0.4431016, 0.5710033, 0.7513511, 1.0
        ], dtype=np.float32)
        
        scale = np.max(np.abs(params))
        if scale == 0:
            scale = 1.0
        
        normalized_params = params / scale
        
        # NEAREST NEIGHBOR MATCHING
        # expand_dim equal to unsqueeze
        # subtract element-wise
        distances = np.abs(np.expand_dims(normalized_params, axis=-1) - nf4_values) # (N,) -> (N, 1) - (16,) -> (N, 16)
        quantized = np.argmin(distances, axis=-1)
        
        return quantized, scale
    
    def dequantize_nf4(self, params_q: np.ndarray, scale: float):
        nf4_values = np.array([
            -1.0, -0.6965825, -0.52507305, -0.39491752, -0.28444136, -0.18477343, -0.09105004, 0.0,
             0.07958029, 0.1609302, 0.2461123, 0.33791524, 0.4431016, 0.5710033, 0.7513511, 1.0
        ], dtype=np.float32)
        
        return nf4_values[params_q] * scale