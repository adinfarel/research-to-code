'''
Testing quantization whether running correctly
'''

import numpy as np

from src.serving.compression.quant.quantization import Quantization


quantizer = Quantization()


def test_symmetric_quant_zero_maps_to_zero():

    params = np.array([-500.0, 0.0, 300.0])
    quantized, scale = quantizer.symmetric_quant(params, bits=8)

    dequant = quantizer.dequantize(quantized, scale)
    zero_idx = np.where(params == 0.0)[0][0]

    assert abs(dequant[zero_idx]) < 1e-3


def test_symmetric_quant_roundtrip_close_to_original():
    np.random.seed(0)
    params = np.random.randn(100) * 10

    quantized, scale = quantizer.symmetric_quant(params, bits=8)
    dequant = quantizer.dequantize(quantized, scale)

    # int8 quantization should keep error reasonably small for this scale
    max_error = np.max(np.abs(params - dequant))
    assert max_error < 1.0


def test_asymmetric_quant_handles_skewed_distribution_better():
    # Docstring claim: asymmetric handles skewed data (e.g. [500,...,0,-10])
    # better than symmetric, because symmetric wastes range on the unused side.
    params = np.concatenate([np.linspace(490, 500, 50), np.array([-10.0])])

    sym_q, sym_scale = quantizer.symmetric_quant(params, bits=8)
    sym_dequant = quantizer.dequantize(sym_q, sym_scale)

    asym_q, asym_scale, asym_zp = quantizer.asymmetric_quant(params, bits=8)
    asym_dequant = quantizer.dequantize(asym_q, asym_scale, asym_zp)

    sym_error = np.mean(np.abs(params - sym_dequant))
    asym_error = np.mean(np.abs(params - asym_dequant))

    assert asym_error < sym_error


def test_nf4_quantizes_to_valid_indices():
    params = np.random.randn(50)
    quantized, scale = quantizer.normal_float4(params)

    assert quantized.min() >= 0
    assert quantized.max() <= 15  # 16 possible NF4 values, index 0-15


def test_nf4_dequant_roundtrip_reasonable():
    np.random.seed(1)
    params = np.random.randn(200)  # NF4 assumes ~Gaussian distribution

    quantized, scale = quantizer.normal_float4(params)
    dequant = quantizer.dequantize_nf4(quantized, scale)

    max_error = np.max(np.abs(params - dequant))
    assert max_error < 0.5  # loose bound, NF4 optimized for gaussian-like data