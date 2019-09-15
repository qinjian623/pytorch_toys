import torch


def _weight_quantize_range(min_w, max_w, bits):
    level = 2 ** bits - 1
    scale = (max_w - min_w) / level
    zero_point = round((0.0 - min_w) / scale)
    if max_w < 0:
        zero_point = level
    if min_w > 0:
        zero_point = 0
    return scale, zero_point


def dequantize(weight, S, Z):
    return S * (weight - Z)


def quantize(weight, S, Z, bits=8):
    return torch.clamp((weight / S).round() + Z, 0, 2 ** bits - 1)


def quantize_tensor(tensor, bits):
    s, z = _weight_quantize_parameter(tensor, bits)
    return dequantize(quantize(tensor, s, z, bits), s, z), s, z


def _weight_quantize_parameter(weight, bits=8):
    return _weight_quantize_range(weight.min().item(), weight.max().item(), bits)


def register_quant_params(m):
    with torch.no_grad():
        for n, module in m.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                module.register_buffer('weight_scale', torch.tensor(0.0))
                module.register_buffer('weight_zero_point', torch.tensor(0))
                module.register_buffer('bias_scale', torch.tensor(0.0))
                module.register_buffer('bias_zero_point', torch.tensor(0))
                module.register_buffer('output_scale', torch.tensor(0.0))
                module.register_buffer('output_zero_point', torch.tensor(0))