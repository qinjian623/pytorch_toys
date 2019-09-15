import torch
import torch.nn as nn
from post_quant.common import quantize_tensor


def quantize_module(
        module: nn.Module,
        bits=8):
    with torch.no_grad():
        for n, module in module.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                module.weight.data, s, z = quantize_tensor(module.weight.data, bits)
                module.register_buffer('weight_scale', torch.tensor(s))
                module.register_buffer('weight_zero_point', torch.tensor(z))
                if module.bias is not None:
                    module.bias.data, s, z = quantize_tensor(module.bias.data, bits)
                    module.register_buffer('bias_scale', torch.tensor(s))
                    module.register_buffer('bias_zero_point', torch.tensor(z))

