import torch

from post_quant.accuracy_test import validate
from post_quant.common import _weight_quantize_range, dequantize, quantize


class ActivationMonitor(object):
    def __init__(self, bits=8, smooth=True):
        self.bits = bits
        self.smooth = smooth

    def __call__(self, m, _, output_):
        o_max = output_.max().item()
        o_min = output_.min().item()
        if m.output_max is None:
            m.output_max = torch.tensor(o_max)
            m.output_min = torch.tensor(o_min)
        else:
            if not self.smooth:
                if m.output_max < o_max:
                    m.output_max = o_max
                if m.output_min > o_min:
                    m.output_min = o_min
            else:
                m.output_max = m.output_max * 0.9 + o_max * 0.1
                m.output_min = m.output_min * 0.9 + o_min * 0.1
        min = m.output_min.item()
        max = m.output_max.item()
        s, z = _weight_quantize_range(min, max, bits=self.bits)
        m.output_scale = torch.tensor(s)
        m.output_zero_point = torch.tensor(z)


def register_activation_monitor(
        net,
        func):
    handles = []
    for n, module in net.named_modules():
        if need_monitor(module):
            h = hook_monitor(module, func)
            handles.append(h)
    return handles


def fake_quant_activation_module(net):
    for n, m in net.named_modules():
        if need_monitor(m):
            replace_forward_op(m)


def need_monitor(module):
    if isinstance(module, torch.nn.Conv2d) or \
            isinstance(module, torch.nn.BatchNorm2d) or \
            isinstance(module, torch.nn.Linear):
        return True
    return False


def hook_monitor(m, func):
    m.register_buffer('output_scale', None)
    m.register_buffer('output_zero_point', None)
    m.output_max = None
    m.output_min = None
    return m.register_forward_hook(func)


# Replace the forward function to record the output
def replace_forward_op(module):
    old_forward = module.forward
    s = module.output_scale.item()
    z = module.output_zero_point.item()

    def quant_forward(*input):
        output_ = old_forward(*input)
        return dequantize(quantize(output_, s, z), s, z)

    module.forward = quant_forward


def calibrate_activation_range(m, db, bits):
    hooks = register_activation_monitor(m, ActivationMonitor(bits=bits))
    validate(db, m)
    for h in hooks:
        h.remove()
