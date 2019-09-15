import torch
from .common import register_quant_params
from .fusion import fuse_module
from .weights import quantize_module
from .activation import fake_quant_activation_module, calibrate_activation_range


def load_fake_quant_model(m, f):
    state_dict = torch.load(f)
    m.eval()
    fuse_module(m)
    register_quant_params(m)
    m.load_state_dict(state_dict)
    fake_quant_activation_module(m)
    return m


def fake_quant(m, db, bits=8):
    m.eval()
    fuse_module(m)
    calibrate_activation_range(m, db, bits)
    quantize_module(m, bits)
    fake_quant_activation_module(m)
    return m
