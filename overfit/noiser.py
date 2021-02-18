import torch
import torch.nn as nn


# Setting anchor to True is actually kind of FeatureNoiser, but running much faster.
# TODO Should NOT be a nn.Module
class Noiser(nn.Module):
    def __init__(self, params, r=0.01, anchor=True):
        super(Noiser, self).__init__()
        self._params = list(params)
        self._r = r
        self._anchor = anchor
        if anchor:
            self._backup_params = []

    def forward(self):
        for idx, p in enumerate(self._params):
            if self._anchor:
                self._backup_params.append(p.detach().clone())
            n = (torch.randn(p.shape) * self._r).to(p.device)
            p.data += p.data*n

    def restore(self):
        if self._anchor:
            for idx, p in enumerate(self._params):
                p.data = self._backup_params[idx].data
            self._backup_params = []


class GradNoiser(nn.Module):
    def __init__(self, params, r=0.02):
        super(GradNoiser, self).__init__()
        self._params = list(params)
        self._r = r

    def forward(self):
        for p in self._params:
            n = (torch.randn(p.shape) * self._r).to(p.device)
            p.grad.data += p.grad.data*n


class FeatureNoiser(nn.Module):
    def __init__(self, model:nn.Module, r=0.02):
        super(FeatureNoiser, self).__init__()
        self._hooks = []
        self._r = r

        def hook(module, _, output):
            n = (torch.randn(output.shape) * self._r).to(output.device)
            output.data += output.data * n
            return output

        for n, m in model.named_modules():
            self._hooks.append(m.register_forward_hook(hook))

    def dimiss(self):
        for hk in self._hooks:
            hk.remove()
