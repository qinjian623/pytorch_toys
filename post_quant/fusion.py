import copy
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from utils.modules import DummyModule


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma

    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode
    )
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    conv = None
    conv_name = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d) and conv:
            bc = fuse(conv, child)
            m._modules[conv_name] = bc
            m._modules[name] = DummyModule()
            conv = None
        elif isinstance(child, nn.Conv2d):
            conv = child
            conv_name = name
        else:
            fuse_module(child)


def validate(net, input_, cuda=True):
    net.eval()
    if cuda:
        net = net.cuda()
        input_ = input_.cuda()

    fused_net = copy.deepcopy(net)
    fuse_module(fused_net)
    fused_net.eval()

    with torch.no_grad():
        with autocast(False):
            a = net(input_)
            b = fused_net(input_)

    if cuda:
        torch.cuda.synchronize()

    return (a - b).abs().max().item()


if __name__ == '__main__':
    import torchvision
    mbnet = torchvision.models.mobilenet_v2(True)
    mbnet.eval()
    print(validate(mbnet, torch.randn(32, 3, 224, 224), True))
