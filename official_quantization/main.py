import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.quantization import QuantStub, DeQuantStub, QConfig

# Official utils
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, cpu=False):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image.cuda() if not cpu else image)
            cnt += 1
            acc1, acc5 = accuracy(output, target.cuda() if not cpu else target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    return top1, top5


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


q_backend = "qnnpack"
qconfig = torch.quantization.get_default_qconfig(q_backend)
torch.backends.quantized.engine = q_backend

r18_o = torchvision.models.resnet18(True)
r18_o.eval()

# Do NOT fuse inplaced relu
r18 = torch.quantization.fuse_modules(
    r18_o,
    [['conv1', 'bn1', 'relu'],
     ['layer1.0.conv1', 'layer1.0.bn1'], # , 'layer1.0.relu'],
     ['layer1.0.conv2', 'layer1.0.bn2'],
     ['layer1.1.conv1', 'layer1.1.bn1'], #, 'layer1.1.relu'],
     ['layer1.1.conv2', 'layer1.1.bn2'],

     ['layer2.0.conv1', 'layer2.0.bn1'], #, 'layer2.0.relu'],
     ['layer2.0.conv2', 'layer2.0.bn2'],
     ['layer2.0.downsample.0', 'layer2.0.downsample.1'],
     ['layer2.1.conv1', 'layer2.1.bn1'], #, 'layer2.1.relu'],
     ['layer2.1.conv2', 'layer2.1.bn2'],

     ['layer3.0.conv1', 'layer3.0.bn1'], #, 'layer3.0.relu'],
     ['layer3.0.conv2', 'layer3.0.bn2'],
     ['layer3.0.downsample.0', 'layer3.0.downsample.1'],
     ['layer3.1.conv1', 'layer3.1.bn1'], #, 'layer3.1.relu'],
     ['layer3.1.conv2', 'layer3.1.bn2'],

     ['layer4.0.conv1', 'layer4.0.bn1'], #, 'layer4.0.relu'],
     ['layer4.0.conv2', 'layer4.0.bn2'],
     ['layer4.0.downsample.0', 'layer4.0.downsample.1'],
     ['layer4.1.conv1', 'layer4.1.bn1'], #, 'layer4.1.relu'],
     ['layer4.1.conv2', 'layer4.1.bn2'],
     ]
)


# Append input/output quant/dequant stub.
def replace_forward(module):
    module.quant = QuantStub()
    module.dequant = DeQuantStub()
    raw_forward = module.forward

    def forward(x):
        x = module.quant(x)
        x = raw_forward(x)
        x = module.dequant(x)
        return x
    module.forward = forward

replace_forward(r18)


# 1K dataset
test_db = torchvision.datasets.ImageFolder(
    'imagenet_1k/val',
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
)

calibration_dataset = torch.utils.data.DataLoader(
    test_db,
    batch_size=256)

# 50K imagenet val
image_net_db = torchvision.datasets.ImageFolder(
    './ILSVRC2012_img_val/',
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
)

imagenet_50k = torch.utils.data.DataLoader(
    image_net_db,
    batch_size=256)


r18_o = r18_o.cuda()
r18.eval()

# Original network from torchvision
top1, top5 = evaluate(r18_o.cuda(), imagenet_50k)
print('Evaluation accuracy %2.2f'%(top1.avg))


# WARNING: Do NOT forget setting qconfig
r18.qconfig = qconfig

torch.quantization.prepare(r18, inplace=True)
evaluate(r18, calibration_dataset, cpu=True)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
r18 = r18.cpu()
torch.quantization.convert(r18, inplace=True)
print('Post Training Quantization: Convert done')


print("Size of model after quantization")
print_size_of_model(r18)
top1, top5 = evaluate(r18, imagenet_50k, cpu=True)
print('Evaluation accuracy %2.2f'%(top1.avg))

