import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from tqdm import tqdm

from overfit.noiser import Noiser


class TallerLeNet(nn.Module):
    def __init__(self, in_channels=3, classes=10):
        super(TallerLeNet, self).__init__()
        self.n = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(256, classes, bias=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.n(x)


def train(model, device, train_loader, optimizer, noiser, loss_function=F.nll_loss):
    model.train()
    correct = 0
    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            for rpt in range(REPEAT):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                optimizer.zero_grad()

                for ns in noiser:
                    ns.forward()

                output = model(data)

                for ns in noiser:
                    ns.restore()

                loss = loss_function(output, target)
                loss.backward()

                optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            t.set_postfix(loss="{:05.3}".format(loss.item()))
            t.update()

    print('Train set: Accuracy: {}/{} ({:.2f}%)'.format(
        correct,
        len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 10,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    if args.aug:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    dataset1 = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]))

    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    student = TallerLeNet()
    student.to(device)
    print("Params size:", sum([p.numel() for p in student.parameters()]))

    optimizer = optim.SGD(student.parameters(), lr=0.2, momentum=0.9)

    ws = Noiser(student.parameters(), r=0.2, anchor=True)

    if isinstance(optimizer, optim.SGD):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    l = nn.CrossEntropyLoss()

    best_acc = 0
    best_epoch = 0

    for epoch in range(1, args.epochs//REPEAT + 1):
        train(student, device, train_loader, optimizer, noiser=[ws], loss_function=l)
        if isinstance(optimizer, optim.SGD):
            scheduler.step()
        acc = test(student, device, test_loader)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(student.state_dict(), "best_dict_state.bin")
        print("----> {}: best acc: {} @ epoch: {}".format(epoch, best_acc, best_epoch))

    if args.save_model:
        torch.save(student.state_dict(), "latest.pt")
    print("Final best acc: {}".format(best_acc))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='CIFAR10 noiser test')

    parser.add_argument('--aug', action='store_true', default=True, help='aug image')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=2048, metavar='N',
                        help='input batch size for testing (default: 2048)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train (default: 120)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    REPEAT = 1
    main()
