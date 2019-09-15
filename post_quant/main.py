import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms.transforms as transforms
import sys
sys.path.append('..')  # TODO Softer
from post_quant.fusion import fuse_module
from post_quant.fake_quantization import load_fake_quant_model, fake_quant
from post_quant.accuracy_test import validate


def main(cali_db_path, validation_path):
    model = torchvision.models.resnet101(True)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    db = datasets.ImageFolder(
        cali_db_path,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]))
    dataset = torch.utils.data.DataLoader(
        db,
        batch_size=128,
        num_workers=8,
        shuffle=False,
        pin_memory=True)
    q_model = fake_quant(model, dataset)
    torch.save(q_model.state_dict(), 'model.quant')
    m = load_fake_quant_model(torchvision.models.resnet101(), 'model.quant')
    m.cuda()
    fuse_module(model)

    db = datasets.ImageFolder(
        validation_path,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]))
    dataset = torch.utils.data.DataLoader(
        db,
        batch_size=128,
        num_workers=8,
        shuffle=False,
        pin_memory=True)
    validate(dataset, m)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
