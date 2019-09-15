# Usage

Check main.py or as below:

```python
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms.transforms as transforms
from post_quant.fake_quantization import fake_quant, load_fake_quant_model

model = torchvision.models.resnet50(True)
model.eval()

db = datasets.ImageFolder(
    "ILSVRC2012_img_val",
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

# Quantize model
q_model = fake_quant(model, dataset)

# Save model with scale & zero point:
torch.save(model.state_dict(), 'model.quant') 

# Reload model:
m = load_fake_quant_model(torchvision.models.resnet50(), 'model.quant')
```


# TODO
 - [ ] Symmetric quantization
 - [ ] Channel-wise weight quantization
 - [ ] More sophisticated activation range calibration