from math import sqrt

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x


class Conv2dCrossAttention(nn.Module):
    r"""
    Attention from 2D vk space to 2D query space. If they are the same, this module will be a self-attention.

    VK space: (bs, D_vk_space, vk_h, vk_w)

    Q space: (bs, D_query_space, query_h, query_w)

    Output: (bs, D_output, query_h, query_w)
    """

    def __init__(self,
                 D_query_space: int,
                 D_vk_space: int,
                 D_emb: int,
                 D_output: int):
        super(Conv2dCrossAttention, self).__init__()
        self.query = nn.Conv2d(D_query_space, D_emb, 1)
        self.key = nn.Conv2d(D_vk_space, D_emb, 1)
        self.value = nn.Conv2d(D_vk_space, D_output, 1)
        self.scale = sqrt(D_emb)
        self.init()  # Special init.

    def init(self):
        # Without ReLU, xavier is better than kaiming, maybe.
        xavier_uniform_(self.query.weight)
        xavier_uniform_(self.key.weight)
        xavier_uniform_(self.value.weight)

    def forward(self, image: Tensor, query_space: Tensor):
        # Boring QKV
        query = self.query(query_space)
        key = self.key(image)
        value = self.value(image)

        bs, bc, bh, bw = query_space.shape
        _, ic, ih, iw = image.shape

        # Reshape to MLP style
        # Eliminate spatial dim
        # Permute to (batch size, seq length, emb) x (bs, emb, seq)
        query = query.reshape(bs, -1, bh * bw)
        key = key.reshape(bs, -1, ih * iw).permute(0, 2, 1)  # Still [bs, emb, seq], for bmm
        value = value.reshape(bs, -1, ih * iw)

        # Scores and outputs
        scores = torch.bmm(key, query) / self.scale
        weights = F.softmax(scores, dim=-2)

        # Weighted sum of value.
        outputs = torch.bmm(value, weights)

        # Back to CNN style
        outputs = outputs.reshape(bs, -1, bh, bw)
        return outputs


if __name__ == '__main__':
    D_query = 64
    D_img = 256
    bs = 16
    import torch

    torch.manual_seed(0)

    sa = Conv2dCrossAttention(D_query, D_img, 32, 256)
    bev = torch.ones(bs, D_query, 16, 10)
    img = torch.ones(bs, D_img, 36, 64)

    r = sa(img, bev)
    print(bev.shape, img.shape, r.shape)
    import time

    s = time.time()
    for i in range(10):
        sa(img, bev)

    print((time.time() - s) / 10)

    # Selfie
    sa = Conv2dCrossAttention(D_img, D_img, 64, 128)
    img = torch.ones(bs, D_img, 36, 64)
    print(sa(img, img).shape)

    s = time.time()
    for i in range(10):
        sa(img, img)

    print((time.time() - s) / 10)
