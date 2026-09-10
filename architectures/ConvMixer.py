import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    # depthwise -> residual -> pointwise
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size, groups=dim,
                                   padding="same")
        self.bn1 = nn.BatchNorm2d(dim)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        identity = x
        out = self.depthwise(x)
        out = F.gelu(out)
        out = self.bn1(out)
        x = identity + out

        x = self.pointwise(x)
        x = F.gelu(x)
        x = self.bn2(x)
        return x


class ConvMixer(nn.Module):
    def __init__(self, blocks, dim=256, kernel_size=5, patch_size=2):
        super().__init__()
        num = blocks[0]
        self.conv1 = nn.Conv2d(3, dim, kernel_size=patch_size,
                               stride=patch_size, padding=0)
        self.bn1 = nn.BatchNorm2d(dim)

        self.blocks = nn.Sequential(*[
            Block(dim, kernel_size) for _ in range(num)
        ])
        self.num_features = dim

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.bn1(x)
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return torch.flatten(x, 1)