import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 first=False):
        super().__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = None if first else nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        # 첫 block은 이미 bn, relu를 거쳐서 identity
        out = x if self.first else F.relu(self.bn1(x))

        identity = x if self.downsample is None else self.downsample(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        return identity + out


class PreActResNet(nn.Module):
    # blocks = stage별 layer 수
    def __init__(self, blocks, base_channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        stages = []
        in_channels = base_channels
        for idx, num_layers in enumerate(blocks):
            out_channels = base_channels * (2 ** idx)
            # 첫 stage 이후에는 feature map 크기 절반씩 줄이기
            stride = 1 if idx == 0 else 2
            stages.append(
                self.make_block(in_channels, out_channels, num_layers, stride,
                                first=(idx == 0))
            )
            in_channels = out_channels

        self.blocks = nn.Sequential(*stages)
        # 마지막 conv 뒤에 비선형 activation function이 없어서 bn, relu를 붙여줌
        self.bn_final = nn.BatchNorm2d(in_channels)
        self.num_features = in_channels

    def make_block(self, in_channels, out_channels, num_layers, stride,
                   first=False):
        layers = []
        for i in range(num_layers):
            # stage 안에서 크기를 줄이는 건 맨 앞 block 뿐 (그 block의 conv1에서 stride 적용)
            cur_stride = stride if i == 0 else 1
            layers.append(
                Block(in_channels, out_channels, cur_stride,
                      self.rescale_identity(in_channels, out_channels,
                                            cur_stride),
                      first=(first and i == 0))
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def rescale_identity(self, in_channels, out_channels, stride):
        # 크기가 그대로면 identity, 바뀔 때만 1x1 conv projection
        if stride == 1 and in_channels == out_channels:
            return None
        return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False)

    def forward_features(self, x):
        """stage별 feature map 리스트. 마지막 stage만 bn, relu까지 거친 최종값"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        features = []
        for stage in self.blocks:
            x = stage(x)
            features.append(x)
        features[-1] = F.relu(self.bn_final(features[-1]))
        return features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.blocks(x)

        x = F.relu(self.bn_final(x))

        x = F.adaptive_avg_pool2d(x, 1)
        return torch.flatten(x, 1)

