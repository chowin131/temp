import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(out + identity)


# 옵션 A용 shortcut, downsampling 일어날 때 padding으로 크기 조절 (파라미터 없음)
class PadShortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        new_x = x[:, :, ::2, ::2]
        new_x = F.pad(
            new_x, (0, 0, 0, 0, 0, self.out_channels - self.in_channels)
        )
        return new_x


# 옵션 B/C용 shortcut, 1x1 conv로 채널과 크기를 맞춤
class ConvShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        new_x = self.conv1(x)
        new_x = self.bn1(new_x)
        return new_x


class ResNet(nn.Module):
    # 기본 downsampling 옵션은 논문과 같은 B
    def __init__(self, blocks, option="B", base_channels=16):
        super().__init__()
        self.option = option
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        stages = []
        in_channels = base_channels
        for idx, num_layers in enumerate(blocks):
            out_channels = base_channels * (2 ** idx)
            # 첫 블록 이후에는 feature map 크기 절반씩 줄이기
            stride = 1 if idx == 0 else 2
            stages.append(
                self.make_block(in_channels, out_channels, num_layers, stride)
            )
            in_channels = out_channels

        self.blocks = nn.Sequential(*stages)
        self.num_features = in_channels

    def make_block(self, in_channels, out_channels, num_layers, stride):
        layers = []
        for i in range(num_layers):
            # 블록 안에서 크기를 줄이는건 첫 layer 뿐
            cur_stride = stride if i == 0 else 1
            layers.append(
                Block(in_channels, out_channels, cur_stride,
                      self.rescale_identity(in_channels, out_channels,
                                            cur_stride))
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def rescale_identity(self, in_channels, out_channels, stride):
        # 옵션 C는 크기가 같아도 항상 projection
        if self.option == "C":
            return ConvShortcut(in_channels, out_channels, stride)

        if stride == 1 and in_channels == out_channels:
            return None
        if self.option == "A":
            return PadShortcut(in_channels, out_channels)
        return ConvShortcut(in_channels, out_channels, stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.blocks(x)

        x = F.adaptive_avg_pool2d(x, 1)
        return torch.flatten(x, 1)

