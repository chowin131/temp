import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    # first=True면 stem(conv-bn-relu) 바로 뒤라서 앞쪽 bn-relu를 생략함
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
        # pre-activation, 첫 block은 stem에서 이미 bn-relu를 거쳤으므로 건너뜀
        if self.first:
            out = x
        else:
            out = F.relu(self.bn1(x))

        # shortcut도 pre-activation 결과를 받아야 identity mapping이 유지됨
        identity = x if self.downsample is None else self.downsample(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        return identity + out


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


# 옵션 B/C용 shortcut, 1x1 conv로 채널과 크기를 맞춤 (pre-act이라 bn 없음)
class ConvShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=stride, padding=0, bias=False)

    def forward(self, x):
        return self.conv1(x)


class PreActResNet(nn.Module):
    # 논문 기본 downsampling 옵션이 B
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
                self.make_block(in_channels, out_channels, num_layers, stride,
                                first=(idx == 0))
            )
            in_channels = out_channels

        self.blocks = nn.Sequential(*stages)
        # pre-act은 마지막 conv 뒤에 활성화가 없어서 bn-relu를 따로 붙여줌
        self.bn_final = nn.BatchNorm2d(in_channels)
        self.num_features = in_channels

    def make_block(self, in_channels, out_channels, num_layers, stride,
                   first=False):
        layers = []
        for i in range(num_layers):
            # 블록 안에서 크기를 줄이는건 첫 layer 뿐
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

        x = F.relu(self.bn_final(x))

        x = F.adaptive_avg_pool2d(x, 1)
        return torch.flatten(x, 1)

