import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, k=12):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            channels = in_channels + i * k
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, 4 * k, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(4 * k),
                nn.ReLU(),
                nn.Conv2d(4 * k, k, kernel_size=3, stride=1,
                          padding=1, bias=False),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat((x, layer(x)), 1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=1, padding=0, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        x = self.avgpool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, blocks, k=12, theta=0.5):
        super().__init__()
        cur_channels = 2 * k
        self.conv1 = nn.Conv2d(3, cur_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for idx, num_layers in enumerate(blocks):
            self.blocks.append(DenseBlock(cur_channels, num_layers, k))
            cur_channels = cur_channels + num_layers * k

            if idx != len(blocks) - 1:
                out_channels = int(cur_channels * theta)
                self.transitions.append(
                    TransitionLayer(cur_channels, out_channels)
                )
                cur_channels = out_channels

        self.bn = nn.BatchNorm2d(cur_channels)
        self.num_features = cur_channels

    def forward(self, x):
        x = self.conv1(x)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx < len(self.transitions):
                x = self.transitions[idx](x)

        x = self.bn(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return torch.flatten(x, 1)

