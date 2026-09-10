import torch
import torch.nn as nn
import torch.nn.functional as F


def join(outputs, p_local=0.0, training=False):
    if training and p_local > 0 and len(outputs) > 1:
        keep = torch.rand(len(outputs)) >= p_local
        if not keep.any():
            keep[torch.randint(len(outputs), (1,))] = True
        outputs = [y for y, k in zip(outputs, keep) if k]

    if len(outputs) == 1:
        return outputs[0]
    return sum(outputs) / len(outputs)


class Base(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_cols, p_local=0.15):
        super().__init__()
        self.num_cols = num_cols
        self.p_local = p_local
        self.base = Base(in_channels, out_channels)
        if num_cols > 1:
            self.deep1 = Block(in_channels, out_channels, num_cols - 1, p_local)
            self.deep2 = Block(out_channels, out_channels, num_cols - 1, p_local)

    def forward(self, x, global_col=None):
        if self.num_cols == 1 or global_col == 0:
            return [self.base(x)]

        deep_col = None if global_col is None else global_col - 1
        hidden = join(self.deep1(x, deep_col), self.p_local, self.training)
        ys = self.deep2(hidden, deep_col)

        if global_col is not None:
            return ys
        return [self.base(x)] + ys


class FractalNet(nn.Module):
    def __init__(self, blocks, p_local=0.15, p_global=0.5, base_channels=16):
        super().__init__()
        self.p_local = p_local
        self.p_global = p_global
        self.block_cols = list(blocks)
        self.max_cols = max(self.block_cols)

        stages = []
        in_channels = 3
        for idx, num_cols in enumerate(self.block_cols):
            out_channels = base_channels * (2 ** idx)
            stages.append(Block(in_channels, out_channels, num_cols, p_local))
            in_channels = out_channels

        self.blocks = nn.ModuleList(stages)
        self.num_features = in_channels

    def sample_global_col(self):
        if self.training and torch.rand(()) < self.p_global:
            return int(torch.randint(self.max_cols, ()))
        return None

    def forward(self, x):
        global_col = self.sample_global_col()

        for block, num_cols in zip(self.blocks, self.block_cols):
            col = None if global_col is None else min(global_col, num_cols - 1)
            x = join(block(x, col), self.p_local, self.training)
            x = F.max_pool2d(x, 2)

        x = F.adaptive_avg_pool2d(x, 1)
        return torch.flatten(x, 1)

