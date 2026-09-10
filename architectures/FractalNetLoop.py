import torch
import torch.nn as nn
import torch.nn.functional as F

from .FractalNet import Base, join


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_cols, p_local=0.15):
        super().__init__()
        self.num_cols = num_cols
        self.p_local = p_local
        self.total = 2 ** (num_cols - 1)
        self.columns = nn.ModuleList()
        for c in range(num_cols):
            layers = nn.ModuleList()
            for i in range(2 ** c):
                if i == 0:
                    layers.append(Base(in_channels, out_channels))
                else:
                    layers.append(Base(out_channels, out_channels))
            self.columns.append(layers)

    def forward(self, x, global_col=None):
        if global_col is not None:
            for layer in self.columns[global_col]:
                x = layer(x)
            return [x]

        xs = [x] * self.num_cols
        depth = [0] * self.num_cols

        for cur_depth in range(1, self.total + 1):
            active = []
            for c in range(self.num_cols):
                period = 2 ** (self.num_cols - (c + 1))
                if cur_depth % period == 0:
                    xs[c] = self.columns[c][depth[c]](xs[c])
                    depth[c] += 1
                    active.append(c)

            if cur_depth == self.total:
                break

            joined = join([xs[c] for c in active], self.p_local, self.training)
            for c in active:
                xs[c] = joined

        return xs


class FractalNetLoop(nn.Module):
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

