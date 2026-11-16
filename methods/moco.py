import copy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits):
        super().__init__(num_features)
        self.num_splits = num_splits

    def forward(self, x):
        if not self.training:
            return super().forward(x)
        n, c, h, w = x.shape
        s = self.num_splits
        running_mean = self.running_mean.repeat(s)
        running_var = self.running_var.repeat(s)
        out = F.batch_norm(x.view(-1, c * s, h, w), running_mean, running_var,
                           self.weight.repeat(s), self.bias.repeat(s),
                           True, self.momentum, self.eps).view(n, c, h, w)
        self.running_mean.copy_(running_mean.view(s, c).mean(0))
        self.running_var.copy_(running_var.view(s, c).mean(0))
        return out


def split_bn(module, num_splits):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, SplitBatchNorm(child.num_features, num_splits))
        else:
            split_bn(child, num_splits)


class MoCo(nn.Module):

    contrastive = True

    def __init__(self, encoder, num_classes=None, proj_dim=128, queue_size=4096,
                 momentum=0.99, temperature=0.1, bn_splits=8):
        super().__init__()
        if bn_splits > 1:
            split_bn(encoder, bn_splits)
        self.encoder = encoder
        d = encoder.num_features
        self.projector = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, proj_dim),
        )

        self.encoder_k = copy.deepcopy(self.encoder)
        self.projector_k = copy.deepcopy(self.projector)
        for p in self.key_params():
            p.requires_grad_(False)

        self.register_buffer("queue", F.normalize(torch.randn(queue_size, proj_dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long))
        self.momentum = momentum
        self.temperature = temperature

    def query_params(self):
        return itertools.chain(self.encoder.parameters(), self.projector.parameters())

    def key_params(self):
        return itertools.chain(self.encoder_k.parameters(), self.projector_k.parameters())

    @torch.no_grad()
    def momentum_update(self):
        for q, k in zip(self.query_params(), self.key_params()):
            k.mul_(self.momentum).add_(q, alpha=1 - self.momentum)

    @torch.no_grad()
    def enqueue(self, k):
        idx = (self.queue_ptr + torch.arange(k.size(0), device=k.device)) % self.queue.size(0)
        self.queue[idx] = k
        self.queue_ptr.copy_((self.queue_ptr + k.size(0)) % self.queue.size(0))

    def forward(self, batch):
        x, _ = batch
        q = F.normalize(self.projector(self.encoder(x[:, 0])), dim=1)

        with torch.no_grad():
            self.momentum_update()
            perm = torch.randperm(x.size(0), device=x.device)
            k = self.projector_k(self.encoder_k(x[perm, 1]))[torch.argsort(perm)]
            k = F.normalize(k, dim=1)

        l_pos = (q * k).sum(1, keepdim=True)
        l_neg = q @ self.queue.clone().T
        logits = torch.cat([l_pos, l_neg], 1) / self.temperature
        labels = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        loss = F.cross_entropy(logits, labels)

        self.enqueue(k)
        return loss
