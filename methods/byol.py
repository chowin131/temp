import copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOL(nn.Module):
    """
        하나의 샘플에 augment 2개 각각 적용해서 view 두 개 만들고
        network를 복사해서 target network, online network 두 개 만들어서 각각 view 2개씩 넣어줌.
        target network에서는 representation(encoder), projection MLP까지 거치고
        online network에서는 prediction MLP까지 거침.
        이후 view끼리 교차해서 loss 계산함 (cosine distance)
        target network를 gradient 없이 online network의 EMA로 업데이트.
        (tau값을 momentum으로 사용하며, cosine scale의 schedule을 따름, base_tau에서 1까지 증가)
    """
    contrastive = True
    def __init__(self, encoder, num_classes=None, base_tau=0.99, hidden_dim=512, out_dim=128):
        super().__init__()
        self.encoder = encoder
        d = encoder.num_features

        self.projector = nn.Sequential(
            nn.Linear(d, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for p in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            p.requires_grad = False

        self.base_tau = base_tau

    def loss(self, online, target):
        online = F.normalize(online, dim=-1)
        target = F.normalize(target.detach(), dim=-1)
        return (1 - (online * target).sum(dim=-1)).mean()

    def forward(self, batch):
        (x1, x2), _ = batch
        on1 = self.predictor(self.projector(self.encoder(x1)))
        on2 = self.predictor(self.projector(self.encoder(x2)))
        with torch.no_grad():
            tar1 = self.target_projector(self.target_encoder(x1))
            tar2 = self.target_projector(self.target_encoder(x2))
        return (self.loss(on1, tar2) + self.loss(on2, tar1)) / 2

    @torch.no_grad()
    def update_target(self, step, total_steps):
        cur_tau = 1 - (1 - self.base_tau) * (math.cos(math.pi * step / total_steps) + 1) / 2
        for online, target in ((self.encoder, self.target_encoder), (self.projector, self.target_projector)):
            for i, j in zip(online.parameters(), target.parameters()):
                j.mul_(cur_tau).add_(i, alpha=1 - cur_tau)
