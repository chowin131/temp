import torch
import torch.nn as nn
import torch.nn.functional as F


class RotNet(nn.Module):
    """PreTraining 단계에서 encoder에 입력을 4방향으로 회전시켜서 넣어주고, 해당 결과를 cross entropy 최소화시킴"""

    def __init__(self, encoder, num_classes, pretrain=True):
        super().__init__()
        self.encoder = encoder
        self.classifier1 = nn.Linear(encoder.num_features, 4) # pre training 용도
        self.classifier2 = nn.Linear(encoder.num_features, num_classes) # fine tuning 용도
        self.pretrain = pretrain

    def forward(self, batch):
        x, y = batch # y는 pretraining 때는 사용하지 않고, x는 (batch 번호, 채널, H, W)로 생김
        if self.pretrain:  # pretraining 때만 회전, 라벨도 회전 라벨로 교체
            B = x.size(0)
            x = torch.cat([torch.rot90(x, k, (2, 3)) for k in range(4)], 0)
            y = torch.tensor([0]*B + [1]*B + [2]*B + [3]*B, device=x.device)
            y_pred = self.classifier1(self.encoder(x))
        else:
            y_pred = self.classifier2(self.encoder(x))
        return F.cross_entropy(y_pred, y)

    def evaluate(self, batch):
        """(loss 합, 맞힌 개수, 샘플 수)를 돌려준다."""
        x, y = batch
        if self.pretrain:  # pretraining 때만 회전, 라벨도 회전 라벨로 교체
            B = x.size(0)
            x = torch.cat([torch.rot90(x, k, (2, 3)) for k in range(4)], 0)
            y = torch.tensor([0]*B + [1]*B + [2]*B + [3]*B, device=x.device)
            y_pred = self.classifier1(self.encoder(x))
        else:
            y_pred = self.classifier2(self.encoder(x))
        loss = F.cross_entropy(y_pred, y, reduction="sum")
        correct = (y_pred.argmax(1) == y).sum()
        return loss, correct, y.size(0)
