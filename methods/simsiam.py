import torch.nn as nn
import torch.nn.functional as F


class SimSiam(nn.Module):
    """
        하나의 샘플에 augment 2개 각각 적용해서 view 두 개 만들고(x1, x2)
        둘 다 representation, projection 통과시킨 다음에(z1, z2) predictor 통과시켜서 p1, p2 만듦.
        => 교차해서 loss 계산함(-cosine similarity를 loss로)
        target인 z1, z2는 stop-gradient (backpropagation할 때 상수로 취급)
    """
    contrastive = True
    def __init__(self, encoder, num_classes=None, dim=256, pred_dim=64):
        super().__init__()
        self.encoder = encoder
        d = encoder.num_features

        self.projector = nn.Sequential(
            nn.Linear(d, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim),
        )

    def loss(self, p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def forward(self, batch):
        (x1, x2), _ = batch
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return (self.loss(p1, z2) + self.loss(p2, z1)) / 2
