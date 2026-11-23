import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    """
        같은 이미지의 view 2장을 positive pair, batch 안의 나머지 2(N-1)장을 negative로 두고 NT-Xent loss로 학습.
        knn과 linear eval을 수행할 때에는 projection head g를 거치기 전 encoder 출력을 사용함.
    """

    contrastive = True  # main이 train loader를 contrastive_dataloader로 바꾸는 데에 필요함

    def __init__(self, encoder, num_classes=None, proj_dim=128, temperature=0.5):
        super().__init__()
        self.encoder = encoder
        d = encoder.num_features
        self.projector = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
            nn.Linear(d, proj_dim),
        )
        self.temperature = temperature

    def nt_xent(self, z):
        """z는 (2N, D), (i, i + N)이 positive 쌍, 나머지 모든 조합은 negative 쌍임."""
        n = z.size(0) // 2
        z = F.normalize(z, dim=1)
        sim = z @ z.T / self.temperature  # 정규화했으니 내적 = cosine similarity
        self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(self_mask, float("-inf")) # 자기 자신과의 유사도는 분모에서 뺌
        target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
        return F.cross_entropy(sim, target)

    def forward(self, batch):
        (x1, x2), _ = batch
        return self.nt_xent(self.projector(self.encoder(torch.cat([x1, x2]))))
