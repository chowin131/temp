import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCo(nn.Module):
    """
        query encoder는 역전파로 학습하고, key encoder는 query encoder의 momentum update로만 따라감.
        이전 배치들의 key를 queue에 쌓아두고 negative로 씀 → batch 크기랑 상관없이 negative 수를 늘릴 수 있음.
        knn과 linear eval은 SimCLR처럼 projector 거치기 전 encoder 출력을 사용함.
    """

    contrastive = True  # main이 train loader를 contrastive_dataloader로 바꾸는 데에 필요함

    def __init__(self, encoder, num_classes=None, proj_dim=128, queue_size=4096,
                 momentum=0.99, temperature=0.1, bn_splits=8):
        super().__init__()
        self.encoder = encoder
        d = encoder.num_features
        self.projector = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, proj_dim))

        # key 쪽은 query 쪽(encoder + projector)을 복사해서 시작, gradient는 안 받음
        self.key_net = copy.deepcopy(nn.Sequential(self.encoder, self.projector))
        self.key_net.requires_grad_(False)

        # queue는 (K, D), 학습 파라미터는 아닌데 .to(device) 할 때 같이 옮겨져야 해서 buffer로 등록
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, proj_dim), dim=1))
        self.momentum = momentum
        self.temperature = temperature
        self.bn_splits = bn_splits

    def query_net(self, x):
        return self.projector(self.encoder(x))

    def split_forward(self, net, x):
        """
            batch를 bn_splits개로 나눠서 따로 forward → BN 통계도 그룹별로 따로 계산됨
        """
        return torch.cat([net(c) for c in x.chunk(self.bn_splits)])

    def forward(self, batch):
        x, _ = batch  # x는 (N, 2, C, H, W), 라벨은 사용 X
        n = x.size(0)
        q = F.normalize(self.split_forward(self.query_net, x[:, 0]), dim=1)

        with torch.no_grad():
            # momentum update
            q_params = [*self.encoder.parameters(), *self.projector.parameters()]
            for p_q, p_k in zip(q_params, self.key_net.parameters()):
                p_k.copy_(self.momentum * p_k + (1 - self.momentum) * p_q)

            # Shuffle BN(섞어서 key 계산하고 원래 순서로 되돌림)
            # q_i랑 k_i가 서로 다른 BN 그룹에 들어가게 함
            perm = torch.randperm(n, device=x.device)
            k = self.split_forward(self.key_net, x[perm, 1])[perm.argsort()]
            k = F.normalize(k, dim=1)

        # positive 1개를 0번 열에, queue의 negative K개를 그 뒤에 → 정답 label은 전부 0
        l_pos = (q * k).sum(1, keepdim=True)  # (N, 1)
        l_neg = q @ self.queue.T              # (N, K)
        logits = torch.cat([l_pos, l_neg], 1) / self.temperature
        loss = F.cross_entropy(logits, torch.zeros(n, dtype=torch.long, device=x.device))

        # 새 key를 앞에 붙이고 가장 오래된 것부터 버림
        self.queue = torch.cat([k, self.queue])[:self.queue.size(0)]
        return loss
