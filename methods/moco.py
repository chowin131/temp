import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCo(nn.Module):
    """
        query encoder는 역전파로 학습하고, key encoder는 query encoder의 momentum update로만 따라감.
        이전 배치들의 key를 queue에 쌓아두고 negative로 씀 → batch 크기랑 상관없이 negative 수를 늘릴 수 있음.
        view 2장을 query, key 양쪽에 다 넣어서 loss를 대칭으로 계산하고 key 2N개를 queue에 넣음.
        knn과 linear eval은 SimCLR처럼 projector 거치기 전 encoder 출력을 사용함.
    """

    contrastive = True  # main이 train loader를 contrastive_dataloader로 바꾸는 데에 필요함

    def __init__(self, encoder, num_classes=None, proj_dim=128, queue_size=4096,
                 momentum=0.99, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.num_features
        self.projector = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, proj_dim))

        # key 쪽은 query 쪽(encoder + projector)을 복사해서 시작, gradient는 연결 안되게 함
        self.key_net = copy.deepcopy(nn.Sequential(self.encoder, self.projector))
        self.key_net.requires_grad_(False)

        # queue는 (K, D), 학습 파라미터는 아닌데 .to(device) 할 때 같이 옮겨져야 해서 buffer로 등록해놓음
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, proj_dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long))
        self.momentum = momentum
        self.temperature = temperature

    def query_net(self, x):
        return self.projector(self.encoder(x))

    def forward(self, batch):
        (x1, x2), _ = batch
        x_q, x_k = torch.cat([x1, x2]), torch.cat([x2, x1])
        q = F.normalize(self.query_net(x_q), dim=1)

        with torch.no_grad():
            q_params = [*self.encoder.parameters(), *self.projector.parameters()]
            for p_q, p_k in zip(q_params, self.key_net.parameters()):
                p_k.mul_(self.momentum).add_(p_q, alpha=1 - self.momentum)
            k = F.normalize(self.key_net(x_k), dim=1)

        l_pos = (q * k).sum(1, keepdim=True)
        l_neg = q @ self.queue.clone().T
        logits = torch.cat([l_pos, l_neg], 1) / self.temperature
        loss = F.cross_entropy(logits, torch.zeros(len(q), dtype=torch.long, device=q.device))

        n, ptr = len(k), int(self.queue_ptr)
        self.queue[ptr:ptr + n] = k
        self.queue_ptr.fill_((ptr + n) % len(self.queue))
        return loss
