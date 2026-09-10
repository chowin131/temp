import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, p_drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    # LN -> MSA -> addition -> LN -> MLP -> addition
    def __init__(self, dim, num_heads, mlp_ratio=4, p_drop=0.0):
        super().__init__()
        self.LN1 = nn.LayerNorm(dim)
        self.MSA = nn.MultiheadAttention(dim, num_heads, dropout=0,
                                         batch_first=True)
        self.proj_drop = nn.Dropout(p_drop)
        self.LN2 = nn.LayerNorm(dim)
        self.MLP = MLP(dim, dim * mlp_ratio, p_drop)

    def forward(self, x):
        identity = x
        out = self.LN1(x)
        out, _ = self.MSA(out, out, out, need_weights=False)
        out = self.proj_drop(out)
        x = identity + out

        identity = x
        out = self.LN2(x)
        out = self.MLP(out)
        x = identity + out

        return x


class Encoder(nn.Sequential):
    def __init__(self, num_blocks, dim, num_heads, mlp_ratio=4, p_drop=0.0):
        super().__init__(*[
            Block(dim, num_heads, mlp_ratio, p_drop) for _ in range(num_blocks)
        ])


class VisionTransformer(nn.Module):
    def __init__(self, blocks, num_heads=3, patch_size=4,
                 latent_vector_size=192, mlp_ratio=4, p_drop=0.1,
                 image_size=32):
        super().__init__()
        num_blocks = blocks[0]
        dim = latent_vector_size
        num_patches = (image_size // patch_size) ** 2

        # patch 분할 + patch별 flatten + projection을 conv 하나로 한 번에 처리
        # kernel=stride=patch_size라 겹치지 않게 잘라서 각각 linear 태우는 것과 같음
        self.conv1 = nn.Conv2d(3, dim, kernel_size=patch_size,
                               stride=patch_size, padding=0, bias=True)

        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        # position embedding은 0으로 두면 위치 구분이 안 되므로 작게 랜덤 초기화
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, dim) * 0.02
        )
        self.dropout = nn.Dropout(p_drop)

        self.encoder = Encoder(num_blocks, dim, num_heads, mlp_ratio, p_drop)
        self.LN = nn.LayerNorm(dim)
        self.num_features = dim

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 2)
        x = x.transpose(1, 2) # conv 돌리면 뒤집혀서

        class_token = self.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((class_token, x), dim=1)

        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.encoder(x)
        x = x[:, 0] # class token만 추출

        return self.LN(x)

