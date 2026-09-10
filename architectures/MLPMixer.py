import torch
import torch.nn as nn

from .VisionTransformer import MLP


class Block(nn.Module):
    def __init__(self, num_patches, dim, token_hidden, channel_hidden,
                 p_drop=0.0):
        super().__init__()
        self.LN1 = nn.LayerNorm(dim)
        self.token_mixing = MLP(num_patches, token_hidden, p_drop)
        self.LN2 = nn.LayerNorm(dim)
        self.channel_mixing = MLP(dim, channel_hidden, p_drop)

    def forward(self, x):
        identity = x
        out = self.LN1(x)
        out = out.transpose(1, 2)
        out = self.token_mixing(out)
        out = out.transpose(1, 2)
        x = identity + out

        identity = x
        out = self.LN2(x)
        out = self.channel_mixing(out)
        x = identity + out
        return x


class MLPMixer(nn.Module):
    def __init__(self, blocks, patch_size=4, latent_vector_size=192,
                 token_mlp_ratio=0.5, channel_mlp_ratio=4, p_drop=0.0,
                 image_size=32):
        super().__init__()
        num = blocks[0]
        dim = latent_vector_size
        num_patches = (image_size // patch_size) ** 2

        # ViT와 같은 방식(patch, projection을 conv으로 처리)
        self.conv1 = nn.Conv2d(3, dim, kernel_size=patch_size,
                               stride=patch_size, padding=0, bias=True)

        self.mixer = nn.Sequential(*[
            Block(num_patches, dim, int(dim * token_mlp_ratio),
                  int(dim * channel_mlp_ratio), p_drop)
            for _ in range(num)
        ])
        self.LN = nn.LayerNorm(dim)
        self.num_features = dim

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 2)
        x = x.transpose(1, 2) # conv 돌리면 뒤집혀서
        x = self.mixer(x)
        x = self.LN(x)
        return x.mean(dim=1) # global avg pooling
