"""encoder 모음. forward(x)는 feature return num_features에 차원 저장"""

from .ConvMixer import ConvMixer
from .DenseNet import DenseNet
from .FractalNet import FractalNet
from .MLPMixer import MLPMixer
from .PreActResNet import PreActResNet
from .ResNet import ResNet
from .VisionTransformer import VisionTransformer

PRESETS = {
    "resnet20": (ResNet, dict(blocks=[3, 3, 3], base_channels=16)),
    "resnet56": (ResNet, dict(blocks=[9, 9, 9], base_channels=16)),
    "resnet18": (ResNet, dict(blocks=[2, 2, 2, 2], base_channels=64)),
    "preactresnet20": (PreActResNet, dict(blocks=[3, 3, 3], base_channels=16)),
    "preactresnet18": (PreActResNet, dict(blocks=[2, 2, 2, 2], base_channels=64)),
    "densenet_bc100": (DenseNet, dict(blocks=[16, 16, 16], k=12, theta=0.5)),
    "fractalnet20": (FractalNet, dict(blocks=[3, 3, 3, 3, 3], p_local=0.15, p_global=0.5)),
    "vit_ti4": (VisionTransformer, dict(blocks=[12], patch_size=4,
                                        latent_vector_size=192, num_heads=3, p_drop=0.1)),
    "mlpmixer_8": (MLPMixer, dict(blocks=[8], patch_size=4, latent_vector_size=192,
                                  token_mlp_ratio=0.5, channel_mlp_ratio=4)),
    "convmixer_256_8": (ConvMixer, dict(blocks=[8], dim=256, kernel_size=5, patch_size=2)),
}

def get_encoder(name):
    cls, kwargs = PRESETS[name]
    return cls(**kwargs)
