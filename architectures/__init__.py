"""encoder 모음. forward(x)가 feature를 주고 차원은 num_features로 알려준다.
resnet, preact는 forward_features(x)로 stage별 feature map도 준다."""

from .ConvMixer import ConvMixer
from .DenseNet import DenseNet
from .FractalNet import FractalNet
from .MLPMixer import MLPMixer
from .PreActResNet import PreActResNet
from .ResNet import ResNet
from .VisionTransformer import VisionTransformer

# 프리셋 이름 -> (클래스, 생성 인자). 구조를 바꾸려면 여기만 고치면 됨
PRESETS = {
    # CIFAR 스타일 ResNet: stage 3개, 시작 채널 16
    "resnet20": (ResNet, dict(blocks=[3, 3, 3], base_channels=16)),
    "resnet56": (ResNet, dict(blocks=[9, 9, 9], base_channels=16)),
    # ImageNet ResNet-18의 stage 구성을 CIFAR용 3x3 stem에 얹은 것
    "resnet18": (ResNet, dict(blocks=[2, 2, 2, 2], base_channels=64)),
    "preactresnet20": (PreActResNet, dict(blocks=[3, 3, 3], base_channels=16)),
    "preactresnet18": (PreActResNet, dict(blocks=[2, 2, 2, 2], base_channels=64)),
    "densenet_bc100": (DenseNet, dict(blocks=[16, 16, 16], k=12, theta=0.5)),
    # C=3인 block 5개 -> 20 layer
    "fractalnet20": (FractalNet, dict(blocks=[3, 3, 3, 3, 3], p_local=0.15,
                                      p_global=0.5)),
    "vit_ti4": (VisionTransformer, dict(blocks=[12], patch_size=4,
                                        latent_vector_size=192, num_heads=3,
                                        p_drop=0.1)),
    "mlpmixer_8": (MLPMixer, dict(blocks=[8], patch_size=4,
                                  latent_vector_size=192,
                                  token_mlp_ratio=0.5, channel_mlp_ratio=4)),
    "convmixer_256_8": (ConvMixer, dict(blocks=[8], dim=256, kernel_size=5,
                                        patch_size=2)),
}


def get_encoder(name):
    cls, kwargs = PRESETS[name]
    return cls(**kwargs)
