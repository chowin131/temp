"""encoder 모음. forward(x)가 feature를 주고 차원은 num_features로 알려준다."""

from .ConvMixer import ConvMixer
from .DenseNet import DenseNet
from .FractalNet import FractalNet
from .FractalNetLoop import FractalNetLoop
from .MLPMixer import MLPMixer
from .PreActResNet import PreActResNet
from .ResNet import ResNet
from .VisionTransformer import VisionTransformer
ARCHITECTURES_REGISTRY = {
    "resnet": ResNet,
    "preactresnet": PreActResNet,
    "densenet": DenseNet,
    "fractalnet": FractalNet,           # 재귀 버전
    "fractalnet_loop": FractalNetLoop,  # 반복문 버전
    "vit": VisionTransformer,
    "mlpmixer": MLPMixer,
    "convmixer": ConvMixer,
}

# blocks를 안 주면 쓰는 기본값
# resnet 계열    : block당 layer 수       (예: [3,3,3] -> ResNet-20)
# densenet       : block당 dense layer 수 (예: [16,16,16] -> DenseNet-BC-100)
# fractalnet 계열: block당 column 수 C    (예: [3]*5 -> 20 layer)
# vit            : encoder block 수       (예: [12] -> ViT-Ti/4)
# mixer 계열     : mixer block 수         (예: [8])
DEFAULT_BLOCKS = {
    "resnet": [3, 3, 3],
    "preactresnet": [3, 3, 3],
    "densenet": [16, 16, 16],
    "fractalnet": [3, 3, 3, 3, 3],
    "fractalnet_loop": [3, 3, 3, 3, 3],
    "vit": [12],
    "mlpmixer": [8],
    "convmixer": [8],
}


def get_encoder(name, blocks, **kwargs):
    return ARCHITECTURES_REGISTRY[name](blocks, **kwargs)
