"""
    method는 encoder를 포함함.
    forward(batch)가 loss 계산해서 줌.
    contrastive = True인 method(SimCLR, MoCo)는 batch의 x가 (N, 2, C, H, W)로 들어옴.
"""

from .supervised_learning import SupervisedLearning
from .rotnet import RotNet
from .simclr import SimCLR
from .moco import MoCo
from .linear_eval import LinearEval

METHOD_REGISTRY = {
    "supervised": SupervisedLearning,
    "rotnet": RotNet,
    "simclr": SimCLR,
    "moco": MoCo,
    "linear": LinearEval,
}


def get_method(name, encoder, num_classes, **kwargs):
    return METHOD_REGISTRY[name](encoder, num_classes, **kwargs)
