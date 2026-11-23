"""
    method는 encoder를 포함함.
    forward(batch)가 loss 계산해서 줌.
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
