"""
    method는 encoder를 포함함.
    forward(batch)가 loss 계산해서 줌.
"""

from .supervised_learning import SupervisedLearning
from .rotnet import RotNet

METHOD_REGISTRY = {
    "supervised": SupervisedLearning,
    "rotnet": RotNet,
}


def get_method(name, encoder, num_classes, **kwargs):
    return METHOD_REGISTRY[name](encoder, num_classes, **kwargs)
