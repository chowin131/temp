"""학습 방식 모음. method는 encoder를 감싸고 forward(batch)가 loss를 준다."""

from .supervised_learning import SupervisedLearning
from .self_supervised_learning_rotnet import RotNet

METHOD_REGISTRY = {
    "supervised": SupervisedLearning,
    "rotnet": RotNet,
}


def get_method(name, encoder, num_classes, **kwargs):
    return METHOD_REGISTRY[name](encoder, num_classes, **kwargs)
