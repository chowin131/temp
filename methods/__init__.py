"""학습 방식 모음. method는 encoder를 감싸고 forward(batch)가 loss를 준다."""

from .supervised_learning import SupervisedLearning

METHOD_REGISTRY = {
    "supervised": SupervisedLearning,
}


def get_method(name, encoder, num_classes, **kwargs):
    return METHOD_REGISTRY[name](encoder, num_classes, **kwargs)
