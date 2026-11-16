from .supervised_learning import SupervisedLearning


class LinearEval(SupervisedLearning):
    """
        pretraining한 encoder를 freeze하고 새 linear classifier만 학습시킴.
        forward, evaluate는 SupervisedLearning 그대로 씀.
    """

    def __init__(self, encoder, num_classes):
        super().__init__(encoder, num_classes)
        self.encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        # trainer가 method.train()을 불러도 encoder 안의 BN은 pretraining때 저장된 평균/분산을 그대로 쓰고, 새로 갱신하지 않게 고정
        return self
