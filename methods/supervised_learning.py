import torch.nn as nn
import torch.nn.functional as F


class SupervisedLearning(nn.Module):
    """encoder가 뽑은 feature를 linear classifier로 분류하는 학습 방식."""

    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.num_features, num_classes)

    def forward(self, batch):
        x, y = batch
        y_pred = self.classifier(self.encoder(x))
        return F.cross_entropy(y_pred, y)

    def evaluate(self, batch):
        """(loss 합, 맞힌 개수, 샘플 수)를 돌려준다."""
        x, y = batch
        y_pred = self.classifier(self.encoder(x))
        loss = F.cross_entropy(y_pred, y, reduction="sum")
        correct = (y_pred.argmax(1) == y).sum()
        return loss, correct, y.size(0)
