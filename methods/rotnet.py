import torch
import torch.nn as nn
import torch.nn.functional as F


class RotNet(nn.Module):
    """
        pre training 단계에서는
        train dataset에서 샘플을 batch 단위로 가져와서 4방향으로 회전시킨 것들을 encoder에 넣어서 분류함.
        cross entropy loss로 encoder와 classifier weight 수정함.

        test에서는 회전 예측이 아니라 클래스 분류로 평가함.
        knn_interval 간격의 epoch가 끝나고 나면 그 시점 encoder를 그대로 가져와서 추가로 학습하지 않는 knn을 붙여 분류 수행
        (여기서도 회전 예측이 아니라 클래스 분류)한 뒤 정확도 측정.

        fine tuning 단계에서는 pre training된 encoder를 가져오고 새 classifier를 붙여 encoder와 같이 학습함.
    """

    def __init__(self, encoder, num_classes=None):  # num_classes는 get_method 형식 맞추기 용도
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.num_features, 4)

    @staticmethod
    def rotate(x):
        num = x.size(0)  # x는 (batch 번호, 채널, H, W)
        x = torch.cat([torch.rot90(x, k, (2, 3)) for k in range(4)], 0)
        y = torch.tensor([0]*num + [1]*num + [2]*num + [3]*num, device=x.device)
        return x, y

    def forward(self, batch):
        x, _ = batch  # 원래 라벨은 사용하지 않음
        x, y = self.rotate(x)
        y_pred = self.classifier(self.encoder(x))
        return F.cross_entropy(y_pred, y)
