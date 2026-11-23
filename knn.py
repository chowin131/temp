"""
    학습 없이 encoder를 평가하는 knn
    현재 학습 중인 encoder 상태를 그대로 가져옴.
    train 이미지에서 feature 벡터들을 뽑아서 각자 정답 label과 짝지어서 저장함.
    그 다음에 test 이미지 batch 단위로 뽑아서 encoder 통과시켜서 feature 뽑음.
    뽑아놓은 train feature들이랑 knn 돌려서(cosine similarity 계산) 예측값 만들고, 실제 test 이미지의 label과 비교해서 정확도 계산함.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_features(encoder, dataloader, device):
    """data들을 encoder에 통과시켜서 feature 뽑아낸 뒤 정규화해서 전달"""
    features, labels = [], []
    for x, y in dataloader:
        features.append(F.normalize(encoder(x.to(device)).float(), dim=1))
        labels.append(y.to(device))
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def knn_evaluate(encoder, knn_dataloader, test_dataloader, device, chunk=512):
    was_training = encoder.training
    encoder.eval()
    train_features, train_labels = extract_features(encoder, knn_dataloader, device)
    test_features, test_labels = extract_features(encoder, test_dataloader, device)
    encoder.train(was_training)

    correct = 0
    # 그냥 통째로 계산하면 메모리가 너무 많이 들어서 test를 chunk장씩 나눠서 계산함
    for i in range(0, test_features.size(0), chunk):
        sim = test_features[i:i + chunk] @ train_features.T  # 정규화했으니 내적 = cosine
        pred = train_labels[sim.argmax(1)]
        correct += (pred == test_labels[i:i + chunk]).sum().item()
    return correct / test_features.size(0)
