"""
학습 없이 encoder를 평가하는 knn

현재 학습 중인 encoder 상태를 그대로 가져옴.
train 이미지에서 feature 벡터들을 뽑아서 각자 정답 label과 짝지어서 저장함.
그 다음에 test 이미지 batch 단위로 뽑아서 encoder 통과시켜서 feature 뽑음.
뽑아놓은 train feature들이랑 knn 돌려서(cosine similarity 계산) 예측값 만들고, 실제 test 이미지의 label과 비교해서 정확도 계산함.

k==1이면 가장 비슷한 train 이미지 1개의 label로 예측함.
k>1이면 가장 비슷한 k개 train 이미지의 label을 cosine similarity를 가중치로 적용(temperature도 반영)해서 예측함.
encoder에 forward_features가 있으면 stage별로 따로 평가함.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_features(encoder, dataloader, device):
    """data들을 encoder에 통과시켜서 feature들을 뽑아냄"""
    features, labels = None, []
    for x, y in dataloader:
        x = x.to(device)
        if hasattr(encoder, "forward_features"):
            outs = [torch.flatten(F.adaptive_avg_pool2d(f, 1), 1)
                    for f in encoder.forward_features(x)]
        else:  # forward_features가 없으면 encoder(x) 출력(classifier에 들어가는 벡터) 하나로만 평가
            outs = [encoder(x)]
        outs = [F.normalize(o.float(), dim=1) for o in outs]

        if features is None:
            features = [[] for _ in outs]
        for i, o in enumerate(outs):
            features[i].append(o)
        labels.append(y.to(device))
    return [torch.cat(f) for f in features], torch.cat(labels)


@torch.no_grad()
def knn_predict(test_features, train_features, train_labels, num_classes,
                k=1, t=0.1, chunk=512):
    preds = []
    # 그냥 통째로 계산하면 메모리가 너무 많이 들어서 test를 chunk장씩 나눠서 계산함
    for i in range(0, test_features.size(0), chunk):
        sim = test_features[i:i + chunk] @ train_features.T  # 정규화했으니 내적 = cosine
        if k == 1:
            preds.append(train_labels[sim.argmax(1)])
            continue
        weight, idx = sim.topk(min(k, sim.size(1)), dim=1)
        weight = (weight / t).exp()
        votes = torch.zeros(sim.size(0), num_classes, device=sim.device)
        votes.scatter_add_(1, train_labels[idx], weight)
        preds.append(votes.argmax(1))
    return torch.cat(preds)


def knn_evaluate(encoder, knn_dataloader, test_dataloader, device,
                 num_classes, k=1, t=0.1):
    """stage별 kNN 정확도 리스트 (stage 없는 encoder는 길이 1)."""
    was_training = encoder.training
    encoder.eval()
    train_features, train_labels = extract_features(encoder, knn_dataloader, device)
    test_features, test_labels = extract_features(encoder, test_dataloader, device)
    encoder.train(was_training)

    accs = []
    for train_f, test_f in zip(train_features, test_features):
        pred = knn_predict(test_f, train_f, train_labels, num_classes, k, t)
        accs.append((pred == test_labels).float().mean().item())
    return accs
