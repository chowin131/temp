import math
import os
import torch
import torch.optim as optim
import architectures
import datasets
import methods
import trainer


def load_encoder(encoder, weight_path):
    """
        fit이 저장한 .pth(encoder + classifier 가중치)에서 encoder 가중치만 꺼내 encoder에 넣음.
        예: encoder.blocks.0.0.conv1.weight
          method.encoder    # encoder의 정보인지 classifier의 정보인지
                .blocks     # encoder에서 사용할 block 구성(stage list)
                .0          # 몇 번째 stage인지(0-based index)
                .0          # 해당 stage의 몇 번째 block인지(0-based index)
                .conv1      # block 안에서 어떤 레이어의 정보가 필요한지
                .weight     # 어떤 정보가 필요한지(weight/bias 등)
        state의 키는 method 기준 이름이라 앞에 "encoder."이 붙어 있음.
        그래서 "encoder."를 떼서 encoder 기준 이름(blocks.0.0.conv1.weight)으로 바꿈
    """
    state = torch.load(weight_path, map_location="cpu")

    state = {k[len("encoder."):]: v for k, v in state.items()
             if k.startswith("encoder.")}
    encoder.load_state_dict(state)


def run_experiment(arch="resnet20", method_name="supervised",
                   dataset="cifar10", data_root="data", batch_size=128,
                   num_workers=2, epochs=164, lr=0.1, momentum=0.9,
                   weight_decay=1e-4, schedule="step", milestones=(82, 123), gamma=0.1,
                   warmup_epochs=0, seed=42, out_dir="result", run_name=None,
                   encoder_weight=None, method_kwargs=None,
                   knn_k=1, knn_interval=10, log_interval=100, resume=False):

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = architectures.get_encoder(arch)

    if run_name is None:
        run_name = f"{arch}_{dataset}_{method_name}"
    if encoder_weight is not None:  # pretraining한 encoder 가져오고 run_name 뒤에 method 방식 붙여줌
        load_encoder(encoder, encoder_weight)
        run_name = f"{os.path.splitext(os.path.basename(encoder_weight))[0]}_{method_name}"

    train_dataloader, test_dataloader, knn_dataloader, contrastive_dataloader = \
        datasets.get_dataloaders(dataset, root=data_root, batch_size=batch_size, num_workers=num_workers)
    method = methods.get_method(
        method_name, encoder, datasets.NUM_CLASSES[dataset], **(method_kwargs or {})
    ).to(device)

    # method에 contrastive가 있다면(SimCLR, MoCo라면) contrastive용도 데이터셋 가져옴
    if getattr(method, "contrastive", False):
        train_dataloader = contrastive_dataloader

    optimizer = optim.SGD(method.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if schedule == "cosine": # lr은 warmup 동안 선형 증가, 이후 cosine으로 감소
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)

    # 학습되는 파라미터만 셈 (MoCo key encoder, linear eval의 encoder는 학습 안하니깐 제외)
    num_params = sum(p.numel() for p in method.parameters() if p.requires_grad)
    print(f"[{run_name}] {arch} on {device}, {num_params:,} trainable params")

    return trainer.fit(
        method, train_dataloader, test_dataloader, optimizer,
        scheduler, device, epochs=epochs,
        weight_path=os.path.join(out_dir, f"{run_name}.pth"),
        result_path=os.path.join(out_dir, f"{run_name}.json"),
        log_interval=log_interval, knn_dataloader=knn_dataloader,
        num_classes=datasets.NUM_CLASSES[dataset], knn_k=knn_k, knn_interval=knn_interval,
        resume=resume,
    )
