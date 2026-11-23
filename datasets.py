import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}

CIFAR = {
    "cifar10": (datasets.CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": (datasets.CIFAR100, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


class MakeTwoViews:
    """sample 하나에 transform을 두 번 각각 적용해서, 배치에서 꺼낼 때 [x1, x2], y 형태로 꺼내짐"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return [self.transform(img), self.transform(img)]


def get_dataloaders(name="cifar10", root="data", batch_size=128, num_workers=2):
    dataset_cls, mean, std = CIFAR[name]

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    # SimCLR, MoCo용 augmentation (CIFAR는 blur 적용 안함)
    contrastive_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(32, scale=(0.08, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    training_data = dataset_cls(root=root, train=True, download=True,
                                transform=train_transform)
    test_data = dataset_cls(root=root, train=False, download=True,
                            transform=test_transform)

    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, drop_last=True,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    # 학습없이 knn할 때 사용할 train 이미지(feature 뽑는 용도), augmentation과 shuffle 없이 가져옴
    knn_data = dataset_cls(root=root, train=True, download=True,
                           transform=test_transform)
    knn_dataloader = DataLoader(
        knn_data, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )

    # SimCLR, MoCo pretraining data sample 하나당 augmentation 적용한 view 2장으로 만들어 사용
    contrastive_data = dataset_cls(root=root, train=True, download=True,
                                   transform=MakeTwoViews(contrastive_transform))
    contrastive_dataloader = DataLoader(
        contrastive_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, drop_last=True,
    )

    return train_dataloader, test_dataloader, knn_dataloader, contrastive_dataloader
