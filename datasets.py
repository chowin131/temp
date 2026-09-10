import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}

# 이름 -> (torchvision 클래스, mean, std)
CIFAR = {
    "cifar10": (datasets.CIFAR10,
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": (datasets.CIFAR100,
                 (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def get_dataloaders(name="cifar10", root="data", batch_size=128,
                    num_workers=2):
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

    return train_dataloader, test_dataloader
