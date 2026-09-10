import os

import torch
import torch.optim as optim

import architectures
import datasets
import methods
import trainer


def run_experiment(arch="resnet", blocks=None, arch_kwargs=None,
                   method_name="supervised", dataset="cifar10",
                   data_root="data", batch_size=128, num_workers=2,
                   epochs=2, lr=0.1, momentum=0.9, weight_decay=1e-4,
                   milestones=(82, 123), gamma=0.1, seed=42, amp=True,
                   out_dir="runs", run_name=None, log_interval=100):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if blocks is None:
        blocks = architectures.DEFAULT_BLOCKS[arch]
    if arch_kwargs is None:
        arch_kwargs = {}
    if run_name is None:
        run_name = f"{arch}_{'-'.join(str(b) for b in blocks)}_{dataset}"

    train_dataloader, test_dataloader = datasets.get_dataloaders(
        dataset, root=data_root, batch_size=batch_size,
        num_workers=num_workers,
    )

    encoder = architectures.get_encoder(arch, blocks, **arch_kwargs)
    method = methods.get_method(
        method_name, encoder, datasets.NUM_CLASSES[dataset]
    ).to(device)

    optimizer = optim.SGD(method.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)

    num_params = sum(p.numel() for p in method.parameters())
    print(f"[{run_name}] {arch} {blocks} on {device}, {num_params:,} params")

    return trainer.fit(
        method, train_dataloader, test_dataloader, optimizer, scheduler,
        device, epochs=epochs, amp=amp,
        ckpt_path=os.path.join(out_dir, f"{run_name}.pth"),
        result_path=os.path.join(out_dir, f"{run_name}.json"),
        log_interval=log_interval,
    )
