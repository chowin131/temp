"""학습/평가 루프.

method 안이 어떻게 생겼는지는 여기서 신경쓰지 않는다.
학습은 loss = method(batch), 평가는 method.evaluate(batch)만 쓴다.
"""

import json
import os
import time

import torch


def to_device(batch, device):
    return [t.to(device) for t in batch]


def train_one_epoch(method, dataloader, optimizer, device, scaler,
                    log_interval=100):
    method.train()
    total_loss, seen = 0.0, 0

    for step, batch in enumerate(dataloader):
        batch = to_device(batch, device)
        size = batch[0].size(0)

        optimizer.zero_grad()

        # scaler가 꺼져 있으면 scale/step/update가 평범한 backward/step이 됨
        with torch.amp.autocast(device_type=device,
                                enabled=scaler.is_enabled()):
            loss = method(batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * size
        seen += size

        if log_interval and step % log_interval == 0:
            print(f"loss: {loss.item():>7f}  [{seen:>5d}]")

    return total_loss / seen


@torch.no_grad()
def evaluate(method, dataloader, device, use_amp=False):
    method.eval()
    total_loss, correct, seen = 0.0, 0, 0

    for batch in dataloader:
        batch = to_device(batch, device)
        with torch.amp.autocast(device_type=device, enabled=use_amp):
            loss_sum, num_correct, size = method.evaluate(batch)
        total_loss += loss_sum.item()
        correct += int(num_correct)
        seen += size

    test_loss = total_loss / seen
    accuracy = correct / seen
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, "
          f"Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy


def fit(method, train_dataloader, test_dataloader, optimizer, scheduler,
        device, epochs=164, amp=True, ckpt_path="runs/checkpoint.pth",
        result_path="runs/result.json", log_interval=100):
    scaler = torch.amp.GradScaler(device, enabled=amp and device == "cuda")
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    result = {"train_loss": [], "test_loss": [], "test_acc": [], "lr": []}
    best_acc = 0.0
    start = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        result["lr"].append(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(method, train_dataloader, optimizer,
                                     device, scaler, log_interval)
        test_loss, test_acc = evaluate(method, test_dataloader, device,
                                       use_amp=scaler.is_enabled())
        scheduler.step()

        result["train_loss"].append(train_loss)
        result["test_loss"].append(test_loss)
        result["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(method.state_dict(), ckpt_path)

        with open(result_path, "w") as f:
            json.dump(result, f)

        elapsed = (time.time() - start) / 60
        eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
        print(f"[{elapsed:.1f}m elapsed | ETA {eta:.1f}m | "
              f"best {best_acc * 100:.2f}%]\n")

    print(f"Done! best test acc = {best_acc * 100:.2f}%")
    return result, best_acc
