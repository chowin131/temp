import json
import os
import time
import warnings
import torch
import knn


def to_device(batch, device):
    """batch를 device로 옮김. contrastive batch는 ([x1, x2], y)"""
    if isinstance(batch, (list, tuple)):
        return [to_device(b, device) for b in batch]
    return batch.to(device)


@torch.no_grad()
def evaluate(method, dataloader, device):
    method.eval()
    total_loss, correct, seen = 0.0, 0, 0

    for batch in dataloader:
        batch = [t.to(device) for t in batch]
        loss_sum, num_correct, size = method.evaluate(batch)
        total_loss += loss_sum.item()
        correct += int(num_correct)
        seen += size

    test_loss = total_loss / seen
    accuracy = correct / seen
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy


def fit(method, train_dataloader, test_dataloader, optimizer, scheduler,
        device, epochs=164, weight_path="result/checkpoint.pth",
        result_path="result/result.json", log_interval=100,
        knn_dataloader=None, knn_interval=10, resume=False):

    os.makedirs(os.path.dirname(weight_path) or ".", exist_ok=True)
    result = {"train_loss": [], "test_loss": [], "test_acc": [], "lr": [], "knn_epoch": [], "knn_acc": []}
    start_epoch = 0

    # resume하면 epoch마다 저장된 .pth(weight)와 .json(log) 보고 이어서 학습
    if resume and os.path.exists(weight_path) and os.path.exists(result_path):
        with open(result_path, "r") as f:
            result = json.load(f)
        start_epoch = len(result["train_loss"])
        method.load_state_dict(torch.load(weight_path, map_location=device))
        with warnings.catch_warnings(): # optimizer.step 전에 scheduler.step 부른다는 경고 무시
            warnings.simplefilter("ignore")
            for _ in range(start_epoch): # lr schedule을 끝난 epoch까지 step으로 진행시킴
                scheduler.step()
        print(f"resume: epoch {start_epoch}까지 끝난 상태에서 이어서 학습\n")

    start = time.time()

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        result["lr"].append(optimizer.param_groups[0]["lr"])

        method.train()
        total_loss, seen = 0.0, 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            size = batch[-1].size(0)
            optimizer.zero_grad()
            loss = method(batch)
            loss.backward()
            optimizer.step()
            if hasattr(method, "update_target"):
                method.update_target(epoch * len(train_dataloader) + step, epochs * len(train_dataloader))
            total_loss += loss.item() * size
            seen += size
            if log_interval and step % log_interval == 0:
                print(f"loss: {loss.item():>7f}  [{seen:>5d}]")

        train_loss = total_loss / seen
        result["train_loss"].append(train_loss)

        # evaluate가 있는 method만 test loss, acc 계산 (pretraining 때는 knn으로 확인)
        if hasattr(method, "evaluate"):
            test_loss, test_acc = evaluate(method, test_dataloader, device)
            result["test_loss"].append(test_loss)
            result["test_acc"].append(test_acc)
        scheduler.step()

        # knn_interval 간격에 해당하는 에포크들과 마지막 에포크 끝난 후 knn으로 encoder 평가
        last = epoch + 1 == epochs
        if knn_dataloader is not None and ((epoch + 1) % knn_interval == 0 or last):
            acc = knn.knn_evaluate(method.encoder, knn_dataloader,
                                   test_dataloader, device)

            result["knn_epoch"].append(epoch + 1)
            result["knn_acc"].append(acc)
            print(f"knn acc: {acc * 100:.2f}%")

        torch.save(method.state_dict(), weight_path)

        with open(result_path, "w") as f:
            json.dump(result, f)

        elapsed = (time.time() - start) / 60
        eta = elapsed / (epoch + 1 - start_epoch) * (epochs - epoch - 1)
        print(f"[{elapsed:.1f}m elapsed | ETA {eta:.1f}m]\n")

    print("Done!")
    return result