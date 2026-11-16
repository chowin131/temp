import json
import os
import time
import torch
import knn


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
        knn_dataloader=None, num_classes=10, knn_k=1, knn_interval=10):

    os.makedirs(os.path.dirname(weight_path) or ".", exist_ok=True)
    result = {"train_loss": [], "test_loss": [], "test_acc": [], "lr": [],
              "knn_epoch": [], "knn_acc": []}
    start = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        result["lr"].append(optimizer.param_groups[0]["lr"])

        # 학습
        method.train()
        total_loss, seen = 0.0, 0
        for step, batch in enumerate(train_dataloader):
            batch = [t.to(device) for t in batch]
            size = batch[0].size(0)
            optimizer.zero_grad()
            loss = method(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * size
            seen += size
            if log_interval and step % log_interval == 0:
                print(f"loss: {loss.item():>7f}  [{seen:>5d}]")

        train_loss = total_loss / seen
        result["train_loss"].append(train_loss)

        # evaluate가 있는 method만 test loss, acc 계산 (pretraining method는 knn으로 확인)
        if hasattr(method, "evaluate"):
            test_loss, test_acc = evaluate(method, test_dataloader, device)
            result["test_loss"].append(test_loss)
            result["test_acc"].append(test_acc)
        scheduler.step()

        # knn_interval 간격의 에포크들과 마지막 에포크 끝난 후 학습 없이 knn으로 encoder 평가함
        last = epoch + 1 == epochs
        if knn_dataloader is not None and ((epoch + 1) % knn_interval == 0 or last):
            accs = knn.knn_evaluate(method.encoder, knn_dataloader, test_dataloader,
                                    device, num_classes, k=knn_k)

            result["knn_epoch"].append(epoch + 1)
            result["knn_acc"].append(accs)
            print("knn acc (stage별): " + ", ".join(f"{a * 100:.2f}%" for a in accs))

        torch.save(method.state_dict(), weight_path)

        with open(result_path, "w") as f:
            json.dump(result, f)

        elapsed = (time.time() - start) / 60
        eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
        print(f"[{elapsed:.1f}m elapsed | ETA {eta:.1f}m]\n")

    print("Done!")
    return result