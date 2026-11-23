import json

import matplotlib.pyplot as plt


def load_result(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_result(result, title=None):
    epochs = range(1, len(result["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, result["train_loss"], label="train")
    if result["test_loss"]:
        axes[0].plot(epochs, result["test_loss"], label="test")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    if result["test_acc"]:
        axes[1].plot(epochs, [acc * 100 for acc in result["test_acc"]])
        axes[1].set_ylabel("test acc (%)")
    else:
        axes[1].plot(result["knn_epoch"], [a * 100 for a in result["knn_acc"]], marker="o")
        axes[1].set_ylabel("kNN acc (%)")
    axes[1].set_xlabel("epoch")

    axes[2].plot(epochs, result["lr"])
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("lr")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
