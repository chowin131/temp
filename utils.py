import json

import matplotlib.pyplot as plt


def load_result(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_result(result, title=None):
    epochs = range(1, len(result["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, result["train_loss"], label="train")
    axes[0].plot(epochs, result["test_loss"], label="test")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(epochs, [acc * 100 for acc in result["test_acc"]])
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("test acc (%)")

    axes[2].plot(epochs, result["lr"])
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("lr")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
