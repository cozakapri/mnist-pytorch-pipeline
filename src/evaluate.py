import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


@torch.no_grad()
def get_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Vraća (predikcije, tačne labele, slike) za ceo loader."""
    model.eval()
    all_preds, all_labels, all_images = [], [], []

    for x, y in loader:
        logits = model(x.to(device))
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y)
        all_images.append(x)

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_images),
    )


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(classification_report(y_true, y_pred, digits=4))


def plot_loss_curves(history: dict, title: str = "") -> plt.Figure:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set_xlabel("Epoha")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss{' – ' + title if title else ''}")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="validation")
    ax2.set_xlabel("Epoha")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy{' – ' + title if title else ''}")
    ax2.set_ylim(0.8, 1.0)
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = [str(i) for i in range(cm.shape[0])]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predviđena klasa")
    ax.set_ylabel("Tačna klasa")
    ax.set_title("Matrica konfuzije")

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()
    return fig


def plot_errors(
    images: torch.Tensor,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 12,
) -> plt.Figure | None:
    """Prikazuje prvih n pogrešno klasifikovanih primera."""
    error_idx = np.where(y_true != y_pred)[0][:n]
    if len(error_idx) == 0:
        print("Nema grešaka!")
        return None

    cols = min(n, len(error_idx))
    fig, axes = plt.subplots(1, cols, figsize=(cols * 1.5, 2.2))
    if cols == 1:
        axes = [axes]

    for ax, idx in zip(axes, error_idx):
        img = images[idx].squeeze()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"T:{y_true[idx]}\nP:{y_pred[idx]}", fontsize=9, color="crimson")
        ax.axis("off")

    fig.suptitle("Pogrešne predikcije  (T = tačno, P = predviđeno)", y=1.04)
    plt.tight_layout()
    return fig


def plot_predictions(
    model: nn.Module,
    dataset,
    device: torch.device,
    n: int = 16,
    seed: int = 0,
) -> plt.Figure:
    """
    Prikazuje n nasumičnih uzoraka iz dataset-a sa predikcijama modela.
    Zeleni naslov = tačno, crveni = pogrešno.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n, replace=False)

    images, true_labels, pred_labels = [], [], []
    with torch.no_grad():
        for idx in indices:
            x, y = dataset[int(idx)]
            logit = model(x.unsqueeze(0).to(device))
            images.append(x.squeeze().cpu())
            true_labels.append(int(y))
            pred_labels.append(int(logit.argmax(1).item()))

    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 2.0))
    axes = np.array(axes).flatten()

    for ax, img, true, pred in zip(axes, images, true_labels, pred_labels):
        ax.imshow(img, cmap="gray")
        color = "green" if true == pred else "crimson"
        ax.set_title(f"T:{true}  P:{pred}", fontsize=8, color=color)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_model_comparison(results: dict) -> plt.Figure:
    """
    Vizuelno poređenje sva tri modela: test accuracy, vreme treninga, broj parametara.
    results = {"Naziv": {"test_acc": float, "time_s": float, "params": int}}
    """
    names = list(results.keys())
    accs = [results[n]["test_acc"] * 100 for n in names]
    times = [results[n]["time_s"] for n in names]
    params = [results[n]["params"] / 1e3 for n in names]  # u hiliadama

    colors = ["#4878cf", "#6acc65", "#d65f5f"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Test accuracy
    bars = axes[0].bar(names, accs, color=colors, edgecolor="white")
    axes[0].bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    axes[0].set_ylim(min(accs) - 1, 100.5)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Test accuracy")

    # Vreme treninga
    bars = axes[1].bar(names, times, color=colors, edgecolor="white")
    axes[1].bar_label(bars, fmt="%.1fs", padding=3, fontsize=9)
    axes[1].set_ylabel("Vreme (s)")
    axes[1].set_title("Ukupno vreme treninga")

    # Broj parametara
    bars = axes[2].bar(names, params, color=colors, edgecolor="white")
    axes[2].bar_label(bars, fmt="%.1fK", padding=3, fontsize=9)
    axes[2].set_ylabel("Broj parametara (K)")
    axes[2].set_title("Veličina modela")

    plt.tight_layout()
    return fig


def compare_loss_curves(histories: dict[str, dict]) -> plt.Figure:
    """
    Overlay validacionih loss krivih za više modela na jednom grafiku.
    Korisno za vizuelno poređenje efekata regularizacije.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for name, h in histories.items():
        epochs = range(1, len(h["val_loss"]) + 1)
        ax1.plot(epochs, h["val_loss"], label=name)
        ax2.plot(epochs, h["val_acc"], label=name)

    ax1.set_xlabel("Epoha")
    ax1.set_ylabel("Validation loss")
    ax1.set_title("Poređenje validation loss")
    ax1.legend()

    ax2.set_xlabel("Epoha")
    ax2.set_ylabel("Validation accuracy")
    ax2.set_title("Poređenje validation accuracy")
    ax2.legend()

    plt.tight_layout()
    return fig
