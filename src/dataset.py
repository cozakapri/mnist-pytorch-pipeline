import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms() -> transforms.Compose:
    # MNIST mean=0.1307, std=0.3081 — standardne vrednosti iz literature
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def load_mnist(data_dir: str = "./data", val_split: float = 0.1, seed: int = 42):
    """
    Vraća (train_set, val_set, test_set).

    Podela se radi pre bilo kakve augmentacije, čime se sprečava data leakage
    augmentovanih verzija istih primera u različitim skupovima.
    """
    transform = get_transforms()
    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    return train_set, val_set, test_set


def get_dataloaders(
    train_set, val_set, test_set, batch_size: int = 64
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


def majority_class_accuracy(dataset) -> float:
    """Tačnost trivijalne strategije: uvek predvidi najčešću klasu."""
    if hasattr(dataset, "dataset"):
        targets = [dataset.dataset.targets[i].item() for i in dataset.indices]
    else:
        targets = dataset.targets.tolist()
    counts = Counter(targets)
    return max(counts.values()) / sum(counts.values())


# ── Vizualizacije ─────────────────────────────────────────────────────────────

def plot_sample_images(dataset, n: int = 10) -> plt.Figure:
    if hasattr(dataset, "dataset"):
        indices = list(dataset.indices[:n])
        images = [dataset.dataset.data[i] for i in indices]
        labels = [dataset.dataset.targets[i].item() for i in indices]
    else:
        images = [dataset.data[i] for i in range(n)]
        labels = [dataset.targets[i].item() for i in range(n)]

    fig, axes = plt.subplots(1, n, figsize=(n * 1.4, 2))
    for ax, img, lbl in zip(axes, images, labels):
        ax.imshow(img, cmap="gray")
        ax.set_title(str(lbl), fontsize=10)
        ax.axis("off")
    fig.suptitle("Uzorci iz trening skupa", y=1.05)
    plt.tight_layout()
    return fig


def plot_class_distribution(dataset, title: str = "Raspodela klasa") -> plt.Figure:
    if hasattr(dataset, "dataset"):
        targets = [dataset.dataset.targets[i].item() for i in dataset.indices]
    else:
        targets = dataset.targets.tolist()

    counts = Counter(targets)
    classes = sorted(counts)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(classes, [counts[c] for c in classes], color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_xlabel("Klasa (cifra)")
    ax.set_ylabel("Broj primera")
    ax.set_title(title)
    ax.set_xticks(classes)
    plt.tight_layout()
    return fig
