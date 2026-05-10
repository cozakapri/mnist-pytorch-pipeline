from dataclasses import dataclass, field


@dataclass
class Config:
    # Podaci
    data_dir: str = "./data"
    val_split: float = 0.1

    # Arhitektura MLP
    hidden_dims: list = field(default_factory=lambda: [256, 128])
    dropout: float = 0.3

    # Arhitektura CNN
    cnn_channels: list = field(default_factory=lambda: [32, 64])
    cnn_dropout: float = 0.25

    # Trening
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 15
    patience: int = 5

    # Ostalo
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42
