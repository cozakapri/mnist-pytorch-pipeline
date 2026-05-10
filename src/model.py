import torch.nn as nn


class MLP(nn.Module):
    """
    Višeslojna mreža sa potpuno povezanim slojevima.

    dropout=0.0 -> bez regularizacije (korisno za demonstraciju overfitting-a)
    dropout>0.0 -> sa Dropout regularizacijom
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] = (256, 128),
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # [B, 1, 28, 28] -> [B, 784]
        return self.net(x.view(x.size(0), -1))


class CNN(nn.Module):
    """
    Konvoluciona mreža za klasifikaciju slika.

    Konvolucioni slojevi uče prostorne obrasce direktno nad pikselima,
    što je prirodniji pristup od razvijanja slike u vektor.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] = (32, 64),
        num_classes: int = 10,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),  # 28x28
            nn.ReLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 14x14
            nn.Dropout2d(dropout),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),           # 14x14
            nn.ReLU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 7x7
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
