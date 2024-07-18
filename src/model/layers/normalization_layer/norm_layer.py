import torch
import torch.nn as nn


class NormDivision(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = None

    def _build(self, x):
        self.in_channels = x.size(-2)
        self.in_height = x.size(-1)
        self.scale = 1 / (self.in_channels**2)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(self.in_channels, self.in_height, dtype=torch.float32)
        )

    def forward(self, x):
        if self.in_channels is None:
            self._build(x)
        weights = torch.sigmoid(self.weights1)
        return x / weights.unsqueeze(0).repeat(x.size(0), 1, 1)


if __name__ == "__main__":
    x = torch.randn(64, 32, 18)
    z = NormDivision()(x)
