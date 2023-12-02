import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, layers: list[int], use_resnets=True) -> None:
        super().__init__()
        self.use_resnets = use_resnets
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(layers[i + 1]) for i in range(len(layers) - 1)]
        )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x_prev = x
            x = layer(x)
            if self.use_resnets and x_prev.shape == x.shape:
                x = x + x_prev
            if layer != self.layers[-1]:
                x = self.activation(x)
                x = batch_norm(x)
        return x
