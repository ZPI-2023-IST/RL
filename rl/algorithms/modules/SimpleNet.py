import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, layers: list[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x
