import torch
import torch.nn as nn

class LinearNN(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size//2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class VectorModel(nn.Module):
    def __init__(self, bias: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class MatrixModel(nn.Module):
    def __init__(self, n_nodes: int, bias: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_nodes, n_nodes, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class BiasOnly(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias