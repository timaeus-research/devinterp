import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Polynomial(nn.Module):
    def __init__(self, powers=[1, 1]):
        super(Polynomial, self).__init__()
        self.powers = torch.tensor(powers)
        self.weights = nn.Parameter(
            torch.tensor(
                torch.zeros_like(self.powers, dtype=torch.float32), requires_grad=True
            )
        )

    def forward(self, x):
        return x * torch.prod(self.weights**self.powers)


class LinePlusDot(nn.Module):
    def __init__(self, dim=2):
        super(LinePlusDot, self).__init__()
        self.weights = nn.Parameter(
            torch.zeros(dim, dtype=torch.float32), requires_grad=True
        )

    def forward(self, x):
        return x * (self.weights[0] - 1) * (torch.sum(self.weights**2) ** 2) # or should this be weights[:2]**2? Not sure


class ReducedRankRegressor(nn.Module):
    def __init__(self, m, h, n):
        super(ReducedRankRegressor, self).__init__()
        self.fc1 = nn.Linear(m, h, bias=False)
        self.fc2 = nn.Linear(h, n, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
