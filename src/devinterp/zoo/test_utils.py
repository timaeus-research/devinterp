import torch
import torch.nn as nn
from more_itertools import pairwise
import numpy as np
import copy
from torch.functional import F
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
        return (
            x * (self.weights[0] - 1) * (torch.sum(self.weights**2) ** 2)
        )  # or should this be weights[:2]**2? Not sure


class ReducedRankRegressor(nn.Module):
    def __init__(self, layer_widths, factors=[5.0, 1 / 5, 1.0]):
        super(ReducedRankRegressor, self).__init__()
        self.factors = copy.deepcopy(factors)
        self.layer1 = nn.Linear(layer_widths[0], layer_widths[1], bias=False)
        self.layer2 = nn.Linear(layer_widths[1], layer_widths[2], bias=False)

    def forward(self, x):
        x = F.relu(self.layer1(x * self.factors[0]))
        x = F.relu(self.layer2(x * self.factors[1]))
        return x
    
    def to_numpy_matrix(self):
        with torch.no_grad():
            matrices = [self.layer1.weight.detach().numpy(),  self.layer2.weight.detach().numpy() ]
            overall_matrix = np.linalg.multi_dot(matrices[::-1])  # Reverse the order for right multiplication
        return overall_matrix
