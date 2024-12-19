import torch
import torch.nn as nn


class Polynomial(nn.Module):
    def __init__(self, powers=[1, 1]):
        super(Polynomial, self).__init__()
        self.powers = torch.tensor(powers)
        self.weights = nn.Parameter(
            torch.zeros_like(self.powers, dtype=torch.float32), requires_grad=True
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
        return x * (self.weights[0] - 1) * (torch.sum(self.weights**2) ** 2)


class ReducedRankRegressor(nn.Module):
    def __init__(self, m, h, n):
        super(ReducedRankRegressor, self).__init__()
        self.fc1 = nn.Linear(m, h, bias=False)
        self.fc2 = nn.Linear(h, n, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)


class DummyNaNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.counter = 0  # Add counter to track number of forward passes

        # Initialize with normal weights first
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
            self.linear.bias.fill_(0.0)

    def forward(self, x):
        self.counter += 1

        # After 5 runs, switch weights to inf to produce NaNs
        if self.counter > 100:
            with torch.no_grad():
                self.linear.weight.fill_(float("inf"))

        return self.linear(x)
