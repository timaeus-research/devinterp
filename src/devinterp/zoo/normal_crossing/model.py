import torch
import torch.nn as nn


# Define the neural network
class PolyModel(nn.Module):
    def __init__(self, powers):
        super(PolyModel, self).__init__()
        self.weights = nn.Parameter(
            torch.tensor([1.0, 0.3], dtype=torch.float32, requires_grad=True)
        )
        self.powers = powers

    def forward(self, x):
        multiplied = torch.prod(self.weights**self.powers)
        x = x * multiplied

        return x
