from typing import List, Union

import torch
from torch import nn


class DLN(nn.Module):
    """
    A deep linear network with `L` layers with dimensions `dims`.

    Weights are initialized with variance `init_variance`.
    """

    def __init__(self, dims: List[int], init_variance: float = 1.0):
        super().__init__()
        self.dims = dims
        self.L = len(dims) - 1
        self.init_variance = init_variance
        self.linears = nn.ModuleList(
            [nn.Linear(d1, d2, bias=False) for d1, d2 in zip(dims[:-1], dims[1:])]
        )

        # Initialize weights and biases
        for l in range(self.L):
            self.linears[l].weight.data.normal_(
                0, self.init_variance
            )  # Note: this is not normalized by the input dimension

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
        return x

    def __repr__(self):
        return f"DLN({self.dims})"

    @classmethod
    def make_rectangular(cls, input_dim: int, output_dim: int, L: int, w: int, gamma: float):
        """
        Make a rectangular DLN with `L` layers and constant hidden width `w`.

        The input dimension is `input_dim` and the output dimension is `output_dim`.

        The weights are initialized from a normal distribution with variance`w ** (-gamma)`.
        """
        init_variance = w ** (-gamma)
        return cls([input_dim] + [w] * (L - 1) + [output_dim], init_variance=init_variance)

    def to_matrix(self):
        """Return the collapsed matrix representation of the DLN."""
        return self.forward(torch.eye(self.dims[0], device=self.device)).T

    @classmethod
    def from_matrix(cls, A: torch.Tensor, L=1):
        if L != 1:
            raise NotImplementedError("Only L=1 is supported for now.")

        output_dim, input_dim = A.shape
        instance = cls([input_dim, output_dim])
        instance.linears[0].weight.data.copy_(A)

        return instance

    def rank(self, **kwargs):
        """Return the rank of the DLN."""
        return torch.linalg.matrix_rank(self.to_matrix().to("cpu"), **kwargs)

    def ranks(self, **kwargs):
        """Return the ranks of the individual layers of the DLN."""
        return [torch.linalg.matrix_rank(l.weight.data.to("cpu"), **kwargs) for l in self.linears]

    def norm(self, p: Union[int, float, str] = 2):
        """Return the nuclear norm of the DLN."""
        return torch.norm(self.to_matrix().to("cpu"), p=p)

    def norms(self, p: Union[int, float, str] = 2):
        """Return the nuclear norms of the individual layers of the DLN."""
        return [torch.norm(l.weight.data.to("cpu"), p=p) for l in self.linears]

    def grad_norm(self, p=2, reduction="sum"):
        """Return the norm of the gradient of the DLN.

        If `reduction` is "sum", return the sum of the norms over all layers.
        If `reduction` is "none", return a list of the norms of the individual layers.

        """
        grad_norm = torch.zeros(self.L + 1, device=self.device)

        if p != 2:
            raise NotImplementedError("Only p=2 is implemented.")

        grad_norms = [torch.sum(linear.weight.grad**p) for linear in self.linears]

        if reduction == "sum":
            return sum(grad_norms)
        elif reduction != "none":
            raise ValueError(f"Unknown reduction {reduction}")

        return (grad_norm ** (1 / p)).to("cpu")

    @property
    def device(self):
        return next(self.parameters()).device
