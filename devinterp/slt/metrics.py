from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch
from torch import nn


class Metric(Protocol):
    @staticmethod
    def __call__(xs: torch.Tensor, ys: torch.Tensor, yhats: torch.Tensor, losses: torch.Tensor, loss: torch.Tensor, model: nn.Module) -> Any:
        raise NotImplementedError



class NegativeLogLikelihood(Metric):
    """
    Computes the relative log conditional probability of a sample given a model.
    """
    def __init__(self, ):
        

    def __call__(self, xs: Tensor, ys: Tensor, yhats: Tensor, model: Module) -> Any:
        return super().__call__(xs, ys, yhats, model)


class SingularFluctuation(Metric):
    """
    Computes the singular fluctuation of a sample given a model.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__("singular_fluctuation", self.compute_singular_fluctuation)
        self.model = model

    def compute_singular_fluctuation(self, sample, chain_idx):
        pass