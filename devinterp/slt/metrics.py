from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch
from torch import nn


class Metric(Protocol):
    @staticmethod
    def __call__(xs: torch.Tensor, ys: torch.Tensor, yhats: torch.Tensor, losses: torch.Tensor, loss: torch.Tensor, model: nn.Module) -> Any:
        raise NotImplementedError