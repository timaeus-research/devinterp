import functools
from typing import Any, Dict, List, Optional, Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader

from devinterp.optim.schedulers import LRScheduler


class Evaluator(Protocol):
    """Defines a protocol for evaluation methods.

    The __call__ method should be implemented to perform evaluation and return results as a dictionary.
    """

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LRScheduler],
    ) -> Dict[str, Any]:
        ...


class MSEEvaluator(Evaluator):
    """Evaluates Mean Squared Error (MSE) over multiple datasets.

    Args:
        delimiter (str): Delimiter used for forming key names in the result dictionary.
        dataloaders (DataLoader): Keyword arguments representing dataset name and corresponding DataLoader.
    """

    def __init__(self, delimiter: str = "/", **dataloaders: DataLoader):
        self.dataloaders = dataloaders
        self.delimiter = delimiter

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LRScheduler],
    ) -> Dict[str, Any]:
        """Calculates and returns MSE for each dataset.

        Returns:
            Dict: Dictionary containing MSE values for each dataset.
        """

        mse_sums = {}
        for name, dataloader in self.dataloaders.items():
            mse_sum = 0
            count = 0
            for x, y in dataloader:
                output = model(x)
                mse_sum += ((output - y) ** 2).sum().item()
                count += len(y)
            mse_sums[f"{name}{self.delimiter}mse"] = mse_sum / count
        return mse_sums


class CrossEntropyEvaluator(Evaluator):
    """Evaluates Cross-Entropy loss and accuracy over multiple datasets.

    Args:
        delimiter (str): Delimiter used for forming key names in the result dictionary.
        dataloaders (DataLoader): Keyword arguments representing dataset name and corresponding DataLoader.
    """

    def __init__(self, delimiter: str = "/", **dataloaders: DataLoader):
        self.dataloaders = dataloaders
        self.delimiter = delimiter

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LRScheduler],
    ) -> Dict[str, Any]:
        """Calculates and returns cross-entropy and accuracy for each dataset.

        Returns:
            Dict: Dictionary containing cross-entropy and accuracy values for each dataset.
        """
        accuracies = {}
        cross_entropies = {}
        for name, dataloader in self.dataloaders.items():
            loss_sum = 0
            count = 0
            correct = 0

            loss_fn = nn.CrossEntropyLoss(reduction="sum")
            for x, y in dataloader:
                output = model(x)
                loss_sum += loss_fn(output, y).item()
                correct += (output.argmax(dim=1) == y).sum().item()
                count += len(y)

            accuracies[f"{name}{self.delimiter}accuracy"] = correct / count
            cross_entropies[f"{name}{self.delimiter}cross_entropy"] = loss_sum / count
        return {**accuracies, **cross_entropies}


class ComposeEvaluators(Evaluator):
    """Composes multiple evaluation methods into one.

    Args:
        evals (List[Evaluator]): List of evaluation methods to be applied.
    """

    def __init__(self, evals: List[Evaluator]):
        self.evals = evals

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LRScheduler],
    ) -> Dict[str, Any]:
        """Applies all evaluation methods and combines their results.

        Returns:
            Dict: Combined dictionary of evaluation metrics from all evaluation methods.
        """
        return functools.reduce(
            lambda x, y: x | y,
            (eval_(model, optimizer, scheduler) for eval_ in self.evals),
            {},
        )
