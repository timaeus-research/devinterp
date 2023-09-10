import functools
import inspect
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from devinterp.optim.schedulers import LRScheduler
from devinterp.slt.observables import MicroscopicObservable, estimate_rlct
from devinterp.utils import flatten_dict, map_nested

# from devinterp.slt.sampler import Sampler


class ModelEvaluator(Protocol):
    """Defines a protocol for evaluation methods.

    The __call__ method should be implemented to perform evaluation and return results as a dictionary.
    """

    def __call__(
        self,
        model: nn.Module,
    ) -> Dict[str, Any]:
        ...


class ModelOptimizerEvaluator(Protocol):
    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, Any]:
        ...


class ModelOptimizerSchedulerEvaluator(ABC):
    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
    ) -> Dict[str, Any]:
        ...


Evaluator = Union[
    ModelEvaluator, ModelOptimizerEvaluator, ModelOptimizerSchedulerEvaluator
]


class MSEEvaluator(ModelEvaluator):
    """Evaluates Mean Squared Error (MSE) over multiple datasets.

    Args:
        delimiter (str): Delimiter used for forming key names in the result dictionary.
        dataloaders (DataLoader): Keyword arguments representing dataset name and corresponding DataLoader.
    """

    def __init__(
        self, delimiter: str = "/", device: str = "cpu", **dataloaders: DataLoader
    ):
        self.dataloaders = dataloaders
        self.delimiter = delimiter
        self.device = device

    def __call__(
        self,
        model: nn.Module,
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
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                mse_sum += ((output - y) ** 2).sum().item()
                count += len(y)
            mse_sums[f"{name}{self.delimiter}mse"] = mse_sum / count
        return mse_sums


class CrossEntropyEvaluator(ModelEvaluator):
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


def clean_evals_results(result: Dict[str, Any], delimiter: str = "/"):
    """Flattens lists and converts tensors to floats"""

    def clean(x):
        if isinstance(x, torch.Tensor):
            return x.item()
        return x

    return flatten_dict(map_nested(clean, result), flatten_lists=True)


def run_eval(eval_: Evaluator, model, optimizer=None, scheduler=None):
    num_params = len(inspect.signature(eval_).parameters)

    if num_params == 1:  # ModelEvaluator
        result = eval_(model)  # type: ignore
    elif num_params == 2:  # ModelOptimizerEvaluator
        result = eval_(model, optimizer)  # type: ignore
    elif num_params == 3:  # ModelOptimizerSchedulerEvaluator
        result = eval_(model, optimizer, scheduler)  # type: ignore
    else:
        raise TypeError(f"Unknown Evaluator with {num_params} parameters.")

    return result


def run_and_clean_evals(eval_: Evaluator, model, optimizer=None, scheduler=None):
    return clean_evals_results(run_eval(eval_, model, optimizer, scheduler))


class CombineEvaluators(ModelOptimizerSchedulerEvaluator):
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
            (
                run_and_clean_evals(eval_, model, optimizer, scheduler)
                for eval_ in self.evals
            ),
            {},
        )


class RepeatEvaluator(ModelOptimizerSchedulerEvaluator):
    """Repeats a stochastic evaluation method multiple times and yields some statistics."""

    def __init__(self, eval_: Evaluator, num_repeats: int = 10, delimiter: str = "/"):
        self.eval_ = eval_
        self.num_repeats = num_repeats
        self.delimiter = delimiter

    def __call__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
    ) -> Dict[str, Any]:
        """Applies the evaluation method multiple times and yields some statistics.

        Returns:
            Dict: Dictionary containing the mean and standard deviation of the evaluation results.
        """
        results = {}

        for _ in range(self.num_repeats):
            result = run_eval(self.eval_, model, optimizer, scheduler)
            for key, value in result.items():
                if key not in results:
                    results[key] = []
                results[key].append(value)

        means = {
            key: torch.tensor(value).mean().item() for key, value in results.items()
        }
        stds = {key: torch.tensor(value).std().item() for key, value in results.items()}

        for key in means.keys():
            results[f"{key}{self.delimiter}mean"] = means[key]
            results[f"{key}{self.delimiter}std"] = stds[key]

        return clean_evals_results(results)


class EvaluatorWrapper(ModelOptimizerSchedulerEvaluator):
    def __init__(self, **evals):
        self.evals = evals

    def __call__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
    ):
        return clean_evals_results(
            {k: run_eval(v, model, optimizer, scheduler) for k, v in self.evals.items()}
        )
