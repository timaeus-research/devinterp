from typing import Callable, Dict, Literal

import torch

from devinterp.utils import flatten_dict

Reduction = Literal["mean", "sum"]
Metric = Callable[
    [torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]  # model, data, target, output -> value
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataloader_map(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    metrics: Dict[str, Metric],
    device: torch.device = DEVICE,
):
    """
    Applies metrics to each batch in a DataLoader and accumulates the results in a dictionary.

    Parameters:
        model: PyTorch model to evaluate.
        loader: DataLoader to pull data from.
        metrics: Dictionary of metrics to evaluate.
        device: Device to use for computation.

    Returns:
        metric_values: Dictionary containing metric results for each item in the DataLoader.
    """
    metric_values = {
        metric_name: torch.zeros(len(loader.dataset), device=device)
        for metric_name in metrics.keys()
    }

    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            for metric_name, metric_fn in metrics.items():
                metric_values[metric_name][
                    i * len(data) : (i + 1) * len(data)
                ] = metric_fn(model, data, target, output)

    return metric_values


def dataloader_reduce(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    metrics: Dict[str, Metric],
    reduction: Reduction = "mean",
    device: torch.device = DEVICE,
):
    """
    Calculates and reduces metrics over an entire DataLoader.

    Parameters:
        model: PyTorch model to evaluate.
        loader: DataLoader to pull data from.
        metrics: Dictionary of metrics to evaluate.
        reduction: Method for reducing metric values ("mean" or other custom methods).
        device: Device to use for computation.

    Returns:
        metric_values: Dictionary containing the reduced metric values.
    """
    metric_values = {
        metric_name: torch.zeros(1, device=device) for metric_name in metrics.keys()
    }
    total = torch.zeros(1, device=device)

    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += len(data)
            for metric_name, metric_fn in metrics.items():
                metric_values[metric_name] += metric_fn(model, data, target, output)

    if reduction == "mean":
        for metric_name in metrics.keys():
            metric_values[metric_name] /= total

    return metric_values


def dataloaders_reduce(
    model: torch.nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    metrics: Dict[str, Metric],
    reduction: Reduction = "mean",
    device: torch.device = DEVICE,
):
    """
    Calculates and reduces metrics over multiple DataLoaders.

    Parameters:
        model: PyTorch model to evaluate.
        loaders: Dictionary of DataLoaders to pull data from.
        metrics: Dictionary of metrics to evaluate.
        reduction: Method for reducing metric values ("mean" or other custom methods).
        device: Device to use for computation.

    Returns:
        Dictionary containing the reduced metric values for each DataLoader.
    """
    return flatten_dict(
        {
            loader_name: dataloader_reduce(
                model, loader, metrics, reduction, device=device
            )
            for loader_name, loader in loaders.items()
        }
    )


class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(self, data: torch.utils.data.Dataset, *args, **kwargs):
        generator = torch.Generator(device="cpu")
        sampler = torch.utils.data.RandomSampler(data, generator=generator)
        kwargs.update({"sampler": sampler})
        super().__init__(data, *args, **kwargs)

    def set_seed(self, seed: int):
        self.sampler.generator.manual_seed(seed)
