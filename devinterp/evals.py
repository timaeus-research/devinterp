from typing import Callable, Dict, Literal

import torch

from devinterp.config import Config
from devinterp.utils import flatten_dict

Reduction = Literal["mean", "sum"]
Metric = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]  # model, data, target, output -> value
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataloader_map(model: torch.nn.Module, loader: torch.utils.data.DataLoader, metrics: Dict[str, Metric], device: torch.device = DEVICE):
    metric_values = {metric_name: torch.zeros(len(loader.dataset), device=device) for metric_name in metrics.keys()}

    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            for metric_name, metric_fn in metrics.items():
                metric_values[metric_name][i * len(data) : (i + 1) * len(data)] = metric_fn(model, data, target, output)

    return metric_values
                

def dataloader_reduce(model: torch.nn.Module, loader: torch.utils.data.DataLoader, metrics: Dict[str, Metric], reduction: Reduction = "mean", device: torch.device = DEVICE):
    metric_values = {metric_name: torch.zeros(1, device=device) for metric_name in metrics.keys()}
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

def dataloaders_reduce(model: torch.nn.Module, loaders: Dict[str, torch.utils.data.DataLoader], metrics: Dict[str, Metric], reduction: Reduction = "mean", device: torch.device = DEVICE):
    return flatten_dict({loader_name: dataloader_reduce(model, loader, metrics, reduction, device=device) for loader_name, loader in loaders.items()})

