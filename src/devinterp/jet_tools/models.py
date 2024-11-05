from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from jet_tools.utils import prepare_input

from devinterp.callbacks import SamplerCallback


# some necessary functions
class PolyModel(nn.Module):
    def __init__(self, powers, device):
        super(PolyModel, self).__init__()
        self.weights = nn.Parameter(
            torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True).to(device)
        )
        self.powers = powers

    def forward(self, x):
        multiplied = torch.prod(self.weights**self.powers)
        x = x * multiplied
        return x


class SumModel(nn.Module):
    def __init__(self, powers, device):
        super(SumModel, self).__init__()
        self.weights = nn.Parameter(
            torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True).to(device)
        )
        self.powers = powers

    def forward(self, x):
        multiplied = torch.sum(self.weights**self.powers)
        x = x + multiplied
        return x


@dataclass
class DLNConfig:
    n_layers: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    initialization_method: str = "xavier"
    device: str = "cpu"


def initialise_xavier(n1: int, n2: int) -> torch.Tensor:
    return nn.init.xavier_normal_(torch.empty(n1, n2))


def initialise_kaiming(n1: int, n2: int) -> torch.Tensor:
    return nn.init.kaiming_normal_(torch.empty(n1, n2))


INITIALIZE_METHODS = {
    "zeros": lambda n1, n2: torch.zeros(n1, n2),
    "xavier": initialise_xavier,
    "kaiming": initialise_kaiming,
    "random": lambda n1, n2: torch.random(n1, n2),
}


def get_init_dln_params(
    dimensions: List[int], method: str = "xavier"
) -> List[torch.Tensor]:
    assert method in ["zeros", "xavier", "kaiming"]
    initialize_fn = INITIALIZE_METHODS[method]
    return [
        initialize_fn(dimensions[i], dimensions[i + 1])
        for i in range(len(dimensions) - 1)
    ]


def dln_rank(weights) -> int:
    matrix = weights[0]
    for layer in weights[1:]:
        matrix = np.matmul(matrix, layer)
    return np.linalg.matrix_rank(matrix)


def dln_learning_coefficient(model, verbose: bool = False) -> float:
    """The following implementation is recycled from https://github.com/edmundlth/validating_lambdahat/."""
    weights = [
        parameter.data.detach().cpu().numpy() for parameter in model.parameters()
    ]
    true_rank = dln_rank(weights)
    input_dim = weights[0].shape[0]
    layer_widths = [layer.shape[1] for layer in weights]

    def _condition(indices, intlist, verbose=False):
        intlist = np.array(intlist)
        ell = len(indices) - 1
        subset = intlist[indices]
        complement = intlist[[i for i in range(len(intlist)) if i not in indices]]
        has_complement = len(complement) > 0
        # print(indices, subset, complement)
        if has_complement and not (np.max(subset) < np.min(complement)):
            if verbose:
                print(
                    f"max(subset) = {np.max(subset)}, min(complement) = {np.min(complement)}"
                )
            return False
        if not (np.sum(subset) >= ell * np.max(subset)):
            if verbose:
                print(
                    f"sum(subset) = {sum(subset)}, ell * max(subset) = {ell * np.max(subset)}"
                )
            return False
        if has_complement and not (np.sum(subset) < ell * np.min(complement)):
            if verbose:
                print(
                    f"sum(subset) = {sum(subset)}, ell * min(complement) = {ell * np.min(complement)}"
                )
            return False
        return True

    def _search_subset(intlist):
        def generate_candidate_indices(intlist):
            argsort_indices = np.argsort(intlist)
            for i in range(1, len(intlist) + 1):
                yield argsort_indices[:i]

        for indices in generate_candidate_indices(intlist):
            if _condition(indices, intlist):
                return indices
        raise RuntimeError("No subset found")

    M_list = np.array([input_dim] + list(layer_widths)) - true_rank
    indices = _search_subset(M_list)
    M_subset = M_list[indices]
    if verbose:
        print(f"M_list: {M_list}, indices: {indices}, M_subset: {M_subset}")
    M_subset_sum = np.sum(M_subset)
    ell = len(M_subset) - 1
    M = np.ceil(M_subset_sum / ell)
    a = M_subset_sum - (M - 1) * ell
    output_dim = layer_widths[-1]

    term1 = (-(true_rank**2) + true_rank * (output_dim + input_dim)) / 2
    term2 = a * (ell - a) / (4 * ell)
    term3 = -ell * (ell - 1) / 4 * (M_subset_sum / ell) ** 2
    term4 = (
        1
        / 2
        * np.sum(
            [
                M_subset[i] * M_subset[j]
                for i in range(ell + 1)
                for j in range(i + 1, ell + 1)
            ]
        )
    )
    learning_coefficient = term1 + term2 + term3 + term4

    # multiplicity = a * (ell - a) + 1

    return learning_coefficient


class DLN(nn.Module):
    def __init__(self, config: DLNConfig):
        super().__init__()

        self.config = config
        dimensions = [config.input_dim, config.hidden_dim]
        layers = [nn.Linear(config.input_dim, config.hidden_dim, bias=False)]

        for _ in range(config.n_layers - 2):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim, bias=False))
            dimensions += [config.hidden_dim]

        layers.append(nn.Linear(config.hidden_dim, config.output_dim, bias=False))
        dimensions += [config.output_dim]
        self.layers = nn.Sequential(*layers)
        init_params = get_init_dln_params(dimensions, config.initialization_method)

        for model_param, param_to_initialize_to in zip(self.parameters(), init_params):
            model_param.data = param_to_initialize_to

    def get_device(self):
        return next(
            self.parameters()
        ).device  # assuming it's not fucky and all on the same device

    def to(self, device):
        return super().to(device)

    def _forward(self, inputs: torch.Tensor):
        return self.layers(inputs)

    def forward(self, input_ids) -> torch.Tensor:
        device = self.get_device()
        model_output = self._forward(prepare_input(input_ids, device=device))
        return model_output


from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import IterableDataset


class ContinuousDataset(IterableDataset):
    def __init__(self, teacher_model, input_dist="normal", seed=None):
        self.teacher_model = deepcopy(teacher_model).to("cpu")
        self.seed = seed
        self.input_dist = input_dist
        self.items_cache = []
        self._format_type = None
        self._shape = None
        self._column_names = None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            seed = self.seed
        else:  # in a worker process
            seed = self.seed + worker_info.id if self.seed is not None else None

        return self.data_generator(seed)

    def data_generator(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        input_dist = (
            (lambda: torch.randn(1, self.teacher_model.config.input_dim))
            if self.input_dist == "normal"
            else self.input_dist
        )

        while True:
            input_ids = input_dist()
            output = self.teacher_model._forward(input_ids)
            item = [input_ids.squeeze(), output.squeeze()]

            self.items_cache.append(item)
            yield item

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, int):
            while len(self.items_cache) <= idx:
                next(iter(self))
            return self.items_cache[idx]
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(max(idx.stop, 0) + 1)
            while len(self.items_cache) < stop:
                next(iter(self))
            return self.items_cache[start:stop:step]
        else:
            raise TypeError("Invalid argument type")

    def with_format(self, format_type):
        if format_type not in ["torch", None]:
            raise ValueError(f"Unsupported format type: {format_type}")
        self._format_type = format_type
        return self


def create_dataset(
    teacher_model, seed: int = None, verbose: bool = True, input_dist: str = "normal"
) -> ContinuousDataset:
    ds = ContinuousDataset(teacher_model, input_dist, seed)

    if verbose:
        print("Created continuous dataset (shuffle parameter is ignored)")
    return ds


def generate_dataset_for_seed(seed, sigma, batch_size, num_samples):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.normal(0, sigma, size=(num_samples,))
    y = torch.normal(0, sigma, size=(num_samples,))
    train_data = TensorDataset(x, y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader, train_data, x, y


class WeightCallback(SamplerCallback):
    def __init__(self, num_chains: int, num_draws: int, model, device="cpu"):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.ws = np.zeros(
            (num_chains, num_draws, *torch.stack(list(model.parameters())).shape),
            dtype=np.float32,
        )
        self.device = device

    def update(self, chain: int, draw: int, model):
        self.ws[chain, draw] = torch.stack(list(model.parameters())).cpu().detach()

    def get_results(self):
        return {
            "ws/trace": self.ws,
        }

    def __call__(self, chain: int, draw: int, model, **kwargs):
        self.update(chain, draw, model)
