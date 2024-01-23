from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from torch import nn

from devinterp.slt.callback import SamplerCallback

WeightAccessor = Callable[[nn.Module], torch.Tensor]


class CovarianceAccumulator(SamplerCallback):
    """
    A callback to iteratively compute and store the covariance matrix of model weights.
    For use with `sample`.

    :param num_weights: Total number of weights.
    :param accessors: Functions to access model weights.
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'.
    :param num_evals: Number of eigenvalues to compute.
    """

    def __init__(
        self,
        num_weights: int,
        accessors: List[WeightAccessor],
        device: Union[torch.device, str] = "cpu",
        num_evals: int = 3,
    ):
        self.num_weights = num_weights
        self.first_moment = torch.zeros(num_weights, device=device)
        self.second_moment = torch.zeros(num_weights, num_weights, device=device)
        self.num_draws = 0
        self.accessors = accessors
        self.num_evals = num_evals
        self.is_finished = False

    def accumulate(self, model: nn.Module):
        # Accumulate moments from model weights.
        assert not self.is_finished, "Cannot accumulate after finalizing."

        weights = torch.cat(
            [accessor(model).view((-1,)) for accessor in self.accessors]
        )
        self.first_moment += weights
        self.second_moment += torch.outer(weights, weights)
        self.num_draws += 1

    def finalize(self):
        self.first_moment /= self.num_draws
        self.second_moment /= self.num_draws
        self.is_finished = True

    def reset(self):
        self.first_moment.zero_()
        self.second_moment.zero_()
        self.num_draws = 0
        self.is_finished = False

    def to_matrix(self):
        # Convert the moments to a covariance matrix.
        return self.second_moment - torch.outer(self.first_moment, self.first_moment)

    def to_eigen(self, include_matrix=False):
        # Convert the covariance matrix to pairs of eigenvalues and vectors.
        cov = self.to_matrix().detach().cpu().numpy()
        evals, evecs = eigsh(cov, k=self.num_evals, which="LM")

        results = {"evals": evals, "evecs": evecs}

        if include_matrix:
            results["matrix"] = cov

        return results

    def sample(self):
        return self.to_eigen(include_matrix=True)

    def __call__(self, model):
        self.accumulate(model)


AttentionHeadWeightsAccessor = Callable[[nn.Module], Tuple[torch.Tensor, ...]]


class WithinHeadCovarianceAccumulator:
    """
    A CovarianceAccumulator to compute covariance within attention heads.
    For use with `sample`.

    :param num_heads: The number of attention heads.
    :param num_weights_per_head: The number of weights per attention head.
    :param accessors: Functions to access attention head weights.
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'.
    :param num_evals: number of eigenvectors / eigenvalues to return.
    """

    def __init__(
        self,
        num_heads: int,
        num_weights_per_head: int,
        accessors: List[AttentionHeadWeightsAccessor],
        device: Union[torch.device, str] = "cpu",
        num_evals: int = 3,
    ):
        self.num_layers = len(accessors)
        self.num_heads = num_heads
        self.num_weights_per_head = num_weights_per_head

        self.first_moment = torch.zeros(
            self.num_layers, num_heads, self.num_weights_per_head, device=device
        )
        self.second_moment = torch.zeros(
            self.num_layers,
            num_heads,
            self.num_weights_per_head,
            self.num_weights_per_head,
            device=device,
        )
        self.num_draws = 0
        self.accessors = accessors
        self.num_evals = num_evals
        self.is_finished = False

    @property
    def num_weights_per_layer(self):
        return self.num_heads * self.num_weights_per_head

    @property
    def num_weights(self):
        return self.num_layers * self.num_weights_per_layer

    def accumulate(self, model: nn.Module):
        # Accumulate moments from model weights.
        assert not self.is_finished, "Cannot accumulate after finalizing."

        for l, accessor in enumerate(self.accessors):
            heads = accessor(model)

            for h, _head in enumerate(heads):
                head = _head.flatten()
                self.first_moment[l, h] += head
                self.second_moment[l, h] += torch.outer(head, head)

        self.num_draws += 1

    def finalize(self):
        self.first_moment /= self.num_draws
        self.second_moment /= self.num_draws
        self.is_finished = True

    def reset(self):
        self.first_moment.zero_()
        self.second_moment.zero_()
        self.num_draws = 0
        self.is_finished = False

    def to_matrix(self):
        # Convert the moments to a covariance matrix.
        covariance = self.second_moment

        for l in range(self.num_layers):
            for h in range(self.num_heads):
                first_moment_head = self.first_moment[l, h]
                covariance[l, h] -= torch.outer(first_moment_head, first_moment_head)

        return covariance

    def to_eigen(self, include_matrix=False):
        # Convert the covariance matrix to pairs of eigenvalues and vectors.
        cov = self.to_matrix().detach().cpu().numpy()
        results = {}

        evals = np.zeros((self.num_evals, self.num_layers, self.num_heads))
        evecs = np.zeros(
            (self.num_evals, self.num_layers, self.num_heads, self.num_weights_per_head)
        )

        for l in range(self.num_layers):
            for h in range(self.num_heads):
                head_cov = cov[l, h]
                head_evals, head_evecs = eigsh(head_cov, k=self.num_evals, which="LM")

                for i in range(self.num_evals):
                    evecs[i, l, h, :] = head_evecs[:, i].reshape(
                        (self.num_weights_per_head,)
                    )
                    evals[i, l, h] = head_evals[i]

        results.update({"evecs": evecs, "evals": evals})

        if include_matrix:
            results["matrix"] = cov

        return results

    def sample(self):
        return self.to_eigen(include_matrix=True)

    def __call__(self, model):
        self.accumulate(model)


LayerWeightsAccessor = Callable[[nn.Module], torch.Tensor]


class BetweenLayerCovarianceAccumulator:
    """
    A CovarianceAccumulator to compute covariance between arbitrary layers.
    For use with `sample`.

    :param model: The model to compute covariances on.
    :param pairs: Named pairs of layers to compute covariances on.
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'.
    :param num_evals: number of eigenvectors / eigenvalues to return.
    :param accessors: Functions to access attention head weights.
    """

    def __init__(
        self,
        model,
        pairs: Dict[str, Tuple[str, str]],
        device: Union[torch.device, str] = "cpu",
        num_evals: int = 3,
        **accessors: LayerWeightsAccessor,
    ):
        self.num_layers = len(accessors)
        self.accessors = accessors
        self.pairs = pairs
        self.num_weights_per_layer = {
            name: len(accessor(model).flatten()) for name, accessor in accessors.items()
        }
        self.first_moments = {
            name: torch.zeros(num_weights, device=device)
            for name, num_weights in self.num_weights_per_layer.items()
        }
        self.second_moments = {
            pair_name: torch.zeros(
                self.num_weights_per_layer[name1],
                self.num_weights_per_layer[name2],
                device=device,
            )
            for pair_name, (name1, name2) in pairs.items()
        }
        self.num_draws = 0
        self.num_evals = num_evals
        self.is_finished = False
        self.device = device

    @property
    def num_weights(self):
        return sum(self.num_weights_per_layer.values())

    def accumulate(self, model: nn.Module):
        # Accumulate moments from model weights.
        assert not self.is_finished, "Cannot accumulate after finalizing."
        weights = {
            name: accessor(model).flatten() for name, accessor in self.accessors.items()
        }

        for name, w in weights.items():
            self.first_moments[name] += w

        for pair_name, (name1, name2) in self.pairs.items():
            self.second_moments[pair_name] += torch.outer(
                weights[name1], weights[name2]
            )

        self.num_draws += 1

    def finalize(self):

        for name in self.accessors:
            self.first_moments[name] /= self.num_draws

        for name in self.pairs:
            self.second_moments[name] /= self.num_draws

        self.is_finished = True

    def reset(self):
        for name in self.accessors:
            self.first_moments[name].zero_()

        for name in self.pairs:
            self.second_moments[name].zero_()

        self.num_draws = 0
        self.is_finished = False

    def to_matrices(self):
        # Convert the moments to a covariance matrix.
        covariances = {}

        for name, (layer1, layer2) in self.pairs.items():
            first_moment1 = self.first_moments[layer1]
            first_moment2 = self.first_moments[layer2]
            covariances[name] = self.second_moments[name] - torch.outer(
                first_moment1, first_moment2
            )

        return covariances

    def to_eigens(self, include_matrix=False):
        # Convert the covariance matrix to pairs of eigenvalues and vectors.
        covariances = {
            k: v.detach().cpu().numpy() for k, v in self.to_matrices().items()
        }
        results = {}

        for name, cov in covariances.items():
            # TODO: U, s, Vt = svds(cov, k=self.num_evals, which='LM')
            evals, evecs = eigsh(cov, k=self.num_evals, which="LM")

            # Reverse the order of the eigenvalues and vectors
            evals = evals[::-1]
            evecs = evecs[:, ::-1]

            results.update({name: {"evecs": evecs, "evals": evals}})

            if include_matrix:
                results[name]["matrix"] = cov

        return results

    def sample(self):
        return self.to_eigens(include_matrix=True)

    def __call__(self, model):
        self.accumulate(model)
