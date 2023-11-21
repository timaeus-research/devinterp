from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from torch import nn

from devinterp.slt.callback import ChainCallback

WeightAccessor = Callable[[nn.Module], torch.Tensor]

class CovarianceAccumulator(ChainCallback):
    """
    A callback to iteratively compute and store the covariance matrix of model weights.
    For use with `estimate`. 

    Attributes:
        num_weights (int): Total number of weights.
        first_moment (torch.Tensor): First moment of weights.
        second_moment (torch.Tensor): Second moment of weights.
        num_draws (int): Number of draws made to accumulate moments.
        accessors (List[WeightAccessor]): Functions to access model weights.
        num_evals (int): Number of eigenvalues to compute.
    """
    def __init__(self, num_weights: int, accessors: List[WeightAccessor], device = "cpu", num_evals=3):
        """
        Initialize the accumulator.
        """        
        self.num_weights = num_weights
        self.first_moment = torch.zeros(num_weights, device=device)
        self.second_moment = torch.zeros(num_weights, num_weights, device=device)
        self.num_draws = 0
        self.accessors = accessors
        self.num_evals = num_evals
        self.is_finished = False

    def accumulate(self, model: nn.Module):
        """Accumulate moments from model weights."""
        assert not self.is_finished, "Cannot accumulate after finalizing."

        weights = torch.cat([accessor(model).view((-1,)) for accessor in self.accessors])
        self.first_moment += weights
        self.second_moment += torch.outer(weights, weights)
        self.num_draws += 1

    def finalize(self):
        """Finalize the moments by dividing by the number of draws."""
        self.first_moment /= self.num_draws
        self.second_moment /= self.num_draws
        self.is_finished = True

    def reset(self):
        """Reset the accumulator."""
        self.first_moment.zero_()
        self.second_moment.zero_()
        self.num_draws = 0
        self.is_finished = False

    def to_matrix(self):
        """Convert the moments to a covariance matrix."""
        return self.second_moment - torch.outer(self.first_moment, self.first_moment)

    def to_eigen(self, include_matrix=False):
        """Convert the covariance matrix to pairs of eigenvalues and vectors."""
        cov = self.to_matrix().detach().cpu().numpy()
        evals, evecs = eigsh(cov, k=self.num_evals, which='LM')

        results = {
            "evals": evals,
            "evecs": evecs
        }

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
    For use with `estimate`.

    Attributes:
        num_heads (int): The number of attention heads.
        num_weights_per_head (int): The number of weights per attention head.
        accessors (List[AttentionHeadWeightsAccessor]): Functions to access attention head weights.
        num_layers (int): The number of layers (= number of accessors).
        num_weights_per_layer (int): The number of weights per layer.
        num_weights (int): The total number of weights.
    """
    def __init__(self, num_heads: int, num_weights_per_head: int, accessors: List[AttentionHeadWeightsAccessor], device = "cpu", num_evals=3):
        self.num_layers = len(accessors)
        self.num_heads = num_heads
        self.num_weights_per_head = num_weights_per_head

        self.first_moment = torch.zeros(self.num_layers, num_heads, self.num_weights_per_head, device=device)
        self.second_moment = torch.zeros(self.num_layers, num_heads, self.num_weights_per_head, self.num_weights_per_head, device=device)
        self.num_draws = 0
        self.accessors = accessors
        self.num_evals = num_evals
        self.is_finished = False

    @property
    def num_weights_per_layer(self):
        """The number of weights per layer."""
        return self.num_heads * self.num_weights_per_head

    @property
    def num_weights(self):
        """The total number of weights."""
        return self.num_layers * self.num_weights_per_layer

    def accumulate(self, model: nn.Module):
        """Accumulate moments from model weights."""
        assert not self.is_finished, "Cannot accumulate after finalizing."

        for l, accessor in enumerate(self.accessors):
            heads = accessor(model)

            for h, _head in enumerate(heads):
                head = _head.flatten()
                self.first_moment[l, h] += head
                self.second_moment[l, h] += torch.outer(head, head)

        self.num_draws += 1

    def finalize(self):
        """Finalize the moments by dividing by the number of draws."""
        self.first_moment /= self.num_draws
        self.second_moment /= self.num_draws
        self.is_finished = True

    def reset(self):
        """Reset the accumulator."""
        self.first_moment.zero_()
        self.second_moment.zero_()
        self.num_draws = 0
        self.is_finished = False

    def to_matrix(self):
        """Convert the moments to a covariance matrix."""
        covariance = self.second_moment

        for l in range(self.num_layers):
            for h in range(self.num_heads):
                first_moment_head = self.first_moment[l, h]
                covariance[l, h] -= torch.outer(first_moment_head, first_moment_head)

        return covariance

    def to_eigen(self, include_matrix=False):
        """Convert the covariance matrix to pairs of eigenvalues and vectors."""
        cov = self.to_matrix().detach().cpu().numpy()
        results = {}

        evals = np.zeros((self.num_evals, self.num_layers, self.num_heads))
        evecs = np.zeros((self.num_evals, self.num_layers, self.num_heads, self.num_weights_per_head))

        for l in range(self.num_layers):
            for h in range(self.num_heads):
                head_cov = cov[l, h]
                head_evals, head_evecs = eigsh(head_cov, k=self.num_evals, which='LM')

                for i in  range(self.num_evals):
                    evecs[i,l,h,:] = head_evecs[:, i].reshape((self.num_weights_per_head,))
                    evals[i,l,h] = head_evals[i]

        results.update({
            "evecs": evecs,
            "evals": evals
        })

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
    For use with `estimate`.
    """
    def __init__(self, model, pairs: Dict[str, Tuple[str, str]], device = "cpu", num_evals=3, **accessors: LayerWeightsAccessor):
        self.num_layers = len(accessors)
        self.accessors = accessors
        self.pairs = pairs
        self.num_weights_per_layer = {name: len(accessor(model).flatten()) for name, accessor in accessors.items()}
        self.first_moments = {name: torch.zeros(num_weights, device=device) for name, num_weights in self.num_weights_per_layer.items()}
        self.second_moments = {pair_name: torch.zeros(self.num_weights_per_layer[name1], self.num_weights_per_layer[name2], device=device) for pair_name, (name1, name2) in pairs.items()}
        self.num_draws = 0
        self.num_evals = num_evals
        self.is_finished = False
        self.device = device

    @property
    def num_weights(self):
        """The total number of weights."""
        return sum(self.num_weights_per_layer.values())

    def accumulate(self, model: nn.Module):
        """Accumulate moments from model weights."""
        assert not self.is_finished, "Cannot accumulate after finalizing."
        weights = {name: accessor(model).flatten() for name, accessor in self.accessors.items()}

        for name, w in weights.items():
            self.first_moments[name] += w

        for pair_name, (name1, name2) in self.pairs.items():
            self.second_moments[pair_name] += torch.outer(weights[name1], weights[name2])

        self.num_draws += 1

    def finalize(self):
        """Finalize the moments by dividing by the number of draws."""

        for name in self.accessors:
            self.first_moments[name] /= self.num_draws
        
        for name in self.pairs:
            self.second_moments[name] /= self.num_draws
        
        self.is_finished = True

    def reset(self):
        """Reset the accumulator."""
        for name in self.accessors:
            self.first_moments[name].zero_()

        for name in self.pairs:
            self.second_moments[name].zero_()

        self.num_draws = 0
        self.is_finished = False
    
    def to_matrices(self):
        """Convert the moments to a covariance matrix."""
        covariances = {}

        for name, (layer1, layer2) in self.pairs.items():
            first_moment1 = self.first_moments[layer1]
            first_moment2 = self.first_moments[layer2]
            covariances[name] = self.second_moments[name] - torch.outer(first_moment1, first_moment2)

        return covariances

    def to_eigens(self, include_matrix=False):
        """Convert the covariance matrix to pairs of eigenvalues and vectors."""
        covariances = {k: v.detach().cpu().numpy() for k, v in self.to_matrices().items()}
        results = {}

        for name, cov in covariances.items():
            # TODO: U, s, Vt = svds(cov, k=self.num_evals, which='LM')
            evals, evecs = eigsh(cov, k=self.num_evals, which='LM')

            # Reverse the order of the eigenvalues and vectors
            evals = evals[::-1]
            evecs = evecs[:, ::-1]

            results.update({
                name: {
                    "evecs": evecs,
                    "evals": evals
                }
            })

            if include_matrix:
                results[name]["matrix"] = cov

        return results

    def sample(self):
        return self.to_eigens(include_matrix=True)
    
    def __call__(self, model):
        self.accumulate(model)
