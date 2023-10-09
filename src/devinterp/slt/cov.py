from typing import Callable, List, Tuple

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from torch import nn

WeightAccessor = Callable[[nn.Module], torch.Tensor]

class CovarianceAccumulator:
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
        self.first_moment = torch.zeros(num_weights, device=device).share_memory_()
        self.second_moment = torch.zeros(num_weights, num_weights, device=device).share_memory_()
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

        self.first_moment = torch.zeros(self.num_layers, num_heads, self.num_weights_per_head, device=device).share_memory_()
        self.second_moment = torch.zeros(self.num_layers, num_heads, self.num_weights_per_head, self.num_weights_per_head, device=device).share_memory_()
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

    def __call__(self, model):
        self.accumulate(model)
