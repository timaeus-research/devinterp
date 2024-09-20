import warnings
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from devinterp.slt.callback import SamplerCallback


def mala_acceptance_probability(
    prev_params: Union[Tensor, List[Tensor]],
    prev_grads: Union[Tensor, List[Tensor]],
    prev_loss: Tensor,
    current_params: Union[Tensor, List[Tensor]],
    current_grads: Union[Tensor, List[Tensor]],
    current_loss: Tensor,
    learning_rate: float,
) -> float:
    """
    Calculate the acceptance probability for a MALA transition. Parameters and
    gradients can either all be given as a tensor (all of the same shape) or
    all as lists of tensors (eg the parameters of a Module).

    Args:
    prev_params: The previous point in parameter space.
    prev_grads: Gradient of the prev point in parameter space.
    prev_loss: Loss of the previous point in parameter space.
    current_params: The current point in parameter space.
    current_grads: Gradient of the current point in parameter space.
    current_loss: Loss of the current point in parameter space.
    learning_rate (float): Learning rate of the model.

    Returns:
    float: Acceptance probability for the proposed transition.
    """
    if current_loss is np.array:
        current_loss = torch.tensor(current_loss)

    if torch.isnan(current_loss):
        return np.nan

    # convert tensors to lists with one element
    if not isinstance(prev_params, list):
        prev_params = [prev_params]
    if not isinstance(prev_grads, list):
        prev_grads = [prev_grads]
    if not isinstance(current_params, list):
        current_params = [current_params]
    if not isinstance(current_grads, list):
        current_grads = [current_grads]

    log_q_current_to_prev = 0
    log_q_prev_to_current = 0
    for current_point, current_grad, prev_point, prev_grad in zip(
        current_params,
        current_grads,
        prev_params,
        prev_grads,
    ):
        # Compute the log of the proposal probabilities (using the Gaussian proposal distribution)
        log_q_current_to_prev += -torch.sum(
            (prev_point - current_point - (learning_rate * 0.5 * -current_grad)) ** 2
        ) / (2 * learning_rate)
        log_q_prev_to_current += -torch.sum(
            (current_point - prev_point - (learning_rate * 0.5 * -prev_grad)) ** 2
        ) / (2 * learning_rate)

    acceptance_log_prob = (
        log_q_current_to_prev - log_q_prev_to_current + prev_loss - current_loss
    )

    return min(1.0, torch.exp(acceptance_log_prob))


class MalaAcceptanceRate(SamplerCallback):
    """
    Callback for computing MALA acceptance rate.

    Attributes:
        num_draws (int): Number of samples to draw. (should be identical to param passed to sample())
        num_chains (int): Number of chains to run. (should be identical to param passed to sample())
        nbeta (float): Effective Inverse Temperature used to calculate the LLC.
        learning_rate (int): Learning rate of the model.
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        learning_rate: float,
        device: Union[torch.device, str] = "cpu",
        nbeta: float = None,
        temperature: Optional[float] = None,
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.learning_rate = learning_rate
        if nbeta is None:
            assert temperature is not None, "Please provide a value for nbeta."
            self.nbeta = temperature
            warnings.warn("Temperature is deprecated. Please use nbeta instead.")
        else:
            self.nbeta = nbeta

        self.mala_acceptance_rate = torch.zeros(
            (num_chains, num_draws - 1), dtype=torch.float32
        ).to(device)
        self.device = device
        self.current_params = []
        self.current_grads = []
        self.prev_params = []
        self.prev_grads = []
        self.prev_mala_loss = 0.0

    def __call__(
        self, chain: int, draw: int, model: nn.Module, loss: float, optimizer, **kwargs
    ):
        self.update(chain, draw, model, loss, optimizer)

    def update(self, chain: int, draw: int, model: nn.Module, loss: float, optimizer):
        # we need the grads & loss from the pass, but the current params are from after the step
        # (so we update those only after the calculation)
        self.current_grads = optimizer.dws
        # mala acceptance loss is different from pytorch supplied loss
        mala_loss = (loss * self.nbeta) + optimizer.localization_loss
        if draw > 1:
            self.mala_acceptance_rate[chain, draw - 1] = mala_acceptance_probability(
                self.prev_params,
                self.prev_grads,
                self.prev_mala_loss,
                self.current_params,
                self.current_grads,
                mala_loss,
                self.learning_rate,
            )
        # move new -> old, then update new after
        self.prev_params = self.current_params
        self.prev_grads = self.current_grads
        self.prev_mala_loss = mala_loss
        # params update only at the end, as decribed
        self.current_params = [
            param.clone().detach()
            for param in model.parameters()
            if param.requires_grad
        ]

    def get_results(self):
        return {
            "mala_accept/trace": self.mala_acceptance_rate.cpu().numpy(),
            "mala_accept/mean": np.mean(self.mala_acceptance_rate.cpu().numpy()),
            "mala_accept/std": np.std(self.mala_acceptance_rate.cpu().numpy()),
        }
