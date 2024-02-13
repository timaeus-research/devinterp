from copy import deepcopy
from typing import Union
import numpy as np
import torch
import torch.nn as nn

from devinterp.slt.callback import SamplerCallback


def mala_acceptance_probability(
    current_point,
    proposed_point,
    current_grad,
    proposed_grad,
    current_loss,
    proposed_loss,
    learning_rate,
):
    """
    Calculate the acceptance probability for a MALA transition.

    Args:
    current_point: The current point in parameter space.
    proposed_point: The proposed point in parameter space.
    current_grad: Gradient of the current point in parameter space.
    proposed_grad: Gradient of the proposed point in parameter space.
    current_loss: Loss of the current point in parameter space.
    proposed_loss: Loss of the proposed point in parameter space.
    learning_rate (float): Learning rate of the model.

    Returns:
    float: Acceptance probability for the proposed transition.
    """

    # Compute the log of the proposal probabilities (using the Gaussian proposal distribution)
    log_q_proposed_to_current = -torch.sum(
        (current_point - proposed_point - (learning_rate * 0.5 * -proposed_grad)) ** 2
    ) / (2 * learning_rate)
    log_q_current_to_proposed = -torch.sum(
        (proposed_point - current_point - (learning_rate * 0.5 * -current_grad)) ** 2
    ) / (2 * learning_rate)

    # Compute the acceptance probability
    acceptance_log_prob = (
        log_q_proposed_to_current
        - log_q_current_to_proposed
        + current_loss
        - proposed_loss
    )
    return torch.minimum(1.0, torch.exp(acceptance_log_prob))


class MalaAcceptanceRate(SamplerCallback):
    """
    Callback for computing the norm of the gradients of the optimizer / sampler.

    Attributes:
        num_draws (int): Number of samples to draw. (should be identical to param passed to sample())
        num_chains (int): Number of chains to run. (should be identical to param passed to sample())
        learning_rate (int): Learning rate of the model.
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        learning_rate: float,
        device: Union[torch.device, str] = "cpu",
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.learning_rate = learning_rate
        self.mala_acceptance_rate = torch.zeros(
            (num_chains, num_draws - 1), dtype=torch.float32
        ).to(device)
        self.device = device

    def __call__(self, chain: int, draw: int, model: nn.Module):
        self.update(chain, draw, model)

    def update(self, chain: int, draw: int, model: nn.Module, loss: float):
        if draw > 0:
            for param, prev_param in zip(
                model.parameters(), self.prev_model.parameters()
            ):
                self.mala_acceptance_rate[chain, draw - 1] = (
                    mala_acceptance_probability(
                        prev_param,
                        param,
                        prev_param.grad,
                        param.grad,
                        self.learning_rate,
                        self.prev_loss,
                        loss,
                    )
                )
        self.prev_model = deepcopy(model)
        self.prev_loss = loss

    def sample(self):
        return {
            "mala_accept/trace": self.gradient_norms.cpu().numpy(),
            "mala_accept/mean": np.mean(self.gradient_norms.cpu().numpy()),
        }
