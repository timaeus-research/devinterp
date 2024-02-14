from copy import deepcopy
from typing import Union
import numpy as np
import torch
import torch.nn as nn

from devinterp.slt.callback import SamplerCallback


def mala_acceptance_probability(
    prev_point,
    prev_grad,
    prev_loss,
    current_point,
    current_grad,
    current_loss,
    learning_rate,
):
    """
    Calculate the acceptance probability for a MALA transition.

    Args:
    prev_point: The previous point in parameter space.
    current_point: The current point in parameter space.
    prev_grad: Gradient of the prev point in parameter space.
    current_grad: Gradient of the current point in parameter space.
    prev_loss: Loss of the previous point in parameter space.
    current_loss: Loss of the current point in parameter space.
    learning_rate (float): Learning rate of the model.

    Returns:
    float: Acceptance probability for the proposed transition.
    """
    # Compute the log of the proposal probabilities (using the Gaussian proposal distribution)
    log_q_current_to_prev = -torch.sum(
        (prev_point - current_point - (learning_rate * 0.5 * -current_grad)) ** 2
    ) / (2 * learning_rate)
    log_q_prev_to_current = -torch.sum(
        (current_point - prev_point - (learning_rate * 0.5 * -prev_grad)) ** 2
    ) / (2 * learning_rate)

    # Compute the acceptance probability
    acceptance_log_prob = (
        log_q_current_to_prev - log_q_prev_to_current + prev_loss - current_loss
    )

    return min(1.0, torch.exp(acceptance_log_prob))


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
        num_samples: int,
        model: nn.Module,
        learning_rate: float,
        elasticity: float,
        device: Union[torch.device, str] = "cpu",
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.learning_rate = learning_rate
        self.elasticity = elasticity
        self.num_samples = num_samples
        self.mala_acceptance_rate = torch.zeros(
            (num_chains, num_draws - 1), dtype=torch.float32
        ).to(device)
        self.device = device
        self.current_params = [
            param.clone().detach() for param in list(model.parameters())
        ]
        self.current_grads = [torch.zeros(param.size()) for param in model.parameters()]
        self.prev_params = []
        self.prev_grads = []
        self.prev_mala_loss = 0.0

    def __call__(self, chain: int, draw: int, model: nn.Module, loss: float, optimizer):
        self.update(chain, draw, model, loss, optimizer)

    def update(self, chain: int, draw: int, model: nn.Module, loss: float, optimizer):
        self.current_grads = optimizer.dws
        mala_loss = (
            (loss * self.num_samples / np.log(self.num_samples))
            + torch.pow(optimizer.initial_param_distance, 2) * self.elasticity / 2
        )
        # print(
        #     draw,
        #     "current",
        #     self.current_params,  # MALA param
        #     self.current_grads[-1],  # MALA gradient! nice
        #     mala_loss,  # MALA loss
        #     "prev",
        #     self.prev_params,
        #     self.prev_grads,
        #     self.prev_mala_loss,
        # )
        if draw > 0:

            for current_param, current_grad, prev_param, prev_grad in zip(
                self.current_params,
                self.current_grads,
                self.prev_params,
                self.prev_grads,
            ):
                self.mala_acceptance_rate[chain, draw - 1] = (
                    mala_acceptance_probability(
                        prev_param,
                        prev_grad,
                        self.prev_mala_loss,
                        current_param,
                        current_grad,
                        mala_loss,
                        self.learning_rate,
                    )
                )
                # input(
                #     mala_acceptance_probability(
                #         prev_param,
                #         prev_grad,
                #         self.prev_mala_loss,
                #         current_param,
                #         current_grad,
                #         mala_loss,
                #         self.learning_rate,
                #     )
                # )
        self.prev_params = self.current_params
        self.current_params = [
            param.clone().detach() for param in list(model.parameters())
        ]
        self.prev_grads = self.current_grads
        self.prev_mala_loss = mala_loss

    def sample(self):
        return {
            "mala_accept/trace": self.mala_acceptance_rate.cpu().numpy(),
            "mala_accept/mean": np.mean(self.mala_acceptance_rate.cpu().numpy()),
            "mala_accept/std": np.std(self.mala_acceptance_rate.cpu().numpy()),
        }
