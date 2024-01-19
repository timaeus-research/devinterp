from typing import Union

import torch
import torch.nn as nn

from devinterp.optim.sgld import SGLD
from devinterp.slt.callback import SamplerCallback


class WeightNorm(SamplerCallback):
    """
    Callback for computing the norm of the weights over the sampling process.

    params:
        num_draws (int): Number of samples to draw. (should be identical to param passed to sample())
        num_chains (int): Number of chains to run. (should be identical to param passed to sample())
        p_norm (int): Order of the norm to be computed (e.g., 2 for Euclidean norm).
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        p_norm: int = 2,
        device: Union[torch.device, str] = "cpu",
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.weight_norms = torch.zeros(
            (num_chains, num_draws), dtype=torch.float32
        ).to(device)
        self.p_norm = p_norm
        self.device = device

    def __call__(self, chain: int, draw: int, model: nn.Module):
        self.update(chain, draw, model)

    def update(self, chain: int, draw: int, model: nn.Module):
        total_norm = torch.tensor(0.0).to(self.device)
        for param in model.parameters():
            total_norm += torch.square(torch.linalg.vector_norm(param, ord=2)).to(
                self.device
            )
        total_norm = torch.pow(total_norm, 1 / self.p_norm)
        self.weight_norms[chain, draw] = total_norm

    def sample(self):
        return {
            "weight_norm/trace": self.weight_norms.cpu().numpy(),
        }


class GradientNorm(SamplerCallback):
    """
    Callback for computing the norm of the gradients of the optimizer / sampler.

    params:
        num_draws (int): Number of samples to draw. (should be identical to param passed to sample())
        num_chains (int): Number of chains to run. (should be identical to param passed to sample())
        p_norm (int): Order of the norm to be computed (e.g., 2 for Euclidean norm).
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        p_norm: int = 2,
        device: Union[torch.device, str] = "cpu",
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.gradient_norms = torch.zeros(
            (num_chains, num_draws), dtype=torch.float32
        ).to(device)
        self.p_norm = p_norm
        self.device = device

    def __call__(self, chain: int, draw: int, model: nn.Module):
        self.update(chain, draw, model)

    def update(self, chain: int, draw: int, model: nn.Module):
        total_norm = torch.tensor(0.0).to(self.device)
        for param in model.parameters():
            total_norm += torch.square(torch.linalg.vector_norm(param.grad, ord=2)).to(
                self.device
            )
        total_norm = torch.pow(total_norm, 1 / self.p_norm)
        self.gradient_norms[chain, draw] = total_norm

    def sample(self):
        return {
            "gradient_norm/trace": self.gradient_norms.cpu().numpy(),
        }


class NoiseNorm(SamplerCallback):
    """
    Callback for computing the norm of the noise added in the optimizer / sampler.

    params:
        num_draws (int): Number of samples to draw. (should be identical to param passed to sample())
        num_chains (int): Number of chains to run. (should be identical to param passed to sample())
        p_norm (int): Order of the norm to be computed (e.g., 2 for Euclidean norm).
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        p_norm: int = 2,
        device: Union[torch.device, str] = "cpu",
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.noise_norms = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(
            device
        )
        self.p_norm = p_norm
        self.device = device

    def __call__(self, chain: int, draw: int, optimizer: SGLD):
        self.update(chain, draw, optimizer)

    def update(self, chain: int, draw: int, optimizer: SGLD):
        total_norm = torch.tensor(0.0).to(self.device)
        for noise in optimizer.noise:
            total_norm += torch.square(torch.linalg.vector_norm(noise, ord=2)).to(
                self.device
            )
        total_norm = torch.pow(total_norm, 1 / self.p_norm)
        self.noise_norms[chain, draw] = total_norm

    def sample(self):
        return {
            "noise_norm/trace": self.noise_norms.cpu().numpy(),
        }
