from typing import Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from devinterp.optim.sgld import SGLD
from devinterp.slt.callback import SamplerCallback


class WeightNorm(SamplerCallback):
    """
    Callback for computing the norm of the weights over the sampling process.

    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_chains: int
    :param p_norm: Order of the norm to be computed (e.g., 2 for Euclidean norm). Default is 2
    :type p_norm: int, optional
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'. Default is 'cpu'
    :type device: str | torch.device, optional
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

    def __call__(self, chain: int, draw: int, model: nn.Module, **kwargs):
        self.update(chain, draw, model)

    def update(self, chain: int, draw: int, model: nn.Module):
        total_norm = torch.tensor(0.0).to(self.device)
        for param in model.parameters():
            total_norm += torch.square(
                torch.linalg.vector_norm(param, ord=self.p_norm)
            ).to(self.device)
        total_norm = torch.pow(total_norm, 1 / self.p_norm)
        self.weight_norms[chain, draw] = total_norm

    def get_results(self):
        """:returns: A dict :python:`{"weight_norm/trace": weight_norms}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [weight_norm_instance], ...)`)"""
        return {
            "weight_norm/trace": self.weight_norms.cpu().numpy(),
        }


class GradientNorm(SamplerCallback):
    """
    Callback for computing the norm of the gradients of the optimizer / sampler.

    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_chains: int
    :param p_norm: Order of the norm to be computed (e.g., 2 for Euclidean norm). Default is 2
    :type p_norm: int, optional
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'. Default is 'cpu'
    :type device: str | torch.device, optional
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

    def __call__(self, chain: int, draw: int, optimizer: Optimizer, **kwargs):
        self.update(chain, draw, optimizer.param_groups)

    def update(self, chain: int, draw: int, param_groups: dict):
        total_norm = torch.tensor(0.0).to(self.device)
        for group in param_groups:
            for param in group["params"]:
                total_norm += torch.square(
                    torch.linalg.vector_norm(
                        param.grad * 0.5 * group["lr"], ord=self.p_norm
                    )
                ).to(self.device)
        total_norm = torch.pow(total_norm, 1 / self.p_norm)
        self.gradient_norms[chain, draw] = total_norm

    def get_results(self):
        """:returns: A dict :python:`{"gradient_norm/trace": gradient_norms}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [grad_norm_instance], ...)`)"""
        return {
            "gradient_norm/trace": self.gradient_norms.cpu().numpy(),
        }


class NoiseNorm(SamplerCallback):
    """
    Callback for computing the norm of the noise added in the optimizer / sampler.

    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_chains: int
    :param p_norm: Order of the norm to be computed (e.g., 2 for Euclidean norm). Default is 2
    :type p_norm: int, optional
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'. Default is 'cpu'
    :type device: str | torch.device, optional
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

    def __call__(self, chain: int, draw: int, optimizer: SGLD, **kwargs):
        self.update(chain, draw, optimizer)

    def update(self, chain: int, draw: int, optimizer: SGLD):
        total_norm = torch.tensor(0.0).to(self.device)

        for group_idx, group_noises in optimizer.noise.items():
            for noise in group_noises:
                total_norm += torch.square(
                    torch.linalg.vector_norm(
                        noise * optimizer.param_groups[group_idx]["lr"] ** 0.5,
                        ord=self.p_norm,
                    )
                ).to(self.device)
        total_norm = torch.pow(total_norm, 1 / self.p_norm)
        self.noise_norms[chain, draw] = total_norm

    def get_results(self):
        """:returns: A dict :python:`{"noise_norm/trace": noise_norms}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [noise_norm_instance], ...)`)"""
        return {
            "noise_norm/trace": self.noise_norms.cpu().numpy(),
        }
