from typing import Union

import torch

from devinterp.slt.callback import SamplerCallback


class OnlineWBICEstimator(SamplerCallback):
    """
    Callback for estimating the Widely Applicable Bayesian Information Criterion (WBIC) in an online fashion.
    The WBIC used here is just n * (average sampled loss). (Watanabe, 2013)

    params:
        num_draws (int): Number of samples to draw. (should be identical to param passed to sample())
        num_chains (int): Number of chains to run. (should be identical to param passed to sample())
        n (torch.Tensor): Number of data points, used in the WBIC calculation.
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        n: int,
        device: Union[torch.device, str] = "cpu",
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws

        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(
            device
        )
        self.wbics = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(
            device
        )

        self.n = torch.tensor(n, dtype=torch.float32).to(device)
        self.wbic_means = torch.zeros(num_draws, dtype=torch.float32).to(device)
        self.wbic_stds = torch.zeros(num_draws, dtype=torch.float32).to(device)

        self.device = device

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss
        mean_loss = self.losses[chain, : draw + 1].mean()
        # using the formula that wbic is estimated by n * (avg sampled loss)
        self.wbics[chain, draw] = mean_loss * self.n

    def finalize(self):
        self.wbic_means = self.wbics.mean(axis=0)
        self.wbic_stds = self.wbics.std(axis=0)

    def sample(self):
        return {
            "wbic/means": self.wbic_means.cpu().numpy(),
            "wbic/stds": self.wbic_stds.cpu().numpy(),
            "wbic/trace": self.wbics.cpu().numpy(),
            "loss/trace": self.losses.cpu().numpy(),
        }

    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)
