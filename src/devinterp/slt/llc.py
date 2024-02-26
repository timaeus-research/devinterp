from typing import Union

import torch

from devinterp.slt.callback import SamplerCallback


class LLCEstimator(SamplerCallback):
    """
    Callback for estimating the Local Learning Coefficient (LLC) in a rolling fashion during a sampling process.
    It calculates the LLC based on the average loss across draws for each chain:
    $$
    TODO
    $$

    Attributes:
        num_chains (int): Number of chains to run. (should be identical to param passed to sample())
        num_draws (int): Number of samples to draw. (should be identical to param passed to sample())
        temperature (float): Temperature used to calculate the LLC.
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        temperature: float,
        device: Union[torch.device, str] = "cpu",
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(
            device
        )
        self.init_loss = 0.0

        self.temperature = torch.tensor(temperature, dtype=torch.float32).to(device)
        self.llc_per_chain = torch.zeros(num_chains, dtype=torch.float32).to(device)
        self.llc_mean = torch.tensor(0.0, dtype=torch.float32).to(device)
        self.llc_std = torch.tensor(0.0, dtype=torch.float32).to(device)

        self.device = device

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss

    def finalize(self):
        avg_losses = self.losses.mean(axis=1)
        self.llc_per_chain = self.temperature * (avg_losses - self.init_loss)
        self.llc_mean = self.llc_per_chain.mean()
        self.llc_std = self.llc_per_chain.std()

    def sample(self):
        return {
            "llc/mean": self.llc_mean.cpu().numpy().item(),
            "llc/std": self.llc_std.cpu().numpy().item(),
            **{
                f"llc-chain/{i}": self.llc_per_chain[i].cpu().numpy().item()
                for i in range(self.num_chains)
            },
            "loss/trace": self.losses.cpu().numpy(),
        }

    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)


class OnlineLLCEstimator(SamplerCallback):
    """
    Callback for estimating the Local Learning Coefficient (LLC) in an online fashion during a sampling process.
    It calculates LLCs using the same formula as LLCEstimator, but continuously and including means and std across draws (as opposed to just across chains).

    Attributes:
        num_chains (int): Number of chains to run. (should be identical to param passed to sample())
        num_draws (int): Number of samples to draw. (should be identical to param passed to sample())
        temperature (float): Temperature used to calculate the LLC.
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.
    """

    def __init__(
        self, num_chains: int, num_draws: int, temperature: float, device="cpu"
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.init_loss = 0.0

        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(
            device
        )
        self.llcs = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.moving_avg_llcs = torch.zeros(
            (num_chains, num_draws), dtype=torch.float32
        ).to(device)

        self.temperature = temperature

        self.llc_means = torch.tensor(num_draws, dtype=torch.float32).to(device)
        self.llc_stds = torch.tensor(num_draws, dtype=torch.float32).to(device)

        self.device = device

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss
        self.llcs[chain, draw] = self.temperature * (loss - self.init_loss)
        if draw == 0:
            self.moving_avg_llcs[chain, draw] = self.temperature * (
                loss - self.init_loss
            )
        else:
            t = draw + 1
            prev_moving_avg = self.moving_avg_llcs[chain, draw - 1]
            self.moving_avg_llcs[chain, draw] = (1 / t) * (
                (t - 1) * prev_moving_avg + self.temperature * (loss - self.init_loss)
            )

    def finalize(self):
        self.llc_means = self.llcs.mean(dim=0)
        self.llc_stds = self.llcs.std(dim=0)

    def sample(self):
        return {
            "llc/means": self.llc_means.cpu().numpy(),
            "llc/moving_avg": self.moving_avg_llcs.cpu().numpy(),
            "llc/stds": self.llc_stds.cpu().numpy(),
            "llc/trace": self.llcs.cpu().numpy(),
            "loss/trace": self.losses.cpu().numpy(),
        }

    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)
