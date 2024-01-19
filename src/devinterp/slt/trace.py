from typing import Union

import torch

from devinterp.slt.callback import SamplerCallback


class OnlineTraceStatistics(SamplerCallback):
    """
    Derivative callback that computes mean/std statistics of a specified trace online. Must
    be called after the base callback has been called at each draw.
    Parameters:
        base_callback (ChainCallback): Base callback that computes original trace metric online.
    """

    def __init__(
        self,
        base_callback: SamplerCallback,
        attribute: str,
        device: Union[torch.device, str] = "cpu",
    ):
        self.base_callback = base_callback
        self.attribute = attribute
        self.validate_base_callback()

        self.attribute = attribute

        self.num_chains = base_callback.num_chains
        self.num_draws = base_callback.num_draws

        self.mean_by_chain = torch.zeros(
            (self.num_chains, self.num_draws), dtype=torch.float32
        ).to(device)
        self.std_by_chain = torch.zeros(
            (self.num_chains, self.num_draws), dtype=torch.float32
        ).to(device)

        self.mean_by_draw = torch.zeros(self.num_draws, dtype=torch.float32).to(device)
        self.std_by_draw = torch.zeros(self.num_draws, dtype=torch.float32).to(device)

    def validate_base_callback(self):
        if not hasattr(self.base_callback, self.attribute):
            raise ValueError(f"Base callback must have attribute {self.attribute}")
        if not hasattr(self.base_callback, "num_chains"):
            raise ValueError("Base callback must have attribute num_chains")
        if not hasattr(self.base_callback, "num_draws"):
            raise ValueError("Base callback must have attribute num_draws")

    def sample(self):
        return {
            f"{self.attribute}/chain/mean": self.mean_by_chain.cpu().numpy(),
            f"{self.attribute}/chain/std": self.std_by_chain.cpu().numpy(),
            f"{self.attribute}/draw/mean": self.mean_by_draw.cpu().numpy(),
            f"{self.attribute}/draw/std": self.std_by_draw.cpu().numpy(),
        }

    def sample_at_draw(self, draw=-1):
        return {
            f"{self.attribute}/chain/mean": self.mean_by_chain[:, draw].cpu().numpy(),
            f"{self.attribute}/chain/std": self.std_by_chain[:, draw].cpu().numpy(),
            f"{self.attribute}/draw/mean": self.mean_by_draw[draw].cpu().numpy(),
            f"{self.attribute}/draw/std": self.std_by_draw[draw].cpu().numpy(),
        }

    def __call__(self, draw: int):
        attribute = getattr(self.base_callback, self.attribute)
        self.mean_by_chain[:, draw] = attribute[:, : draw + 1].mean(axis=1)
        self.std_by_chain[:, draw] = attribute[:, : draw + 1].std(axis=1)
        self.mean_by_draw[draw] = attribute[:, draw].mean()
        self.std_by_draw[draw] = attribute[:, draw].std()
