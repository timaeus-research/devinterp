from typing import Union

import torch

from devinterp.slt.callback import SamplerCallback


class OnlineTraceStatistics(SamplerCallback):
    """
    Derivative callback that computes mean/std statistics of a specified trace online. Must be called after the base callback has been called at each draw.

    .. |colab5| image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/diagnostics.ipynb

    See `the diagnostics notebook <https://www.github.com/timaeus-research/devinterp/blob/main/examples/diagnostics.ipynb>`_ |colab5| for examples on how to use this to diagnose your sample health.

    :param base_callback: Base callback that computes some metric.
    :type base_callback: :func:`~devinterp.slt.callback.SamplerCallback`
    :param attribute: Name of attribute to compute which mean/std statistics of.
    :type attribute: str
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'. Default is 'cpu'
    :type device: str | torch.device, optional

    :raises: ValueError if underlying trace does not have the requested :python:`attribute`, :python:`num_chains` or :python:`num_draws`.

    Note:
        - Requires base trace stats to be computed first, so call using f.e. :python:`devinterp.slt.sampler.sample(..., [weight_norm_instance, online_trace_stats_instance], ...)`
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

    def get_results(self):
        """
        :returns: A dict :python:`"{self.attribute}/chain/mean": mean_attribute_by_chain, "{self.attribute}/chain/std": std_attribute_by_chain, "{self.attribute}/draw/mean": mean_attribute_by_draw, "{self.attribute}/draw/std": std_attribute_by_draw}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [some_thing_to_calc_stats_of, ..., trace_stats_instance], ...)`).
        """
        return {
            f"{self.attribute}/chain/mean": self.mean_by_chain.cpu().numpy(),
            f"{self.attribute}/chain/std": self.std_by_chain.cpu().numpy(),
            f"{self.attribute}/draw/mean": self.mean_by_draw.cpu().numpy(),
            f"{self.attribute}/draw/std": self.std_by_draw.cpu().numpy(),
        }

    def sample_at_draw(self, draw=-1):
        """
        :param draw: draw index to return stats at. Default is -1
        :type draw: int, optional
        :returns: A dict :python:`"{self.attribute}/chain/mean": mean_attribute_of_draw_by_chain, "{self.attribute}/chain/std": std_attribute_of_draw_by_chain, "{self.attribute}/draw/mean": mean_attribute_of_draw, "{self.attribute}/draw/std": std_attribute_of_draw}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [some_thing_to_calc_stats_of, ..., trace_stats_instance], ...)`).
        """
        return {
            f"{self.attribute}/chain/mean": self.mean_by_chain[:, draw].cpu().numpy(),
            f"{self.attribute}/chain/std": self.std_by_chain[:, draw].cpu().numpy(),
            f"{self.attribute}/draw/mean": self.mean_by_draw[draw].cpu().numpy(),
            f"{self.attribute}/draw/std": self.std_by_draw[draw].cpu().numpy(),
        }

    def __call__(self, draw: int, **kwargs):
        attribute = getattr(self.base_callback, self.attribute)
        self.mean_by_chain[:, draw] = attribute[:, : draw + 1].mean(axis=1)
        self.std_by_chain[:, draw] = attribute[:, : draw + 1].std(axis=1)
        self.mean_by_draw[draw] = attribute[:, draw].mean()
        self.std_by_draw[draw] = attribute[:, draw].std()
