from typing import List, Union
import math
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn

from devinterp.slt.callback import SamplerCallback


class GradientDistribution(SamplerCallback):
    """
    Callback for plotting the distribution of gradients as a function of draws. Does some magic to automatically adjust bins as more draws are taken.
    For use with :func:`devinterp.slt.sampler.sample`.

    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :param num_chains: int
    :param min_bins: Minimum number of bins for histogram approximation. Default is 20
    :type min_bins: int, optional
    :param param_names: List of parameter names to track. If None, all parameters are tracked. Default is None
    :type param_names: List[str], optional
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'.
    :type device: : str | torch.device, optional

    Raises:
        ValueError: If gradients are not computed before calling this callback.
    """

    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        min_bins: int = 20,
        param_names: List[str] = None,
        device: Union[torch.device, str] = "cpu",
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.grad_dists = {}

        self.min_bins = min_bins
        self.param_names = param_names

        self.bin_size = torch.tensor(0.0).to(device)
        self.num_bins = torch.tensor(0).to(device)
        self.min_grad = torch.tensor(0.0).to(device)

        self.device = device

    @property
    def max_grad(self):
        return self.min_grad + self.bin_size * self.num_bins

    def __call__(self, chain: int, draw: int, model: nn.Module):
        self.update(chain, draw, model)

    # Updates the gradient histograms for each parameter.
    def update(self, chain: int, draw: int, model: nn.Module):
        for param_name, param in model.named_parameters():
            if self.param_names is not None and param_name not in self.param_names:
                continue
            if param.grad is None:
                raise ValueError(
                    f"GradientDistribution callback requires gradients to be computed first"
                )
            self._update_param_bins(
                chain, draw, param_name, param.grad.detach().flatten()
            )

    def _update_param_bins(
        self, chain: int, draw: int, param_name: str, grads: torch.Tensor
    ):
        # init bins, estimating a good bin size using the first draw of the first params
        if self.num_bins == 0:
            self._init_bins(grads)
        if param_name not in self.grad_dists:
            self.grad_dists[param_name] = torch.zeros(
                (self.num_chains, self.num_draws, self.num_bins), dtype=torch.int64
            ).to(self.device)
        max_grad = grads.max()
        min_grad = grads.min()
        # extend bins as necessary to include new min/max grad values
        if min_grad < self.min_grad:
            self._extend_bins(min_grad, self.max_grad)
        if max_grad > self.max_grad:
            self._extend_bins(self.min_grad, max_grad)
        # merge bins as necessary so the num of bins is at most 2 * min_bins
        self._merge_bins()
        # update bins with new grads
        for grad in grads:
            bin_idx = math.floor((grad - self.min_grad) / self.bin_size)
            # correct bin idx for the max grad value
            bin_idx = bin_idx - 1 if bin_idx == self.num_bins else bin_idx
            self.grad_dists[param_name][chain, draw, bin_idx] += 1

    def _init_bins(self, grads: torch.Tensor):
        _, bin_ends = torch.histogram(grads.to("cpu"), bins=self.min_bins)
        self.bin_size = bin_ends[1] - bin_ends[0]
        self.num_bins = self.min_bins
        self.min_grad = bin_ends[0]

    def _extend_bins(self, new_min: torch.Tensor, new_max: torch.Tensor):
        # extend bins to cover new_min and new_max
        min_diff = abs(new_min - self.min_grad)
        num_new_min_bins = math.ceil(min_diff / self.bin_size)
        max_diff = abs(new_max - self.max_grad)
        num_new_max_bins = math.ceil(max_diff / self.bin_size)
        if num_new_min_bins:
            for param_name in self.grad_dists:
                zeros = torch.zeros(
                    (self.num_chains, self.num_draws, num_new_min_bins),
                    dtype=torch.int64,
                ).to(self.device)
                self.grad_dists[param_name] = torch.cat(
                    [zeros, self.grad_dists[param_name]], dim=2
                )
        if num_new_max_bins:
            for param_name in self.grad_dists:
                zeros = torch.zeros(
                    (self.num_chains, self.num_draws, num_new_max_bins),
                    dtype=torch.int64,
                ).to(self.device)
                self.grad_dists[param_name] = torch.cat(
                    [self.grad_dists[param_name], zeros], dim=2
                )
        self.min_grad -= num_new_min_bins * self.bin_size
        self.num_bins += num_new_min_bins + num_new_max_bins

    def _merge_bins(self):
        while self.num_bins > 2 * self.min_bins:
            new_bin_count = math.ceil(self.num_bins / 2)
            for param_name in self.grad_dists:
                new_grad_dists = torch.zeros(
                    (self.num_chains, self.num_draws, new_bin_count), dtype=torch.int64
                ).to(self.device)
                for i in range(self.num_bins):
                    new_grad_dists[:, :, i // 2] += self.grad_dists[param_name][:, :, i]
                self.grad_dists[param_name] = new_grad_dists
            self.num_bins = new_bin_count
            self.bin_size *= 2

    def get_results(self):
        """
        :returns: A dict :python:`{"gradient/distributions": grad_dists}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [gradient_dist_instance], ...)`)
        """
        return {
            "gradient/distributions": self.grad_dists,
        }

    def plot(self, param_name: str, color="blue", plot_zero=True, chain: int = None):
        """Plots the gradient distribution for a specific parameter.
        
        :param param_name: the name of the parameter plot gradients for.
        :type param_name: str
        :param color: The color to plot gradient bins in. Default is blue
        :type color: str, optional
        :param plot_zero: Whether to plot the line through y=0. Default is True
        :type plot_zero: bool, optional
        :param chain: The model to compute covariances on.
        :type chain: int, optional
        
        :returns: None, but shows the denisty gradient bins over sampling steps.
        """
        grad_dist = self.grad_dists[param_name]
        if chain is not None:
            max_count = grad_dist[chain].max()
        else:
            max_count = grad_dist.sum(axis=0).max()

        def get_color_alpha(count):
            if count == 0:
                return torch.tensor(0).to(self.device)
            min_alpha = 0.35
            max_alpha = 0.85
            return (count / max_count) * (max_alpha - min_alpha) + min_alpha

        def build_rect(count, bin_min, bin_max, draw):
            alpha = get_color_alpha(count)
            pos = (draw, bin_min)
            height = bin_max - bin_min
            width = 1
            return plt.Rectangle(
                pos,
                width,
                height,
                color=color,
                alpha=alpha.cpu().numpy().item(),
                linewidth=0,
            )

        _, ax = plt.subplots()
        patches = []
        for draw in range(self.num_draws):
            for pos in range(self.num_bins):
                bin_min = self.min_grad + pos * self.bin_size
                bin_max = bin_min + self.bin_size
                if chain is None:
                    count = grad_dist[:, draw, pos].sum()
                else:
                    count = grad_dist[chain, draw, pos]
                if count != 0:
                    rect = build_rect(count, bin_min, bin_max, draw)
                    patches.append(rect)
        patches = PatchCollection(patches, match_original=True)
        ax.add_collection(patches)

        # note that these y min/max values are relative to *all* gradients, not just the ones for this param
        y_min = self.min_grad
        y_max = self.max_grad
        # ensure the 0 line is visible
        y_min = y_min if y_min < 0 else -y_max
        y_max = y_max if y_max > 0 else -y_min
        plt.ylim(y_min, y_max)

        plt.xlim(0, self.num_draws)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        if plot_zero:
            plt.axhline(color="black", linestyle=":", linewidth=1)

        plt.xlabel("Sampler steps")
        plt.ylabel("gradient distribution")
        plt.title(f"Distribution of {param_name} gradients at each sampler step")
        plt.show()
