import matplotlib.pyplot as plt
import torch

from devinterp.slt.callback import SamplerCallback
from devinterp.slt.llc import OnlineLLCEstimator


class OnlineLossStatistics(SamplerCallback):
    """
    Derivative callback that computes various loss statistics for :func:`~devinterp.slt.llc.OnlineLLCEstimator`. Must
    be called after the base :func:`~devinterp.slt.llc.OnlineLLCEstimator` has been called at each draw.

    .. |colab5| image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/diagnostics.ipynb

    See `the diagnostics notebook <https://www.github.com/timaeus-research/devinterp/blob/main/examples/diagnostics.ipynb>`_ |colab5| for examples on how to use this to diagnose your sample health.

    :param base_callback: Base callback that computes original loss metric.
    :type base_callback: :func:`~devinterp.slt.llc.OnlineLLCEstimator`

    Note:
        - Requires losses to be computed first, so call using f.e. :python:`devinterp.slt.sampler.sample(..., [llc_estimator_instance, ..., online_loss_stats_instance], ...)`
    """

    def __init__(self, base_callback: OnlineLLCEstimator):
        self.base_callback = base_callback

        self.num_chains = base_callback.num_chains
        self.num_draws = base_callback.num_draws

        # relative loss is loss - init loss
        # percentage of draws with negative relative loss
        self.percent_neg_steps = torch.zeros(
            (self.num_chains, self.num_draws), dtype=torch.float32
        )

        # percentage of draw with negative (cumulative) mean relative loss
        self.percent_mean_neg_steps = torch.zeros(
            (self.num_chains, self.num_draws), dtype=torch.float32
        )

        # percentage of draws with relative loss < -estimated_noise
        self.percent_thresholded_neg_steps = torch.zeros(
            (self.num_chains, self.num_draws), dtype=torch.float32
        )

        # measured by num of std devs of init losses
        self.z_scores = torch.zeros(
            (self.num_chains, self.num_draws), dtype=torch.float32
        )

    def __call__(self, chain: int, draw: int, loss: float, **kwargs):
        self.update(chain, draw, loss)

    def update(self, chain: int, draw: int, loss: float):
        init_loss = self.base_callback.init_loss
        estimated_noise = self.est_minibatch_noise
        t = draw + 1
        losses = self.base_callback.losses[chain, :t]
        relative_losses = losses - init_loss

        self.percent_neg_steps[chain, draw] = (relative_losses < 0).sum() / (t)

        prev_percent = self.percent_mean_neg_steps[chain, draw - 1] if draw > 0 else 0
        self.percent_mean_neg_steps[chain, draw] = (
            (t - 1) * prev_percent + (relative_losses.mean() < 0)
        ) / t

        self.percent_thresholded_neg_steps[chain, draw] = (
            relative_losses < -estimated_noise
        ).sum() / t

        # only compute if estimated noise is nonzero; this might not happen if e.g. using same random seed for all chains
        if estimated_noise > 0:
            self.z_scores[chain, draw] = (loss - init_loss) / estimated_noise
        else:
            self.z_scores[chain, draw] = float("nan")

    def get_results(self):
        """
        :returns: A dict :python:`{"loss/percent_neg_steps": percent_neg_steps, "loss/percent_mean_neg_steps": percent_mean_neg_steps, "loss/percent_thresholded_neg_steps": percent_thresholded_neg_steps, "loss/z_scores": z_scores}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [llc_estimator_instance, online_loss_stats_instance], ...)`)
        """
        return {
            "loss/percent_neg_steps": self.percent_neg_steps.cpu().numpy(),
            "loss/percent_mean_neg_steps": self.percent_mean_neg_steps.cpu().numpy(),
            "loss/percent_thresholded_neg_steps": self.percent_thresholded_neg_steps.cpu().numpy(),
            "loss/z_scores": self.z_scores.cpu().numpy(),
        }

    @property
    def est_minibatch_noise(self):
        init_losses = self.base_callback.losses[:, 0]
        return init_losses.std()

    def loss_hist_by_draw(self, draw: int = 0, bins: int = 10):
        """Plots a histogram of chain losses for a given draw index.

        :param draw: Draw index to plot histogram for. Default is 0
        :type draw: int, optional
        :param bins: number of histogram bins. Default is 10
        :type bins: int, optional
        """
        losses_at_draw = self.base_callback.losses[:, draw]
        plt.hist(losses_at_draw, bins=bins)
        return losses_at_draw
