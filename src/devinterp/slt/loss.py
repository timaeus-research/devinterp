import matplotlib.pyplot as plt
import torch

from devinterp.slt.callback import SamplerCallback
from devinterp.slt.llc import OnlineLLCEstimator


class OnlineLossStatistics(SamplerCallback):
    """
    Derivative callback that computes various loss statistics for OnlineLLCEstimator. Must
    be called after the base OnlineLLCEstimator has been called at each draw.
    Parameters:
        base_callback (OnlineLLCEstimator): Base callback that computes original loss metric.
    """
    def __init__(self, base_callback: OnlineLLCEstimator):
        self.base_callback = base_callback

        self.num_chains = base_callback.num_chains
        self.num_draws = base_callback.num_draws

        # relative loss is loss - init loss
        # percentage of draws with negative relative loss
        self.percent_neg_steps = torch.zeros((self.num_chains, self.num_draws), dtype=torch.float32)

        # percentage of draw with negative (cumulative) mean relative loss
        self.percent_mean_neg_steps = torch.zeros((self.num_chains, self.num_draws), dtype=torch.float32)

        # percentage of draws with relative loss < -estimated_noise
        self.percent_thresholded_neg_steps = torch.zeros((self.num_chains, self.num_draws), dtype=torch.float32)

        # measured by num of std devs of init losses
        self.z_scores = torch.zeros((self.num_chains, self.num_draws), dtype=torch.float32)

    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)
    
    def update(self, chain: int, draw: int, loss: float):
        init_loss = self.base_callback.init_loss
        estimated_noise = self.est_minibatch_noise
        t = draw + 1
        losses = self.base_callback.losses[chain, :t]
        relative_losses = losses - init_loss

        self.percent_neg_steps[chain, draw] = (relative_losses < 0).sum() / (t)

        prev_percent = self.percent_mean_neg_steps[chain, draw-1] if draw > 0 else 0
        self.percent_mean_neg_steps[chain, draw] = ((t - 1) * prev_percent + (relative_losses.mean() < 0)) / t

        self.percent_thresholded_neg_steps[chain, draw] = (relative_losses < -estimated_noise).sum() / t

        # only compute if estimated noise is nonzero; this might not happen if e.g. using same random seed for all chains
        if estimated_noise > 0:
            self.z_scores[chain, draw] = (loss - init_loss) / estimated_noise
        else:
            self.z_scores[chain, draw] = float('nan')
    
    def sample(self):
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
        losses_at_draw = self.base_callback.losses[:, draw]
        plt.hist(losses_at_draw, bins=bins)
        return losses_at_draw
    