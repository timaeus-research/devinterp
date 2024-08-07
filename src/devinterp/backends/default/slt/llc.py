from typing import Union

import torch

from devinterp.slt.callback import SamplerCallback


class LLCEstimator(SamplerCallback):
    r"""
    Callback for estimating the Local Learning Coefficient (LLC) in a rolling fashion during a sampling process.
    It calculates the LLC based on the average loss across draws for each chain:
    
    $$LLC = \textrm{n \beta} * (\textrm{avg_loss} - \textrm{init_loss})$$

    For use with :func:`devinterp.slt.sampler.sample`.
    
    Note: 
        - `init_loss` gets set inside :func:`devinterp.slt.sample()`. It can be passed as an argument to that function, 
        and if not passed will be the average loss of the supplied model over `num_chains` batches.

    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_chains: int
    :param nbeta: Effective Inverse Temperature, float (default: 1., set by sample() to utils.optimal_nbeta(dataloader)=len(batch_size)/np.log(len(batch_size)))
    :type nbeta: int
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'.
    :type device: str | torch.device, optional
    """
    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        nbeta: float,
        init_loss: torch.Tensor = None,
        device: Union[torch.device, str] = "cpu",
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(
            device
        )
        self.init_loss = init_loss or torch.zeros(1, dtype=torch.float32).to(device)
        self.nbeta = torch.tensor(nbeta, dtype=torch.float32).to(device)
        self.llc_per_chain = torch.zeros(num_chains, dtype=torch.float32).to(device)
        self.llc_mean = torch.tensor(0.0, dtype=torch.float32).to(device)
        self.llc_std = torch.tensor(0.0, dtype=torch.float32).to(device)

        self.device = device

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss

    def finalize(self):
        avg_losses = self.losses.mean(axis=1)
        self.llc_per_chain = self.nbeta * (avg_losses - self.init_loss)
        self.llc_mean = self.llc_per_chain.mean()
        self.llc_std = self.llc_per_chain.std()

    def sample(self):
        """
        :returns: A dict :python:`{"llc/mean": llc_mean, "llc/std": llc_std, "llc-chain/{i}": llc_trace_per_chain, "loss/trace": loss_trace_per_chain}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [llc_estimator_instance], ...)`).
        """
        return {
            "llc/mean": self.llc_mean.cpu().numpy().item(),
            "llc/std": self.llc_std.cpu().numpy().item(),
            **{
                f"llc-chain/{i}": self.llc_per_chain[i].cpu().numpy().item()
                for i in range(self.num_chains)
            },
            "loss/trace": self.losses.cpu().numpy(),
        }

    def __call__(self, chain: int, draw: int, loss: float, **kwargs):
        self.update(chain, draw, loss)


class OnlineLLCEstimator(SamplerCallback):
    """
    Callback for estimating the Local Learning Coefficient (LLC) in an online fashion during a sampling process.
    It calculates LLCs using the same formula as :func:`devinterp.slt.llc.LLCEstimator`, but continuously and including means and std across draws (as opposed to just across chains).
    For use with :func:`devinterp.slt.sampler.sample`.
    
    Note: 
        - `init_loss` gets set inside :func:`devinterp.slt.sample()`. It can be passed as an argument to that function, 
        and if not passed will be the average loss of the supplied model over `num_chains` batches.
    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_chains: int
    :param nbeta: Effective Inverse Temperature, float (default: 1., set by sample() to utils.optimal_nbeta(dataloader)=len(batch_size)/np.log(len(batch_size)))
    :type nbeta: int
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'. Default is 'cpu'
    :type device: str | torch.device, optional
    """

    def __init__(
        self, num_chains: int, num_draws: int, nbeta: float, init_loss,
 device="cpu"
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.init_loss = init_loss # gets set by devinterp.slt.sample()

        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(
            device
        )
        self.llcs = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.moving_avg_llcs = torch.zeros(
            (num_chains, num_draws), dtype=torch.float32
        ).to(device)

        self.nbeta = nbeta

        self.llc_means = torch.tensor(num_draws, dtype=torch.float32).to(device)
        self.llc_stds = torch.tensor(num_draws, dtype=torch.float32).to(device)

        self.device = device

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss
        self.llcs[chain, draw] = self.nbeta * (loss - self.init_loss)
        if draw == 0:
            self.moving_avg_llcs[chain, draw] = self.nbeta * (
                loss - self.init_loss
            )
        else:
            t = draw + 1
            prev_moving_avg = self.moving_avg_llcs[chain, draw - 1]
            self.moving_avg_llcs[chain, draw] = (1 / t) * (
                (t - 1) * prev_moving_avg + self.nbeta * (loss - self.init_loss)
            )


    def finalize(self):
        self.llc_means = self.llcs.mean(dim=0)
        self.llc_stds = self.llcs.std(dim=0)

    def sample(self):
        """    
        :returns: A dict :python:`{"llc/means": llc_means, "llc/stds": llc_stds, "llc/trace": llc_trace_per_chain, "loss/trace": loss_trace_per_chain}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [llc_estimator_instance], ...)`).
        """
        return {
            "llc/means": self.llc_means.cpu().numpy(),
            "llc/moving_avg": self.moving_avg_llcs.cpu().numpy(),
            "llc/stds": self.llc_stds.cpu().numpy(),
            "llc/trace": self.llcs.cpu().numpy(),
            "loss/trace": self.losses.cpu().numpy(),
        }

    def __call__(self, chain: int, draw: int, loss: float, **kwargs):
        self.update(chain, draw, loss)
