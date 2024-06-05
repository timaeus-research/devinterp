from typing import Union
from slms.analysis.slt.sampler import ChainHealthError


import numpy as np
import torch
import torch_xla.core.xla_model as xm
from devinterp.slt.callback import SamplerCallback


import warnings


class LLCEstimator(SamplerCallback):
    r"""
    Callback for estimating the Local Learning Coefficient (LLC) in a rolling fashion during a sampling process.
    It calculates the LLC based on the average loss across draws for each chain:

    $$LLC = \textrm{T} * (\textrm{avg_loss} - \textrm{init_loss})$$

    For use with :func:`devinterp.slt.sampler.sample`.

    Note:
        - `init_loss` gets set inside :func:`devinterp.slt.sample()`. It can be passed as an argument to that function,
        and if not passed will be the average loss of the supplied model over `num_chains` batches.

    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_chains: int
    :param nbeta: nbeta, float (default: 1., set by sample() to utils.optimal_nbeta(dataloader)=len(batch_size)/np.log(len(batch_size)))
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
        eval_field: str = "loss",
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.loss = torch.zeros(1, dtype=torch.float32).to(device)
        self.sq_loss = torch.zeros(1, dtype=torch.float32).to(device)
        self.init_loss = (init_loss or torch.zeros(1, dtype=torch.float32)).to(device)
        self.nbeta = torch.tensor(nbeta, dtype=torch.float32).to(device)
        self.device = device
        self.count = 0.0
        self.eval_field = eval_field

    def update(self, chain: int, draw: int, loss: torch.Tensor):
        self.loss += loss
        self.sq_loss += loss**2
        self.count += 1.0

    def finalize(self):
        scale = 1.0 / (self.count * xm.xrt_world_size())
        self.loss = xm.all_reduce(xm.REDUCE_SUM, self.loss, scale=scale)
        self.sq_loss = xm.all_reduce(xm.REDUCE_SUM, self.sq_loss, scale=scale)

        self.llc = self.nbeta * (self.loss - self.init_loss)

    def sample(self):
        """
        :returns: A dict :python:`{"llc/mean": llc_mean, "llc/std": llc_std, "llc-chain/{i}": llc_trace_per_chain, "loss/trace": loss_trace_per_chain}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [llc_estimator_instance], ...)`).
        """
        return {
            "init_loss": self.init_loss.cpu().item(),
            "llc": self.llc.cpu().item(),
            "loss": self.loss.cpu().item(),
            "sq_loss": self.sq_loss.cpu().item(),
        }

    def __call__(self, chain: int, draw: int, **kwargs):
        self.update(chain, draw, kwargs[self.eval_field])



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
    :param nbeta: nbeta, float (default: 1., set by sample() to utils.optimal_nbeta(dataloader)=len(batch_size)/np.log(len(batch_size)))
    :type nbeta: int
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'. Default is 'cpu'
    :type device: str | torch.device, optional
    """

    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        nbeta: float,
        init_loss,
        device="cpu",
        eval_field="loss",
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.init_loss = init_loss  # gets set by devinterp.slt.sample()

        self.losses = [[] for _ in range(num_chains)]
        self.llcs = np.zeros((num_chains, num_draws))
        self.nbeta = nbeta
        self.device = device
        self.eval_field = eval_field

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain].append(loss)

    def finalize(self):
        self.losses = torch.tensor(self.losses).cpu().numpy()
        cum_loss_avgs = np.cumsum(np.mean(self.losses, axis=0), axis=0) / np.arange(
            1, self.num_draws + 1
        )
        self.llcs = self.nbeta * (cum_loss_avgs - self.init_loss.cpu().numpy())

    def sample(self):
        """
        :returns: A dict :python:`{"llc/means": llc_means, "llc/stds": llc_stds, "llc/trace": llc_trace_per_chain, "loss/trace": loss_trace_per_chain}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [llc_estimator_instance], ...)`).
        """
        return {
            "init_loss": self.init_loss.item(),
            "llc": self.llcs[-1],
            "loss": self.losses[-1],
            "llcs": self.llcs,
            "losses": self.losses,
        }

    def __call__(self, chain: int, draw: int, **kwargs):
        self.update(chain, draw, kwargs[self.eval_field])


def llc_estimator_factory(
    num_chains: int,
    num_draws: int,
    nbeta: float,
    init_loss,
    device="cpu",
    eval_field: str = "loss",
    online: bool = False,
    per_token: bool = False,
) -> SamplerCallback:
    if online and per_token:
        raise ValueError("Cannot have both online and per_token set to True")
    elif online:
        return OnlineLLCEstimator(
            num_chains=num_chains,
            num_draws=num_draws,
            nbeta=nbeta,
            init_loss=init_loss,
            device=device,
            eval_field=eval_field,
        )
    elif per_token:
        return PerTokenLLCEstimator(
            num_chains=num_chains,
            num_draws=num_draws,
            nbeta=nbeta,
            init_loss=init_loss,
            device=device,
            eval_field=eval_field,
        )
    else:
        return LLCEstimator(
            num_chains=num_chains,
            num_draws=num_draws,
            nbeta=nbeta,
            device=device,
            init_loss=init_loss,
            eval_field=eval_field,
        )
