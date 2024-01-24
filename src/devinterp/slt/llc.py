from typing import Callable, Dict, List, Literal, Optional, Type, Union

import torch
from torch.utils.data import DataLoader

from devinterp.optim.sgld import SGLD
from devinterp.slt.callback import SamplerCallback
from devinterp.slt.sampler import sample


class LLCEstimator(SamplerCallback):
    r"""
    Callback for estimating the Local Learning Coefficient (LLC) in a rolling fashion during a sampling process.
    It calculates the LLC based on the average loss across draws for each chain:
    
    $$LLC = \frac{n}{\log(n)} * (\textrm{avg_loss} - \textrm{init_loss})$$

    For use with :func:`devinterp.slt.sampler.sample`.

    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_chains: int
    :param n: Number of samples $n$ used to calculate the LLC.
    :type n: int
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'.
    :type device: str | torch.device, optional
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

        self.n = torch.tensor(n, dtype=torch.float32).to(device)
        self.llc_per_chain = torch.zeros(num_chains, dtype=torch.float32).to(device)
        self.llc_mean = torch.tensor(0.0, dtype=torch.float32).to(device)
        self.llc_std = torch.tensor(0.0, dtype=torch.float32).to(device)

        self.device = device

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss

    @property
    def init_loss(self):
        return self.losses[0, 0]

    def finalize(self):
        avg_losses = self.losses.mean(axis=1)
        self.llc_per_chain = (self.n / self.n.log()) * (avg_losses - self.init_loss)
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

    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)


class OnlineLLCEstimator(SamplerCallback):
    """
    Callback for estimating the Local Learning Coefficient (LLC) in an online fashion during a sampling process.
    It calculates LLCs using the same formula as :func:`devinterp.slt.llc.LLCEstimator`, but continuously and including means and std across draws (as opposed to just across chains).
    For use with :func:`devinterp.slt.sampler.sample`.

    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_chains: int
    :param n: Number of samples $n$ used to calculate the LLC.
    :type n: int
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'. Default is 'cpu'
    :type device: str | torch.device, optional
    
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
        self.llcs = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)

        self.n = torch.tensor(n, dtype=torch.float32).to(device)

        self.llc_means = torch.tensor(num_draws, dtype=torch.float32).to(device)
        self.llc_stds = torch.tensor(num_draws, dtype=torch.float32).to(device)

        self.device = device

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss
        init_loss = self.losses[chain, 0]

        if (
            draw == 0
        ):  # TODO: We can probably drop this and it still works (but harder to read)
            self.llcs[chain, draw] = 0.0
        else:
            t = draw + 1
            prev_llc = self.llcs[chain, draw - 1]
            # print(chain, draw, prev_llc, self.n, loss, init_loss, loss - init_loss)

            with torch.no_grad():
                self.llcs[chain, draw] = (1 / t) * (
                    (t - 1) * prev_llc + (self.n / self.n.log()) * (loss - init_loss)
                )

    @property
    def init_loss(self):
        return self.losses[:, 0].mean()

    def finalize(self):
        self.llc_means = self.llcs.mean(dim=0)
        self.llc_stds = self.llcs.std(dim=0)

    def sample(self):
        """    
        :returns: A dict :python:`{"llc/means": llc_means, "llc/stds": llc_stds, "llc/trace": llc_trace_per_chain, "loss/trace": loss_trace_per_chain}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [llc_estimator_instance], ...)`).
        """
        return {
            "llc/means": self.llc_means.cpu().numpy(),
            "llc/stds": self.llc_stds.cpu().numpy(),
            "llc/trace": self.llcs.cpu().numpy(),
            "loss/trace": self.losses.cpu().numpy(),
        }

    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)


def estimate_learning_coeff_with_summary(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Callable,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    callbacks: List[Callable] = [],
    online: bool = False,
) -> dict:
    if online:
        llc_estimator = OnlineLLCEstimator(
            num_chains, num_draws, optimizer_kwargs["num_samples"], device=device
        )
    else:
        llc_estimator = LLCEstimator(
            num_chains, num_draws, optimizer_kwargs["num_samples"], device=device
        )

    callbacks = [llc_estimator, *callbacks]

    sample(
        model=model,
        loader=loader,
        criterion=criterion,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        cores=cores,
        seed=seed,
        device=device,
        verbose=verbose,
        callbacks=callbacks,
    )

    results = {}

    for callback in callbacks:
        if hasattr(callback, "sample"):
            results.update(callback.sample())

    return results


def estimate_learning_coeff(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Callable,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    callbacks: List[Callable] = [],
) -> float:
    return estimate_learning_coeff_with_summary(
        model=model,
        loader=loader,
        criterion=criterion,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        cores=cores,
        seed=seed,
        device=device,
        verbose=verbose,
        callbacks=callbacks,
        online=False,
    )["llc/mean"]
