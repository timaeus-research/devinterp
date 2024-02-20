from typing import Callable, Dict, List, Literal, Optional, Type, Union

import torch
from torch.utils.data import DataLoader

from devinterp.optim.sgld import SGLD
from devinterp.slt.callback import SamplerCallback
from devinterp.slt.sampler import sample
from devinterp.utils import (
    optimal_temperature,
    get_init_loss_one_batch,
    get_init_loss_full_batch,
)


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

        self.temperature = torch.tensor(temperature, dtype=torch.float32).to(device)
        self.llc_per_chain = torch.zeros(num_chains, dtype=torch.float32).to(device)
        self.llc_mean = torch.tensor(0.0, dtype=torch.float32).to(device)
        self.llc_std = torch.tensor(0.0, dtype=torch.float32).to(device)

        self.device = device

    def update(self, chain: int, draw: int, loss: float, init_loss):
        self.losses[chain, draw] = loss
        self.init_loss = init_loss  # This is clunky, sorry

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

    def __call__(self, chain: int, draw: int, loss: float, init_loss: float):
        self.update(chain, draw, loss, init_loss)


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

        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(
            device
        )
        self.llcs = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.moving_avg_llcs = torch.zeros(
            (num_chains, num_draws), dtype=torch.float32
        ).to(device)

        self.temperature = torch.tensor(temperature, dtype=torch.float32).to(device)

        self.llc_means = torch.tensor(num_draws, dtype=torch.float32).to(device)
        self.llc_stds = torch.tensor(num_draws, dtype=torch.float32).to(device)

        self.device = device

    def update(self, chain: int, draw: int, loss: float, init_loss: float):
        self.init_loss = init_loss
        self.losses[chain, draw] = loss
        self.llcs[chain, draw] = self.temperature * (loss - self.init_loss)
        if draw == 0:
            self.moving_avg_llcs[chain, draw] = self.temperature * (loss - self.init_loss)
        else:
            t = draw + 1
            prev_llc = self.llcs[chain, draw - 1]
            with torch.no_grad():
                self.moving_avg_llcs[chain, draw] = (1 / t) * (
                    (t - 1) * prev_llc + self.temperature * (loss - self.init_loss)
                )

    def finalize(self):
        self.llc_means = self.llcs.mean(dim=0)
        self.llc_stds = self.llcs.std(dim=0)

    def sample(self):
        return {
            "llc/means": self.llc_means.cpu().numpy(),
            "llc/stds": self.llc_stds.cpu().numpy(),
            "llc/trace": self.llcs.cpu().numpy(),
            "loss/trace": self.losses.cpu().numpy(),
        }

    def __call__(self, chain: int, draw: int, loss: float, init_loss: float):
        self.update(chain, draw, loss, init_loss)


def estimate_learning_coeff_with_summary(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Callable,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict] = {},
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
    init_loss: float = None,
) -> dict:
    optimizer_kwargs.setdefault("temperature", optimal_temperature(loader.dataset))
    if not init_loss:
        init_loss = get_init_loss_one_batch(loader, model, criterion, device)
        # alternative: init_loss = get_init_loss_full_batch(loader, model, criterion, device)
    if online:
        llc_estimator = OnlineLLCEstimator(
            num_chains, num_draws, optimizer_kwargs["temperature"], device=device
        )
    else:
        llc_estimator = LLCEstimator(
            num_chains, num_draws, optimizer_kwargs["temperature"], device=device
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
        init_loss=init_loss,
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
    init_loss: float = None,
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
        init_loss=init_loss,
    )["llc/mean"]
