from typing import Callable, Dict, List, Literal, Optional, Type, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from devinterp.optim.sgld import SGLD
from devinterp.slt.callback import SamplerCallback
from devinterp.slt.sampler import sample


class LLCEstimator(SamplerCallback):
    def __init__(self, num_chains: int, num_draws: int, n: int, device="cpu"):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)

        self.n = torch.tensor(n, dtype=torch.float32).to(device)
        self.llc_per_chain = torch.zeros(num_chains, dtype=torch.float32).to(device)
        self.llc_mean = torch.tensor(0., dtype=torch.float32).to(device)
        self.llc_std = torch.tensor(0., dtype=torch.float32).to(device)

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
        return {
            "llc/mean": self.llc_mean.cpu().numpy().item(),
            "llc/std": self.llc_std.cpu().numpy().item(),
            **{f"llc-chain/{i}": self.llc_per_chain[i].cpu().numpy().item() for i in range(self.num_chains)},
            "loss/trace": self.losses.cpu().numpy(),
        }
    
    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)


class OnlineLLCEstimator(SamplerCallback):
    def __init__(self, num_chains: int, num_draws: int, n: int, device="cpu"):
        self.num_chains = num_chains
        self.num_draws = num_draws

        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.llcs = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)

        self.n = torch.tensor(n, dtype=torch.float32).to(device)
        self.llc_means = torch.tensor(num_draws, dtype=torch.float32).to(device)
        self.llc_stds = torch.tensor(num_draws, dtype=torch.float32).to(device)

        self.device = device

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss 

        if draw == 0:  # TODO: We can probably drop this and it still works (but harder to read)
            self.llcs[0, draw] = 0.
        else:
            t = draw + 1
            prev_llc = self.llcs[chain, draw - 1]

            with torch.no_grad():
                self.llcs[chain, draw] = (1 / t) * (
                    (t - 1) * prev_llc + (self.n / self.n.log()) * (loss - self.init_loss)
                )

    @property
    def init_loss(self):
        return self.losses[:, 0].mean()

    def finalize(self):
        self.llc_means = self.llcs.mean(axis=0)
        self.llc_stds = self.llcs.std(axis=0)

    def sample(self):
        return {
            "llc/means": self.llc_means.cpu().numpy(),
            "llc/stds": self.llc_stds.cpu().numpy(),
            "llc/trace": self.llcs.cpu().numpy(),
            "loss/trace": self.losses.cpu().numpy()
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
        llc_estimator = OnlineLLCEstimator(num_chains, num_draws, optimizer_kwargs['num_samples'], device=device)
    else:
        llc_estimator = LLCEstimator(num_chains, num_draws, optimizer_kwargs['num_samples'], device=device)

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
    online: bool = False,
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
        online=online,
    )["llc/mean"]