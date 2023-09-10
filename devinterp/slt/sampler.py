import itertools
import multiprocessing
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from logging import Logger
from multiprocessing import Pool, cpu_count
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from devinterp.optim.sgld import SGLD
from devinterp.slt.observables import MicroscopicObservable
from devinterp.utils import Criterion


def sample_single_chain(
    ref_model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    num_draws=100,
    num_burnin_steps=0,
    num_steps_bw_draws=1,
    step: Literal["sgld"] = "sgld",
    optimizer_kwargs: Optional[Dict] = None,
    chain: int = 0,
    seed: Optional[int] = None,
    pbar: bool = False,
    observor: Optional[MicroscopicObservable] = None,
    device: torch.device = torch.device("cpu"),
):
    # Initialize new model and optimizer for this chain
    model = deepcopy(ref_model).to(device)

    if step == "sgld":
        optimizer_kwargs = optimizer_kwargs or {}
        optimizer = SGLD(
            model.parameters(), **optimizer_kwargs
        )  # Replace with your actual optimizer kwargs

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps
    local_draws = pd.DataFrame(index=range(num_draws), columns=["chain", "loss"])

    iterator = zip(range(num_steps), itertools.cycle(loader))

    if pbar:
        iterator = tqdm(iterator, desc=f"Chain {chain}", total=num_steps)

    model.train()

    for i, (xs, ys) in iterator:
        xs, ys = xs.to(device), ys.to(device)
        y_preds = model(xs)
        print(y_preds.requires_grad, ys.requires_grad, xs.requires_grad)
        loss = criterion(y_preds, ys)
        print(loss.requires_grad, loss.grad_fn)
        loss.backward()

        if i >= num_burnin_steps and (i - num_burnin_steps) % num_steps_bw_draws == 0:
            draw_idx = (i - num_burnin_steps) // num_steps_bw_draws
            local_draws.loc[draw_idx, "chain"] = chain
            local_draws.loc[draw_idx, "loss"] = loss.detach().item()

        optimizer.step()
        optimizer.zero_grad()

    return local_draws


def sample(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    step: Literal["sgld"],
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: Optional[int] = 1,
    seed: Optional[Union[int, List[int]]] = None,
    pbar: bool = True,
    device: torch.device = torch.device("cpu"),
):
    """
    Sample model weights using a given optimizer, supporting multiple chains.

    Parameters:
        model (torch.nn.Module): The neural network model.
        step (Literal['sgld']): The name of the optimizer to use to step.
        loader (DataLoader): DataLoader for input data.
        criterion (torch.nn.Module): Loss function.
        num_draws (int): Number of samples to draw.
        num_chains (int): Number of chains to run.
        num_burnin_steps (int): Number of burn-in steps before sampling.
        num_steps_bw_draws (int): Number of steps between each draw.
        cores (Optional[int]): Number of cores for parallel execution.
        seed (Optional[Union[int, List[int]]]): Random seed(s) for sampling.
        progressbar (bool): Whether to display a progress bar.
        optimizer_kwargs (Optional[Dict[str, Union[float, Literal['adaptive']]]]): Keyword arguments for the optimizer.
    """
    if cores is None:
        cores = min(4, cpu_count())

    if seed is not None:
        if isinstance(seed, int):
            seeds = [seed + i for i in range(num_chains)]
        elif len(seed) != num_chains:
            raise ValueError("Length of seed list must match number of chains")
        else:
            seeds = seed
    else:
        seeds = [None] * num_chains

    def _sample_single_chain(chain_seed):
        return sample_single_chain(
            chain=chain_seed[0],
            seed=chain_seed[1],
            ref_model=model,
            loader=loader,
            criterion=criterion,
            num_draws=num_draws,
            num_burnin_steps=num_burnin_steps,
            num_steps_bw_draws=num_steps_bw_draws,
            step=step,
            optimizer_kwargs=optimizer_kwargs,
            pbar=pbar,
            device=device,
        )

    # with Pool(cores) as pool:
    #    results = pool.map(func, list(zip(range(num_chains), seeds)))

    results = []

    for chain, seed in zip(range(num_chains), seeds):
        results.append(_sample_single_chain((chain, seed)))

    results_df = pd.concat(results, ignore_index=True)
    return results_df


def estimate_rlct(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Criterion,
    step: Literal["sgld"],
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: Optional[int] = 1,
    seed: Optional[Union[int, List[int]]] = None,
    pbar: bool = True,
    baseline: Literal["init", "min"] = "init",
    device: torch.device = torch.device("cpu"),
) -> float:
    trace = sample(
        model=model,
        loader=loader,
        criterion=criterion,
        step=step,
        optimizer_kwargs=optimizer_kwargs,
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        cores=cores,
        seed=seed,
        pbar=pbar,
        device=device,
    )

    if baseline == "init":
        baseline_loss = trace.loc[trace["chain"] == 0, "loss"].iloc[0]
    elif baseline == "min":
        baseline_loss = trace["loss"].min()

    avg_loss = trace.groupby("chain")["loss"].mean().mean()
    num_samples = len(loader.dataset)

    return (avg_loss - baseline_loss) * num_samples / np.log(num_samples)
