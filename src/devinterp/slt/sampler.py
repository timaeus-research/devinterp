import inspect
import itertools
from copy import deepcopy
from typing import Callable, Dict, List, Literal, Optional, Type, Union
import warnings

import torch
from torch import nn
from torch.multiprocessing import cpu_count, get_context
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from devinterp.optim.sgld import SGLD
from devinterp.slt.callback import validate_callbacks, SamplerCallback


def call_with(func: Callable, **kwargs):
    """Check the func annotation and call with only the necessary kwargs."""
    sig = inspect.signature(func)

    # Filter out the kwargs that are not in the function's signature
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    # Call the function with the filtered kwargs
    return func(**filtered_kwargs)


def sample_single_chain(
    ref_model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    num_draws=100,
    num_burnin_steps=0,
    num_steps_bw_draws=1,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict] = None,
    chain: int = 0,
    seed: Optional[int] = None,
    verbose=True,
    device: torch.device = torch.device("cpu"),
    callbacks: List[SamplerCallback] = [],
    tqdm_kwargs: dict = {},
):
    # Initialize new model and optimizer for this chain
    model = deepcopy(ref_model).to(device, print_details=False)

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = sampling_method(model.parameters(), **optimizer_kwargs)

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps
    model.train()

    for i, (xs, ys) in tqdm(zip(range(num_steps), itertools.cycle(loader)), desc=f"Chain {chain}", total=num_steps, disable=not verbose, **tqdm_kwargs):
        optimizer.zero_grad()
        xs, ys = xs.to(device), ys.to(device)
        y_preds = model(xs)
        loss = criterion(y_preds, ys)

        loss.backward()
        optimizer.step()

        if i >= num_burnin_steps and (i - num_burnin_steps) % num_steps_bw_draws == 0:
            draw = (i - num_burnin_steps) // num_steps_bw_draws  # required for locals()
            loss = loss.item()

            with torch.no_grad():
                for callback in callbacks:
                    call_with(callback, **locals())  # Cursed. This is the way.


def _sample_single_chain(kwargs):
    return sample_single_chain(**kwargs)


def sample(
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
    device: Union[torch.device, str] = torch.device("cpu"),
    verbose: bool = True,
    callbacks: List[SamplerCallback] = [],
    tqdm_kwargs: dict = {},
):
    """
    Sample model weights using a given sampling_method, supporting multiple chains.
    See the example notebooks examples/diagnostics.ipynb and examples/sgld_calibration.ipynb for (respectively)
    info on what callbacks to pass along and how to calibrate sampler/optimizer hyperparams.

    Parameters:
        model (torch.nn.Module): The neural network model.
        loader (DataLoader): DataLoader for input data.
        criterion (torch.nn.Module): Loss function.
        sampling_method (torch.optim.Optimizer): Sampling method to use (really a PyTorch optimizer).
        optimizer_kwargs (Optional[Dict[str, Union[float, Literal['adaptive']]]]): Keyword arguments for the PyTorch optimizer (used as sampler here).
        num_draws (int): Number of samples to draw.
        num_chains (int): Number of chains to run.
        num_burnin_steps (int): Number of burn-in steps before sampling.
        num_steps_bw_draws (int): Number of steps between each draw.
        cores (Optional[int]): Number of cores for parallel execution.
        seed (Optional[Union[int, List[int]]]): Random seed(s) for sampling. Each chain gets a different (deterministic) seed if this is passed.
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.
        verbose (bool): whether to print sample chain progress
        callbacks (List[SamplerCallback]): list of callbacks, each of type SamplerCallback
    """
    if num_burnin_steps:
        warnings.warn('Burn-in is currently not implemented correctly, please set num_burnin_steps to 0.')
    if num_draws > len(loader):
        warnings.warn('You are taking more sample batches than there are dataloader batches available, this removes some randomness from sampling but is probably fine. (All sample batches beyond the number dataloader batches are cycled from the start, f.e. 9 samples from [A, B, C] would be [B, A, C, B, A, C, B, A, C].)')
      
    if cores is None:
        cores = min(4, cpu_count())

    if seed is not None:
        warnings.warn("You are using seeded runs, for full reproducibility check https://pytorch.org/docs/stable/notes/randomness.html")
        if isinstance(seed, int):
            seeds = [seed + i for i in range(num_chains)]
        elif len(seed) != num_chains:
            raise ValueError("Length of seed list must match number of chains")
        else:
            seeds = seed
    else:
        seeds = [None] * num_chains

    validate_callbacks(callbacks)

    tqdm_kwargs = {**tqdm_kwargs, "position": tqdm_kwargs.get("position", 0)}
    def get_args(i, *, is_parallel: bool):
        args = dict(
            chain=i,
            seed=seeds[i],
            ref_model=model,
            loader=loader,
            criterion=criterion,
            num_draws=num_draws,
            num_burnin_steps=num_burnin_steps,
            num_steps_bw_draws=num_steps_bw_draws,
            sampling_method=sampling_method,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
            verbose=verbose,
            callbacks=callbacks,
        )
        if is_parallel:
            # every progress bar should have a separate position since they update simultaneously
            args["tqdm_kwargs"] = {**tqdm_kwargs,
                "position": tqdm_kwargs.get("position", 0)+1+i,
                "leave": True, # this might help with async updates
            }
        else:
            # only one position, but we need to leave the bar in place on the last iteration
            args["tqdm_kwargs"] = {**tqdm_kwargs,
                "position": tqdm_kwargs.get("position", 0)+1,
                "leave": tqdm_kwargs.get("leave", True) and (i == num_chains - 1),
            }
        return args

    if cores > 1:
        ctx = get_context("spawn")
        with ctx.Pool(cores) as pool:
            pool.map(_sample_single_chain, [get_args(i, is_parallel=True) for i in range(num_chains)])
    else:
        for i in tqdm(range(num_chains), desc="Chain", **tqdm_kwargs):
            _sample_single_chain(get_args(i, is_parallel=False))

    for callback in callbacks:
        if hasattr(callback, "finalize"):
            callback.finalize()
