import inspect
import itertools
from copy import deepcopy
from typing import Callable, Dict, List, Literal, Optional, Type, Union
import warnings

import torch
from torch import nn
from torch.multiprocessing import cpu_count, get_context
from torch.utils.data import DataLoader
from tqdm import tqdm

from devinterp.optim.sgld import SGLD
from devinterp.utils import (
    optimal_temperature,
    get_init_loss_one_batch,
    get_init_loss_multi_batch,
    get_init_loss_full_batch,
)
from devinterp.slt.callback import validate_callbacks, SamplerCallback
from devinterp.slt.mala import MalaAcceptanceRate
from devinterp.slt.norms import NoiseNorm
from devinterp.slt.llc import OnlineLLCEstimator, LLCEstimator


def call_with(func: Callable, **kwargs):
    # Check the func annotation and call with only the necessary kwargs.
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
    init_loss: float = None,
):
    if num_draws > len(loader):
        warnings.warn(
            "You are taking more sample batches than there are dataloader batches available, this removes some randomness from sampling but is probably fine. (All sample batches beyond the number dataloader batches are cycled from the start, f.e. 9 samples from [A, B, C] would be [B, A, C, B, A, C, B, A, C].)"
        )
    # Initialize new model and optimizer for this chain
    model = deepcopy(ref_model).to(device)

    optimizer_kwargs = optimizer_kwargs or {}
    if any(isinstance(callback, MalaAcceptanceRate) for callback in callbacks):
        optimizer_kwargs.setdefault("save_mala_vars", True)
    if any(isinstance(callback, NoiseNorm) for callback in callbacks):
        optimizer_kwargs.setdefault("save_noise", True)
    optimizer_kwargs.setdefault("temperature", optimal_temperature(loader))
    optimizer = sampling_method(model.parameters(), **optimizer_kwargs)

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps

    for i, (xs, ys) in tqdm(
        zip(range(num_steps), itertools.cycle(loader)),
        desc=f"Chain {chain}",
        total=num_steps,
        disable=not verbose,
    ):
        model.train()
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
    callbacks: List[SamplerCallback],
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    init_loss: float = None,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: Union[torch.device, str] = torch.device("cpu"),
    verbose: bool = True,
):
    """
    Sample model weights using a given sampling_method, supporting multiple chains/cores, 
    and calculate the observables (loss, llc, etc.) for each callback passed along. 
    The :python:`update`, :python:`finalize` and :python:`sample` methods of each :func:`~devinterp.slt.callback.SamplerCallback` are called 
    during sampling, after sampling, and at :python:`sampler_callback_object.sample()` respectively.
    
    After calling this function, the stats of interest live in the callback object.

    :param model: The neural network model.
    :type model: torch.nn.Module
    :param loader: DataLoader for input data.
    :type loader: DataLoader
    :param criterion: Loss function.
    :type criterion: Callable
    :param callbacks: list of callbacks, each of type SamplerCallback
    :type callbacks: list[SamplerCallback]
    :param sampling_method: Sampling method to use (a PyTorch optimizer under the hood). Default is SGLD
    :type sampling_method: torch.optim.Optimizer, optional
    :param optimizer_kwargs: Keyword arguments for the PyTorch optimizer (used as sampler here). Default is None (using standard SGLD parameters as defined in the SGLD class)
    :type optimizer_kwargs: dict, optional
    :param num_draws: Number of samples to draw. Default is 100
    :type num_draws: int, optional
    :param num_chains: Number of chains to run. Default is 10
    :type num_chains: int, optional
    :param num_burnin_steps: Number of burn-in steps before sampling. Default is 0
    :type num_burnin_steps: int, optional
    :param num_steps_bw_draws: Number of steps between each draw. Default is 1
    :type num_steps_bw_draws: int, optional
    :param init_loss: Initial loss for use in `LLCEstimator` and `OnlineLLCEstimator`
    :type init_loss: float, optional
    :param cores: Number of cores for parallel execution. Default is 1
    :type cores: int, optional
    :param seed: Random seed(s) for sampling. Each chain gets a different (deterministic) seed if this is passed. Default is None
    :type seed: int, optional
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'. Default is 'cpu'
    :type device: str or torch.device, optional
    :param verbose: whether to print sample chain progress. Default is True
    :type verbose: bool, optional
    
    :raises ValueError: if derivative callbacks (f.e. :func:`~devinterp.slt.loss.OnlineLossStatistics`) are passed before base callbacks (f.e. :func:`~devinterp.slt.llc.OnlineLLCEstimator`)
    :raises Warning: if num_burnin_steps < num_draws
    :raises Warning: if num_draws > len(loader)
    :raises Warning: if using seeded runs
    
    :returns: None (access LLCs or other observables through `callback_object.sample()`)
    """
    if num_burnin_steps < num_draws:
        warnings.warn(
            "You are taking more draws than burn-in steps, your LLC estimates will likely be underestimates. Please check LLC chain convergence."
        )
    if num_draws > len(loader):
        warnings.warn(
            "You are taking more sample batches than there are dataloader batches available, "
            "this removes some randomness from sampling but is probably fine. (All sample batches "
            "beyond the number dataloader batches are cycled from the start, f.e. 9 samples from [A, B, C] would be [B, A, C, B, A, C, B, A, C].)"
        )
    if not init_loss:
        init_loss = get_init_loss_multi_batch(
            loader, num_chains, model, criterion, device
        )
        # alternative: init_loss = get_init_loss_full_batch(loader, model, criterion, device)
        # alternative: init_loss = get_init_loss_one_batch(loader, model, criterion, device)
    for callback in callbacks:
        if isinstance(callback, (OnlineLLCEstimator, LLCEstimator)):
            setattr(callback, "init_loss", init_loss)

    if cores is None:
        cores = min(4, cpu_count())

    if seed is not None:
        warnings.warn(
            "You are using seeded runs, for full reproducibility check https://pytorch.org/docs/stable/notes/randomness.html"
        )
        if isinstance(seed, int):
            seeds = [seed + i for i in range(num_chains)]
        elif len(seed) != num_chains:
            raise ValueError("Length of seed list must match number of chains")
        else:
            seeds = seed
    else:
        seeds = [None] * num_chains

    validate_callbacks(callbacks)

    def get_args(i):
        return dict(
            chain=i,
            seed=seeds[i],
            ref_model=model,
            loader=loader,
            criterion=criterion,
            num_draws=num_draws,
            num_burnin_steps=num_burnin_steps,
            num_steps_bw_draws=num_steps_bw_draws,
            init_loss=init_loss,
            sampling_method=sampling_method,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
            verbose=verbose,
            callbacks=callbacks,
        )

    if cores > 1:
        ctx = get_context("spawn")
        with ctx.Pool(cores) as pool:
            pool.map(_sample_single_chain, [get_args(i) for i in range(num_chains)])
    else:
        for i in range(num_chains):
            _sample_single_chain(get_args(i))

    for callback in callbacks:
        if hasattr(callback, "finalize"):
            callback.finalize()

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
    optimizer_kwargs.setdefault("temperature", optimal_temperature(loader))
    if not init_loss:
        init_loss = get_init_loss_multi_batch(
            loader, num_chains, model, criterion, device
        )
        # alternative: init_loss = get_init_loss_full_batch(loader, model, criterion, device)
        # alternative: init_loss = get_init_loss_one_batch(loader, model, criterion, device)
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
