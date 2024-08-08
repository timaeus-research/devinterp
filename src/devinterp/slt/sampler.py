import inspect
import itertools
import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Literal, Optional, Type, Union

import torch
from torch import nn
from torch.multiprocessing import cpu_count, get_context
from torch.utils.data import DataLoader
from tqdm import tqdm

from devinterp.optim.sgld import SGLD
from devinterp.slt.callback import SamplerCallback, validate_callbacks
from devinterp.slt.llc import LLCEstimator, OnlineLLCEstimator
from devinterp.slt.mala import MalaAcceptanceRate
from devinterp.slt.norms import NoiseNorm
from devinterp.utils import (
    EvaluateFn,
    get_init_loss_multi_batch,
    optimal_temperature,
    prepare_input,
    split_results,
)



def sample_single_chain(
    ref_model: nn.Module,
    loader: DataLoader,
    evaluate: EvaluateFn,
    num_draws=100,
    num_burnin_steps=0,
    num_steps_bw_draws=1,
    grad_accum_steps=1,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict] = None,
    chain: int = 0,
    seed: Optional[int] = None,
    verbose: bool = True,
    device: torch.device = torch.device("cpu"),
    optimize_over_per_model_param: Optional[dict] = None,
    callbacks: List[SamplerCallback] = [],
    init_loss: float = None,
    
):
    if grad_accum_steps > 1:
        assert type(grad_accum_steps) == int, "grad_accum_steps must be an integer."
        num_steps_bw_draws *= grad_accum_steps
        num_burnin_steps *= grad_accum_steps
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
    if optimize_over_per_model_param:
        param_groups = []
        for name, parameter in model.named_parameters():
            param_groups.append(
                {
                    "params": parameter,
                    "optimize_over": optimize_over_per_model_param[name].to(device),
                }
            )
        optimizer = sampling_method(
            param_groups,
            **optimizer_kwargs,
        )
    else:
        optimizer = sampling_method(model.parameters(), **optimizer_kwargs)


    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps

    cumulative_loss = 0
    with tqdm(desc=f"Chain {chain}",
        total=num_steps // grad_accum_steps,
        disable=not verbose) as pbar:
        for i, data in zip(range(num_steps), itertools.cycle(loader)):
            model.train()
            data = prepare_input(data, device)

            results = evaluate(model, data)
            loss, results = split_results(results)

            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
                cumulative_loss += loss.item()
            loss.backward()

            # i+1 instead of i so that the gradient accumulates to an entire batch first
            # otherwise the first draw happens after batch_size/grad_accum_steps samples instead of batch_size samples
            if (i+1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

            if i >= num_burnin_steps and (i + 1 - num_burnin_steps) % num_steps_bw_draws == 0:
                draw = (i - num_burnin_steps) // num_steps_bw_draws  # required for locals()
                if grad_accum_steps > 1:
                    loss = cumulative_loss
                    cumulative_loss = 0
                else:
                    loss = loss.item()

                with torch.no_grad():
                    for callback in callbacks:
                        callback(**locals())  # Cursed. This is the way.


def _sample_single_chain(kwargs):
    return sample_single_chain(**kwargs)


def sample(
    model: torch.nn.Module,
    loader: DataLoader,
    callbacks: List[SamplerCallback],
    evaluate: Optional[EvaluateFn] = None,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    init_loss: float = None,
    grad_accum_steps: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: Union[torch.device, str] = torch.device("cpu"),
    verbose: bool = True,
    optimize_over_per_model_param: Optional[Dict[str, List[bool]]] = None,
):
    """
    Sample model weights using a given sampling_method, supporting multiple chains/cores,
    and calculate the observables (loss, llc, etc.) for each callback passed along.
    The :python:`update`, :python:`finalize` and :python:`sample` methods of each :func:`~devinterp.slt.callback.SamplerCallback` are called
    during sampling, after sampling, and at :python:`sampler_callback_object.get_results()` respectively.

    After calling this function, the stats of interest live in the callback object.

    :param model: The neural network model.
    :type model: torch.nn.Module
    :param loader: DataLoader for input data.
    :type loader: DataLoader
    :param evaluate: Maps a model and batch of data to an object with a loss attribute.
    :type evaluate: EvaluateFn
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

    :returns: None (access LLCs or other observables through `callback_object.get_results()`)
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
            loader, num_chains, model, evaluate, device
        )
        # alternative: init_loss = get_init_loss_full_batch(loader, model, evaluate, device)
        # alternative: init_loss = get_init_loss_one_batch(loader, model, evaluate, device)
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

    if evaluate is None:

        def evaluate(model, data):
            return model(data)

    validate_callbacks(callbacks)

    def get_args(i):
        return dict(
            chain=i,
            seed=seeds[i],
            ref_model=model,
            loader=loader,
            evaluate=evaluate,
            num_draws=num_draws,
            num_burnin_steps=num_burnin_steps,
            num_steps_bw_draws=num_steps_bw_draws,
            init_loss=init_loss,
            grad_accum_steps=grad_accum_steps,
            sampling_method=sampling_method,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
            verbose=verbose,
            callbacks=callbacks,
            optimize_over_per_model_param=optimize_over_per_model_param,
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
    callbacks: List[Callable] = [],
    evaluate: Optional[EvaluateFn] = None,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    init_loss: float = None,
    grad_accum_steps: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: Union[torch.device, str] = torch.device("cpu"),
    verbose: bool = True,
    optimize_over_per_model_param: Optional[Dict[str, List[bool]]] = None,
    online: bool = False,
) -> dict:
    optimizer_kwargs.setdefault("temperature", optimal_temperature(loader))
    if not init_loss:
        init_loss = get_init_loss_multi_batch(
            loader, num_chains, model, evaluate, device
        )
        # alternative: init_loss = get_init_loss_full_batch(loader, model, evaluate, device)
        # alternative: init_loss = get_init_loss_one_batch(loader, model, evaluate, device)
    if online:
        llc_estimator = OnlineLLCEstimator(
            num_chains, num_draws, nbeta=optimizer_kwargs["temperature"], device=device, init_loss=init_loss
        )
    else:
        llc_estimator = LLCEstimator(
            num_chains, num_draws, nbeta=optimizer_kwargs["temperature"], device=device, init_loss=init_loss
        )

    callbacks = [llc_estimator, *callbacks]

    sample(
        model=model,
        loader=loader,
        evaluate=evaluate,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        grad_accum_steps=grad_accum_steps,
        cores=cores,
        seed=seed,
        device=device,
        verbose=verbose,
        callbacks=callbacks,
        init_loss=init_loss,
        optimize_over_per_model_param=optimize_over_per_model_param,
    )

    results = {}

    for callback in callbacks:
        if hasattr(callback, "get_results"):
            results.update(callback.get_results())

    return results


def estimate_learning_coeff(
    model: torch.nn.Module,
    loader: DataLoader,
    callbacks: List[Callable] = [],
    evaluate: Optional[EvaluateFn] = None,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    init_loss: float = None,
    grad_accum_steps: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: Union[torch.device, str] = torch.device("cpu"),
    verbose: bool = True,
    optimize_over_per_model_param: Optional[Dict[str, List[bool]]] = None
) -> float:
    return estimate_learning_coeff_with_summary(
        model=model,
        loader=loader,
        evaluate=evaluate,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        grad_accum_steps=grad_accum_steps,
        cores=cores,
        seed=seed,
        device=device,
        verbose=verbose,
        callbacks=callbacks,
        online=False,
        init_loss=init_loss,
        optimize_over_per_model_param=optimize_over_per_model_param,
    )["llc/mean"]
