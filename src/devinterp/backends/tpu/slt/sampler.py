import contextlib
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from devinterp.slt.callback import SamplerCallback, validate_callbacks
from torch import nn
from torch.multiprocessing import cpu_count, get_context
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from devinterp.backends.tpu.slt.llc import LLCEstimator, OnlineLLCEstimator
from devinterp.backends.default.slt.mala import MalaAcceptanceRate
from devinterp.backends.default.slt.norms import NoiseNorm
from devinterp.optim.sgld import SGLD
from devinterp.slt.callback import SamplerCallback, validate_callbacks
from devinterp.utils import (
    USE_TPU_BACKEND,
    EvaluateFn,
    call_with,
    get_init_loss_multi_batch,
    optimal_nbeta,
    prepare_input,
    set_seed
)

def mark_step_if_xla():
    if USE_TPU_BACKEND:
        xm.mark_step()


def sample_single_chain(
    ref_model: nn.Module,
    loader: torch.utils.data.DataLoader,
    evaluate: Callable[[nn.Module, torch.Tensor], Tuple[torch.Tensor, Dict[str, Any]]],
    num_draws=100,
    num_burnin_steps=0,
    num_steps_bw_draws=1,
    grad_accum_steps: int = 1,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict] = None,
    scheduler_cls: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
    scheduler_kwargs: Optional[Dict] = None,
    chain: int = 0,
    seed: Optional[int] = None,
    verbose=True,
    device: Union[str, torch.device] = torch.device("xla"),
    callbacks: List[Callable] = [],
    batch_size: int = 32,
    optimize_over_per_model_param: Optional[Dict[str, torch.Tensor]] = None,
    init_noise: Optional[float] = None,
    shuffle=True,
    use_alternate_batching = False, # See George's alternate SGLD sampling method
    **kwargs,
):
    """
    Base function to sample a single chain. This function is called by the `sample` function on both single and multi-core setups.
    """


    # == Model ==
    model = deepcopy(ref_model).to(device)

    if seed is not None:
        set_seed(seed, device=device)

    # == Optimizer ==
    optimizer_kwargs = optimizer_kwargs or {}
    if any(isinstance(callback, MalaAcceptanceRate) for callback in callbacks):
        optimizer_kwargs.setdefault("save_mala_vars", True)

    param_groups = []

    for name, parameter in model.named_parameters():
        if optimize_over_per_model_param:
            if name in optimize_over_per_model_param:
                param_groups.append(
                    {
                        "params": parameter,
                        "optimize_over": optimize_over_per_model_param[name].to(device),
                    }
                )
        else:
            param_groups.append(parameter)

    optimizer = sampling_method(
        param_groups,
        **optimizer_kwargs,
    )

    # == (Optional) Init Noise ==
    if init_noise:
        for pg in optimizer.param_groups:
            params = pg["params"]
            for parameter in params:
                optimize_over = pg.get("optimize_over", 1.0) or 1.0
                noise_term = (
                    torch.randn_like(parameter.data) * init_noise * optimize_over
                )
                parameter.data.add_(noise_term)

    # == (Optional) Scheduler ==
    scheduler = None

    if scheduler_cls is not None:
        scheduler_kwargs = scheduler_kwargs or {}
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

    # == Sampling ==
    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps

    if use_alternate_batching:
        # We take one very large batch and sample SGLD on the fixed batch.
        cycler = cycle(loader)
        feed = []
        for step in range(grad_accum_steps):
            data = next(cycler)
            feed.append(data)
        feed = zip(range(num_steps * grad_accum_steps), cycle(feed))
    else:
        feed = zip(range(num_steps * grad_accum_steps), cycle(loader))

    model.train()
    no_grad = not any(map(lambda pg: pg["temperature"] > 0, optimizer.param_groups))

    mark_step_if_xla()

    # The nested loop structure is present to prevent XLA from recompiling the computation graph at each iteration,
    # which is what happens if we have an if statement that changes control flow at each iteration.

    # Will: If my modifications break on TPUs, let me know.
    with trange(0, num_steps,
                    desc = f"[{device}] Chain {chain}", 
                    disable = not verbose) as pbar:
        for i in pbar:
            # optimizer.zero_grad()
            loss, results = None, {}

            # Note: The effective batch size is grad_accum_steps * batch_size
            # To implement George's alternate SGLD sampling method, we set grad_accum_steps to, say, 100
            # and batch_size to 32 to sample from 1 effective batch of size 3.2k

            for j in range(grad_accum_steps):
                data = next(feed)[1]
                _loss, _results = evaluate(model, prepare_input(data, device))
                _mean_loss = _loss.mean() / grad_accum_steps

                if not no_grad:
                    _mean_loss.backward()

                if j == 0:
                    # First gradient accumulation iteration: create the loss object
                    loss = _loss.detach() / grad_accum_steps
                    for k, v in _results.items():
                        if torch.is_tensor(v):
                            results[k] = v.detach() / grad_accum_steps

                else:
                    # Later gradient accumulation iterations: accumulate the loss
                    loss += _loss.detach() / grad_accum_steps

                    for k, v in _results.items():
                        if torch.is_tensor(v):
                            results[k] += v.detach() / grad_accum_steps

                pbar.set_postfix({"grad_accum_steps": j})

                mark_step_if_xla()

            # Check loss is not nan or inf
            # if torch.isnan(loss).any() or torch.isinf(loss).any():
            #     raise ChainHealthError(
            #         f"Chain {chain} failed: Loss is NaN or Inf"
            #     )

            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            mark_step_if_xla()

            if (
                i >= num_burnin_steps
                and (i - num_burnin_steps) % num_steps_bw_draws == 0
            ):
                draw = (
                    i - num_burnin_steps
                ) // num_steps_bw_draws  # required for locals()

                loss = loss.detach()

                for k, v in results.items():
                    if torch.is_tensor(v):
                        results[k] = v.detach()

                with torch.no_grad():
                    for callback in callbacks:
                        callback(
                            loss=loss,
                            draw=draw,
                            chain=chain,
                            **results,
                        )

                mark_step_if_xla()


    # except ChainHealthError as e:
    #     warnings.warn(f"Chain failed: {e}")

    # if scheduler:
    #     del scheduler


def _sample_single_chain(kwargs):
    return sample_single_chain(**kwargs)


def _sample_single_chain_xla(kwargs):
    device = kwargs.pop("device")
    ordinal = xm.get_ordinal()
    num_chains = kwargs["num_chains"]
    seeds = kwargs.pop("seeds")

    if ordinal > num_chains:
        return

    for chain in range(ordinal, num_chains, xm.xrt_world_size()):
        seed = seeds[chain]

        for callback in kwargs["callbacks"]:
            if hasattr(callback, "to"):
                callback.to(device, chain=chain)

        sample_single_chain(**kwargs, chain=chain, seed=seed, device=device)


def sample(
    model: torch.nn.Module,
    loader: torch.utils.data.Dataset,
    callbacks: Union[List[SamplerCallback], Dict[str, SamplerCallback]],
    evaluate: Callable = lambda model, data: model(data),
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    scheduler_cls: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    grad_accum_steps: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: Union[torch.device, str] = torch.device("cpu"),
    verbose: bool = True,
    optimize_over_per_model_param: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: int = 32,
    init_noise: Optional[float] = None,
    shuffle: bool = True,
    use_alternate_batching = False, # See George's alternate SGLD sampling method
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

    callbacks_list = (
        callbacks if isinstance(callbacks, list) else list(callbacks.values())
    )

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

    validate_callbacks(callbacks_list)

    # shared_kwargs is passed into sample_single_chain for each chain. 
    # Args differ slightly based on chain index.
    shared_kwargs = dict(
        ref_model=model,
        evaluate=evaluate,
        num_draws=num_draws,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_cls=scheduler_cls,
        scheduler_kwargs=scheduler_kwargs,
        verbose=verbose,
        callbacks=callbacks_list,
        loader=loader,
        device=device,
        num_chains=num_chains,
        seeds=seeds,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        optimize_over_per_model_param=optimize_over_per_model_param,
        init_noise=init_noise,
        shuffle=shuffle,
        use_alternate_batching=use_alternate_batching,
    )

    def get_args(i):
        return dict(
            chain=i,
            seed=seeds[i],
            **shared_kwargs,
        )

    if "xla" in str(device):
        if num_chains % xm.xrt_world_size() != 0:
            raise ValueError(
                f"Number of chains ({num_chains}) must be divisible by number of TPU cores ({xm.xrt_world_size()})"
            )

        if seed is None:
            warnings.warn("No seed provided. Each process will be identical.")

        _sample_single_chain_xla(shared_kwargs)
        # xm.wait_device_ops()

    elif cores > 1:
        ctx = get_context("spawn")
        with ctx.Pool(cores) as pool:
            pool.map(_sample_single_chain, [get_args(i) for i in range(num_chains)])
    else:
        for i in range(num_chains):
            _sample_single_chain(get_args(i))

    for callback in callbacks_list:
        if hasattr(callback, "finalize"):
            callback.finalize()

    results = {}

    if isinstance(callbacks, dict):
        for name, callback in callbacks.items():
            if name == "":
                results.update(callback.get_results())
            else:
                results[name] = callback.get_results()
    else:
        for callback in callbacks:
            if hasattr(callback, "get_results"):
                results.update(callback.get_results())

    return results
