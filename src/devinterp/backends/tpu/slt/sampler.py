import itertools
import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Literal, Optional, Type, Union

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch import nn
from torch.multiprocessing import cpu_count, get_context
from torch.utils.data import DataLoader
from tqdm import tqdm

from devinterp.backends.default.slt.llc import LLCEstimator, OnlineLLCEstimator
from devinterp.backends.default.slt.mala import MalaAcceptanceRate
from devinterp.backends.default.slt.norms import NoiseNorm
from devinterp.optim.sgld import SGLD
from devinterp.slt.callback import SamplerCallback, validate_callbacks
from devinterp.utils import (
    EvaluateFn,
    call_with,
    get_init_loss_multi_batch,
    optimal_temperature,
    prepare_input,
)


def sample_single_chain(
    ref_model: nn.Module,
    loader: DataLoader,
    evaluate: EvaluateFn,
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
    optimizer_kwargs.setdefault("temperature", optimal_temperature(loader))
    optimizer = sampling_method(model.parameters(), **optimizer_kwargs)

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps

    pbar = tqdm(
        zip(range(num_steps), itertools.cycle(loader)),
        desc=f"[{device}] Chain {chain}",
        total=num_steps,
        disable=not verbose,
    )

    model.train()

    for i, data in pbar:
        optimizer.zero_grad()

        results = evaluate(model, data)

        if isinstance(results, dict):
            loss = results.pop("loss")
        elif isinstance(results, tuple):
            loss = results[0]
            if len(results) > 1:
                results = loss[1:]
        elif isinstance(results, torch.Tensor):
            loss = results
            results = None
        elif hasattr(results, "loss"):
            loss = results.loss
        else:
            raise ValueError("compute_loss must return a dict, tuple, or torch.Tensor")

        mean_loss = loss.mean()
        mean_loss.backward()

        optimizer.step()
        xm.mark_step()

        if i >= num_burnin_steps and (i - num_burnin_steps) % num_steps_bw_draws == 0:
            draw = (i - num_burnin_steps) // num_steps_bw_draws  # required for locals()

            loss = loss.detach()

            for k, v in results.items():
                if torch.is_tensor(v):
                    results[k] = v.detach()

            with torch.no_grad():
                for callback in callbacks:
                    call_with(callback, **locals())  # Cursed. This is the way.

            xm.mark_step()


def _sample_single_chain(kwargs):
    device = xm.xla_device()
    ordinal = xm.get_ordinal()
    num_chains = kwargs["num_chains"]
    seeds = kwargs["seeds"]

    # TODO: Make sure the loaders have different batches

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
        evaluate = lambda model, data: model(data)

    validate_callbacks(callbacks)

    shared_kwargs = dict(
        ref_model=model,
        evaluate=evaluate,
        num_draws=num_draws,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        verbose=verbose,
        callbacks=callbacks,
        loader=loader,
        device=device,
        num_chains=num_chains,
        seeds=seeds,
    )

    if cores != xm.xrt_world_size():
        w = f"Number of cores ({cores}) does not match number of TPU cores ({xm.xrt_world_size()}), overwriting to {xm.xrt_world_size()}"
        cores = xm.xrt_world_size()
        warnings.warn(w)

    if num_chains % xm.xrt_world_size() != 0:
        raise ValueError(
            f"Number of chains ({num_chains}) must be divisible by number of TPU cores ({xm.xrt_world_size()})"
        )

    if seed is None:
        warnings.warn("No seed provided. Each process will be identical.")

    xmp.spawn(
        _sample_single_chain, args=(shared_kwargs,), nprocs=cores, start_method="fork"
    )

    for callback in callbacks:
        if hasattr(callback, "finalize"):
            callback.finalize()
