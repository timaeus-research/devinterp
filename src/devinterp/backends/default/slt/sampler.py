import warnings
from copy import deepcopy
from typing import Dict, List, Literal, Optional, Type, Union

import cloudpickle
import numpy as np
import torch
from devinterp.optim.sgld import SGLD
from devinterp.slt.callback import SamplerCallback, validate_callbacks
from devinterp.slt.llc import LLCEstimator, OnlineLLCEstimator
from devinterp.slt.mala import MalaAcceptanceRate
from devinterp.slt.norms import NoiseNorm
from devinterp.utils import (
    EvaluateFn,
    cycle,
    default_nbeta,
    get_init_loss_multi_batch,
    prepare_input,
    split_results,
)
from torch import nn
from torch.multiprocessing import cpu_count, get_context
from torch.utils.data import DataLoader
from tqdm import tqdm


def sample_single_chain(
    ref_model: nn.Module,
    loader: DataLoader,
    evaluate: EvaluateFn,
    optimizer_kwargs: Dict,
    num_draws=100,
    num_burnin_steps=0,
    num_steps_bw_draws=1,
    grad_accum_steps=1,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    chain: int = 0,
    seed: Optional[int] = None,
    verbose: bool = True,
    device: torch.device = torch.device("cpu"),
    optimize_over_per_model_param: Optional[dict] = None,
    callbacks: List[SamplerCallback] = [],
    use_amp: bool = False,
    **kwargs,
):
    if grad_accum_steps > 1:
        assert isinstance(grad_accum_steps, int), "grad_accum_steps must be an integer."
        num_steps_bw_draws *= grad_accum_steps
        num_burnin_steps *= grad_accum_steps
    if num_draws > len(loader):
        warnings.warn(
            "You are taking more sample batches than there are dataloader batches available, this removes some randomness from sampling but is probably fine. (All sample batches beyond the number dataloader batches are cycled from the start, f.e. 9 samples from [A, B, C] would be [B, A, C, B, A, C, B, A, C].)"
        )

    # Initialize new model and optimizer for this chain
    model = deepcopy(ref_model).to(device)
    if "temperature" in optimizer_kwargs:
        assert (
            not "nbeta" in optimizer_kwargs
        ), "Set either nbeta or temperature in optimizer_kwargs, not both"
        optimizer_kwargs["nbeta"] = optimizer_kwargs.pop("temperature")
    assert "nbeta" in optimizer_kwargs, "Set nbeta in optimizer_kwargs"
    if any(isinstance(callback, MalaAcceptanceRate) for callback in callbacks):
        optimizer_kwargs.setdefault("save_mala_vars", True)
    if any(isinstance(callback, NoiseNorm) for callback in callbacks):
        optimizer_kwargs.setdefault("save_noise", True)
    optimizer_kwargs.setdefault("nbeta", default_nbeta(loader))
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
    with tqdm(
        desc=f"Chain {chain}", total=num_steps // grad_accum_steps, disable=not verbose
    ) as pbar:
        model.train()
        for i, data in zip(range(num_steps), cycle(loader)):
            model.train()
            data = prepare_input(data, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                results = evaluate(model, data)
                loss, results = split_results(results)

                loss /= grad_accum_steps
                cumulative_loss += loss
                loss.backward()

            # i+1 instead of i so that the gradient accumulates to an entire batch first
            # otherwise the first draw happens after batch_size/grad_accum_steps samples instead of batch_size samples
            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()

            if (
                i >= num_burnin_steps
                and (i + 1 - num_burnin_steps) % num_steps_bw_draws == 0
            ):
                draw = (
                    i - num_burnin_steps
                ) // num_steps_bw_draws  # required for locals()
                loss = cumulative_loss

                with torch.no_grad():
                    for callback in callbacks:
                        callback(**locals())  # Cursed. This is the way

            if (i + 1) % grad_accum_steps == 0:
                optimizer.zero_grad()
                cumulative_loss = 0
                pbar.update(1)


def _sample_single_chain(kwargs):
    pickled_args = ["evaluate", "loader"]
    evaluate = cloudpickle.loads(kwargs["evaluate"])
    loader = cloudpickle.loads(kwargs["loader"])
    kwargs = {k: v for k, v in kwargs.items() if k not in pickled_args}
    return sample_single_chain(**kwargs, evaluate=evaluate, loader=loader)


def get_args(chain_idx: int, seeds: List[int], device, callbacks, shared_kwargs):
    if isinstance(device, list):
        instance_device = device[chain_idx % len(device)]
    else:
        instance_device = device
    return dict(
        chain=chain_idx,
        seed=seeds[chain_idx],
        device=instance_device,
        callbacks=callbacks,
        **shared_kwargs,
    )


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
    cores: Union[int, List[Union[str, torch.device]]] = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: Union[torch.device, str] = torch.device("cpu"),
    verbose: bool = True,
    optimize_over_per_model_param: Optional[Dict[str, List[bool]]] = None,
    gpu_idxs: Optional[List[int]] = None,
    batch_size: bool = 1,
    use_amp: bool = False,
    **kwargs,
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
    :param cores: Number or list of cores for parallel execution. Default is 1
    :type cores: int, optional
    :param seed: Random seed(s) for sampling. Each chain gets a different (deterministic) seed if this is passed. Default is None
    :type seed: int, optional
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'. Default is 'cpu'
    :type device: str or torch.device, optional
    :param verbose: whether to print sample chain progress. Default is True
    :type verbose: bool, optional
    :param optimize_over_per_model_param: Dictionary of booleans indicating whether to optimize over each parameter of the model. \
    Keys are parameter names, and values are boolean tensors that match the shape of the parameter. \
    A value of True (or 1) indicates that this particular element of the parameter should be optimized over. \
    None by default, which means that we optimize over all parameters.
    :type optimize_over_per_model_param: dict, optional
    :param use_amp: Whether to use automatic mixed precision. Casts to float16 on GPUs.
    :type use_amp: bool, optional
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
        if seed is not None and not isinstance(seed, int):
            init_loss_seed = seed[0]
        else:
            init_loss_seed = seed
        init_loss = get_init_loss_multi_batch(
            loader,
            num_chains * grad_accum_steps,
            model,
            evaluate,
            device,
            init_loss_seed,
        )
        # alternative: init_loss = get_init_loss_full_batch(loader, model, evaluate, device)
        # alternative: init_loss = get_init_loss_one_batch(loader, model, evaluate, device)
    for callback in callbacks:
        if isinstance(callback, (OnlineLLCEstimator, LLCEstimator)):
            setattr(callback, "init_loss", init_loss)

    # Temperature consistency warning
    if optimizer_kwargs is not None and (
        "nbeta" in optimizer_kwargs or "temperature" in optimizer_kwargs
    ):
        if "nbeta" in optimizer_kwargs:
            assert not any(
                getattr(callback, "temperature", None) is not None
                for callback in callbacks
            ), "If you're setting nbeta in optimizer_kwargs, don't set temperature in the callbacks."
        if "temperature" in optimizer_kwargs:
            assert not any(
                (
                    getattr(callback, "nbeta", None) is not None
                    and getattr(callback, "temperature") is None
                )
                for callback in callbacks
            ), "If you're setting temperature in optimizer_kwargs, don't set nbeta in the callbacks."
        warnings.warn(
            "If you're setting a nbeta or temperature in optimizer_kwargs, please also make sure to set it in the callbacks."
        )

    if cores is None:
        cores = min(cpu_count(), 4)

    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cpu" and use_amp:
        warnings.warn(
            "Automatic Mixed Precision (AMP) is not supported on CPU devices. Disabling AMP."
        )
        use_amp = False

    if device.type == "cuda":
        if gpu_idxs is not None:
            assert cores >= len(
                gpu_idxs
            ), "Number of cores must be greater than number of devices."
    else:
        assert (
            gpu_idxs is None
        ), "Multi-GPU sampling is only supported for CUDA devices. Check your device parameter."

    if seed is not None:
        warnings.warn(
            "You are using seeded runs, for full reproducibility check https://pytorch.org/docs/stable/notes/randomness.html"
        )
        if isinstance(seed, int):
            seeds = np.random.SeedSequence(seed).generate_state(num_chains)
        elif len(seed) != num_chains:
            raise ValueError("Length of seed list must match number of chains")
        else:
            seeds = seed
    else:
        seeds = [None] * num_chains

    if evaluate is None:
        # NOTE: If you'd like to update evaluate, please update _sample_single_chain as well.
        def evaluate(model, data):
            return model(data)

    validate_callbacks(callbacks)

    shared_kwargs = dict(
        ref_model=model,
        loader=cloudpickle.dumps(loader),
        evaluate=cloudpickle.dumps(evaluate),
        num_draws=num_draws,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        init_loss=init_loss,
        grad_accum_steps=grad_accum_steps,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        verbose=verbose,
        optimize_over_per_model_param=optimize_over_per_model_param,
        use_amp=use_amp,
    )

    if cores > 1:
        # mp.spawn(
        #     _sample_single_chain_mp,
        #     args=(seeds, shared_kwargs),
        #     nprocs=cores,
        #     join=True,
        #     start_method="spawn",
        # )

        if gpu_idxs is not None:
            device = [torch.device(f"cuda:{i}") for i in gpu_idxs]
        ctx = get_context("spawn")
        with ctx.Pool(cores) as pool:
            pool.map(
                _sample_single_chain,
                [
                    get_args(i, seeds, device, callbacks, shared_kwargs)
                    for i in range(num_chains)
                ],
            )
    else:
        for i in range(num_chains):
            _sample_single_chain(get_args(i, seeds, device, callbacks, shared_kwargs))

    for callback in callbacks:
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
