import warnings
from typing import Callable, Dict, List, Literal, Optional, Type, Union

import torch
from torch.utils.data import DataLoader

from devinterp.optim.sgld import SGLD
from devinterp.slt.llc import LLCEstimator, OnlineLLCEstimator
from devinterp.utils import (
    USE_TPU_BACKEND,
    EvaluateFn,
    default_nbeta,
    get_init_loss_multi_batch,
)

if USE_TPU_BACKEND:
    from devinterp.backends.tpu.slt.sampler import sample
else:
    from devinterp.backends.default.slt.sampler import sample


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
    cores: Union[int, List[Union[str, torch.device]]] = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: Union[torch.device, str] = torch.device("cpu"),
    verbose: bool = True,
    optimize_over_per_model_param: Optional[Dict[str, List[bool]]] = None,
    online: bool = False,
) -> dict:
    """
    Estimates the local learning coefficient and returns a dictionary of results.
    :param cores: Number of cores to use for parallel sampling. Can be either an integer (will use cores starting from device 0) or a list of cores.
    :type cores: int or list of torch.device or str
    """

    # Temperature consistency warning
    if "nbeta" in optimizer_kwargs and not "temperature" in optimizer_kwargs:
        warnings.warn(
            "Using passed in nbeta. Make sure callbacks are also initialized with the same nbeta."
        )
    elif not "nbeta" in optimizer_kwargs and "temperature" in optimizer_kwargs:
        warnings.warn(
            "Temperature is deprecated, please switch to using nbeta here and in callbacks."
        )
        warnings.warn(
            "Using passed in temperature. Make sure callbacks are also initialized with the same temperature."
        )
        optimizer_kwargs["nbeta"] = optimizer_kwargs.pop("temperature")
    elif "nbeta" in optimizer_kwargs and "temperature" in optimizer_kwargs:
        raise ValueError(
            "Found temperature and nbeta in optimizer_kwargs. Temperature is deprecated, please switch to using nbeta only (also in callbacks)."
        )

    else:
        warnings.warn("nbeta not set - using default nbeta.")

    optimizer_kwargs.setdefault("nbeta", default_nbeta(loader))
    if not init_loss:
        init_loss = get_init_loss_multi_batch(
            loader, num_chains, model, evaluate, device
        )
        # alternative: init_loss = get_init_loss_full_batch(loader, model, evaluate, device)
        # alternative: init_loss = get_init_loss_one_batch(loader, model, evaluate, device)
    if online:
        llc_estimator = OnlineLLCEstimator(
            num_chains,
            num_draws,
            nbeta=optimizer_kwargs["nbeta"],
            device=device,
            init_loss=init_loss,
        )
    else:
        llc_estimator = LLCEstimator(
            num_chains,
            num_draws,
            nbeta=optimizer_kwargs["nbeta"],
            device=device,
            init_loss=init_loss,
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
    cores: Union[int, List[Union[str, torch.device]]] = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: Union[torch.device, str] = torch.device("cpu"),
    verbose: bool = True,
    optimize_over_per_model_param: Optional[Dict[str, List[bool]]] = None,
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
        cores=1,
        seed=seed,
        device=device,
        verbose=verbose,
        callbacks=callbacks,
        online=False,
        init_loss=init_loss,
        optimize_over_per_model_param=optimize_over_per_model_param,
    )["llc/mean"]
