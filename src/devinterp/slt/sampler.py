import itertools
import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Literal, Optional, Type, Union

import torch
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
    USE_TPU_BACKEND,
    EvaluateFn,
    call_with,
    get_init_loss_multi_batch,
    optimal_temperature,
    prepare_input,
    split_results,
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
        if hasattr(callback, "sample"):
            results.update(callback.sample())

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
