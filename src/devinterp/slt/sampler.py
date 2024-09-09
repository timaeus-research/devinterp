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
    gpu_idxs: Optional[List[int]] = None,
    verbose: bool = True,
    optimize_over_per_model_param: Optional[Dict[str, torch.Tensor]] = None,
    online: bool = False,
    use_amp: bool = False,
) -> dict:
    """
    Estimates the local learning coefficient and returns a dictionary of results.

    :param cores: Number of cores to use for parallel sampling. Can be either an integer (will use cores starting from device 0) or a list of cores.
    :type cores: int or list of torch.device or str
    :param seed: Seed for reproducibility. If a list of seeds is provided, each chain will be seeded with the corresponding seed. 
    Otherwise, this parameter will be used as an offset that will be added to the chain index to seed each chain.
    :type seed: int or list of int
    :param device: Device to run the sampling on. Can be a torch.device or a string (e.g. "cuda:0"). Supports GPUs and TPUs. To use TPUs:
    .. code-block:: python

        import os
        os.environ["USE_TPU_BACKEND"] = "1"
        import torch_xla.core.xla_model as xm
        DEVICE = xm.xla_device()
    \
    If you are using a TPU, make sure to set the environment variable `USE_TPU_BACKEND` to `1` before importing `devinterp`.
    :type device: torch.device or str
    :param gpu_idxs: List of GPU indices to use. If None, the device will be used as is. Provide a list of indices \
    and set cores greater than the length of the list to use multiple GPUs.
    :param verbose: Whether to display progress.
    :type verbose: bool
    :param optimize_over_per_model_param: Dictionary of booleans indicating whether to optimize over each parameter of the model. \
    Keys are parameter names, and values are boolean tensors that match the shape of the parameter. \
    A value of True (or 1) indicates that this particular element of the parameter should be optimized over. \
    None by default, which means that we optimize over all parameters.
    :type optimize_over_per_model_param: dict of str -> torch.Tensor[bool]
    :param online: Whether to use the online version of the LLC estimator.
    :type online: bool
    :param use_amp: Whether to use automatic mixed precision (casts to float16 on GPUs). Significantly speeds up sampling at the cost of a minor loss in precision (default: False).
    :type use_amp: bool
    :returns: A dictionary containing the local learning coefficient and loss traces.
    """

    model.to(device)
    # Temperature consistency warning
    if "nbeta" in optimizer_kwargs or "temperature" in optimizer_kwargs:
        warnings.warn(
            "Using passed in nbeta. Make sure callbacks are also initialized with the same nbeta."
        )
    else:
        warnings.warn("nbeta not set - using default nbeta.")

    optimizer_kwargs.setdefault("nbeta", default_nbeta(dataloader = loader, 
                                                       grad_accum_steps = grad_accum_steps))
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
        gpu_idxs=gpu_idxs,
        use_amp=use_amp,
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
