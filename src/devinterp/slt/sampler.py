import os
from typing import Callable, Dict, List, Literal, Optional, Type, Union

import torch
from torch.utils.data import DataLoader

from devinterp.backends.default.slt.sampler import sample
from devinterp.optim.sgld import SGLD
from devinterp.slt.llc import LLCEstimator, OnlineLLCEstimator
from devinterp.utils import (
    EvaluateFn,
    default_nbeta,
    get_init_loss_multi_batch,
)


def estimate_learning_coeff_with_summary(
    model: torch.nn.Module,
    loader: DataLoader,
    callbacks: List[Callable] = [],
    evaluate: Optional[EvaluateFn] = None,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    sampling_method_kwargs: Optional[
        Dict[str, Union[float, Literal["adaptive"]]]
    ] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    init_loss: Optional[Union[float, torch.Tensor]] = None,
    gradient_accumulation_steps: int = 1,
    cores: Union[int, List[Union[str, torch.device]]] = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: Union[torch.device, str] = torch.device("cpu"),
    gpu_idxs: Optional[List[int]] = None,
    verbose: bool = True,
    optimize_over_per_model_param: Optional[Dict[str, torch.Tensor]] = None,
    online: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> dict:
    """
    Estimates the local learning coefficient and returns a dictionary of results.


    :param model: PyTorch model to sample from.
    :type model: torch.nn.Module
    :param loader: PyTorch DataLoader to sample from.
    :type loader: torch.utils.data.DataLoader
    :param callbacks: Additional callbacks to use during sampling. Since this function will automatically add the LLC estimator callback, \
    please do not include it in this list. An example of an additional callback is `devinterp.slt.mala.MalaAcceptanceRate`, which uses the Metropolis-Adjusted Langevin Algorithm's acceptance step to compute an acceptance rate.
    :type callbacks: list of callable
    :type evaluate: callable
    :param evaluate: Function to evaluate the model. Should take in a model and a batch of data and return a Transformer-like dictionary of metrics. \
    An example of an evaluate function is:
    ```python
    def evaluate(model, data):
        inputs, outputs = data
        return F.cross_entropy(model(inputs).logits, outputs), {
            "logits": model(inputs).logits
        } 
    ```

    :param sampling_method: PyTorch optimizer to use for sampling. Default is SGLD. Currently implemented alternatives include SGNHT.
    :type sampling_method: torch.optim.Optimizer
    :param sampling_method_kwargs: Dictionary of keyword arguments to pass to the optimizer. \
    For SGLD, this includes nbeta.
    :type sampling_method_kwargs: dict
    :param num_draws: Number of draws to sample per chain.
    :type num_draws: int
    :param num_chains: Number of chains to sample from.
    :type num_chains: int
    :param num_burnin_steps: Number of burn-in steps to use.
    :type num_burnin_steps: int
    :param num_steps_bw_draws: Number of steps between each draw.
    :type num_steps_bw_draws: int
    :param init_loss: Initial loss to use for the LLC estimator. If None, the initial loss will be computed using the first batch of data.
    :type init_loss: float
    :param gradient_accumulation_steps: Number of gradient accumulation steps to use per backward pass. Note that the effective batch size is batch_size * gradient_accumulation_steps.
    :type gradient_accumulation_steps: int
    :param cores: Number of cores to use for parallel sampling. Can be either an integer (will use cores starting from device 0) or a list of cores.
    :type cores: int or list of torch.device or str
    :param seed: Seed for reproducibility. If a list of seeds is provided, each chain will be seeded with the corresponding seed. 
    Otherwise, this parameter will be used as an offset that will be added to the chain index to seed each chain.
    :type seed: int or list of int
    :param device: Device to run the sampling on. Can be a torch.device or a string (e.g. "cuda:0"). Supports GPUs.
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
    :param dtype: Data type for sampling. Default is bfloat16 if BF16 is True, float16 if FP16 is True, or float32 otherwise.
    :type dtype: torch.dtype, optional
    :returns: A dictionary containing the local learning coefficient and loss traces.
    """

    model.to(device)
    if (
        "nbeta" not in sampling_method_kwargs
        and "temperature" in sampling_method_kwargs
    ):
        raise ValueError(
            "Temperature is deprecated, please switch to using nbeta here and in callbacks."
        )
    elif "nbeta" in sampling_method_kwargs and "temperature" in sampling_method_kwargs:
        raise ValueError(
            "Found temperature and nbeta in sampling_method_kwargs. Temperature is deprecated, please switch to using nbeta only (also in callbacks)."
        )

    sampling_method_kwargs.setdefault(
        "nbeta",
        default_nbeta(
            dataloader=loader, gradient_accumulation_steps=gradient_accumulation_steps
        ),
    )
    if not init_loss:
        init_loss = get_init_loss_multi_batch(
            loader, num_chains * gradient_accumulation_steps, model, evaluate, device
        )
        # alternative: init_loss = get_init_loss_full_batch(loader, model, evaluate, device)
        # alternative: init_loss = get_init_loss_one_batch(loader, model, evaluate, device)
    if online:
        llc_estimator = OnlineLLCEstimator(
            num_chains,
            num_draws,
            nbeta=sampling_method_kwargs["nbeta"],
            device=device,
            init_loss=init_loss,
        )
    else:
        llc_estimator = LLCEstimator(
            num_chains,
            num_draws,
            nbeta=sampling_method_kwargs["nbeta"],
            device=device,
            init_loss=init_loss,
        )

    callbacks = [llc_estimator, *callbacks]

    sample(
        model=model,
        loader=loader,
        evaluate=evaluate,
        sampling_method=sampling_method,
        sampling_method_kwargs=sampling_method_kwargs,
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        gradient_accumulation_steps=gradient_accumulation_steps,
        cores=cores,
        seed=seed,
        device=device,
        verbose=verbose,
        callbacks=callbacks,
        optimize_over_per_model_param=optimize_over_per_model_param,
        gpu_idxs=gpu_idxs,
        dtype=dtype,
        init_loss=init_loss,
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
    sampling_method_kwargs: Optional[
        Dict[str, Union[float, Literal["adaptive"]]]
    ] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    init_loss: Optional[Union[float, torch.Tensor]] = None,
    gradient_accumulation_steps: int = 1,
    cores: Union[int, List[Union[str, torch.device]]] = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: Union[torch.device, str] = torch.device("cpu"),
    verbose: bool = True,
    optimize_over_per_model_param: Optional[Dict[str, List[bool]]] = None,
    dtype: Optional[torch.dtype] = (
        torch.bfloat16
        if os.environ.get("BF16")
        else torch.float16
        if os.environ.get("FP16")
        else torch.float32
    ),
) -> float:
    """
    Estimates the local learning coefficient and returns a dictionary of results.


    :param model: PyTorch model to sample from.
    :type model: torch.nn.Module
    :param loader: PyTorch DataLoader to sample from.
    :type loader: torch.utils.data.DataLoader
    :param callbacks: Additional callbacks to use during sampling. Since this function will automatically add the LLC estimator callback, \
    please do not include it in this list. An example of an additional callback is `devinterp.slt.mala.MalaAcceptanceRate`, which uses the Metropolis-Adjusted Langevin Algorithm's acceptance step to compute an acceptance rate.
    :type callbacks: list of callable
    :type evaluate: callable
    :param evaluate: Function to evaluate the model. Should take in a model and a batch of data and return a Transformer-like dictionary of metrics. \
    An example of an evaluate function is:
    ```python
    def evaluate(model, data):
        inputs, outputs = data
        return F.cross_entropy(model(inputs).logits, outputs), {
            "logits": model(inputs).logits
        } 
    ```

    :param sampling_method: PyTorch optimizer to use for sampling. Default is SGLD. Currently implemented alternatives include SGNHT.
    :type sampling_method: torch.optim.Optimizer
    :param sampling_method_kwargs: Dictionary of keyword arguments to pass to the optimizer. \
    For SGLD, this includes nbeta.
    :type sampling_method_kwargs: dict
    :param num_draws: Number of draws to sample per chain.
    :type num_draws: int
    :param num_chains: Number of chains to sample from.
    :type num_chains: int
    :param num_burnin_steps: Number of burn-in steps to use.
    :type num_burnin_steps: int
    :param num_steps_bw_draws: Number of steps between each draw.
    :type num_steps_bw_draws: int
    :param init_loss: Initial loss to use for the LLC estimator. If None, the initial loss will be computed using the first batch of data.
    :type init_loss: float
    :param gradient_accumulation_steps: Number of gradient accumulation steps to use per backward pass. Note that the effective batch size is batch_size * gradient_accumulation_steps.
    :type gradient_accumulation_steps: int
    :param cores: Number of cores to use for parallel sampling. Can be either an integer (will use cores starting from device 0) or a list of cores.
    :type cores: int or list of torch.device or str
    :param seed: Seed for reproducibility. If a list of seeds is provided, each chain will be seeded with the corresponding seed. 
    Otherwise, this parameter will be used as an offset that will be added to the chain index to seed each chain.
    :type seed: int or list of int
    :param device: Device to run the sampling on. Can be a torch.device or a string (e.g. "cuda:0"). Supports GPUs.
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
    :param dtype: Data type for sampling. Default is bfloat16 if BF16 is True, float16 if FP16 is True, or float32 otherwise.
    :type dtype: torch.dtype, optional
    :returns: A single float representing the local learning coefficient.
    """

    return estimate_learning_coeff_with_summary(
        model=model,
        loader=loader,
        evaluate=evaluate,
        sampling_method=sampling_method,
        sampling_method_kwargs=sampling_method_kwargs,
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        gradient_accumulation_steps=gradient_accumulation_steps,
        cores=1,
        seed=seed,
        device=device,
        verbose=verbose,
        callbacks=callbacks,
        online=False,
        init_loss=init_loss,
        optimize_over_per_model_param=optimize_over_per_model_param,
        dtype=dtype,
    )["llc/mean"]
