"""SGLD sampler: runs a single SGLD chain with callbacks.

The inner loop for SGLD sampling. Called by sample() in sampling.py
once per chain.
"""

from __future__ import annotations

import gc
import random
import warnings
from collections.abc import Iterator
from copy import deepcopy
from itertools import cycle
from typing import Any, Callable, Protocol

import numpy as np
import torch
from devinterp.optim import SGLD
from devinterp.slt.config import EpochMode
from devinterp.slt.lm_loss import NonFiniteLogitsError
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import trange

# param name -> mask tensor (or None for unrestricted).
# Only params in the dict are optimized; all others are frozen.
ParamMasks = dict[str, torch.Tensor | None]


class ChainHealthError(Exception):
    pass


class StepCallback(Protocol):
    def __call__(
        self,
        chain: int,
        step: int,
        optimizer: torch.optim.Optimizer,
    ) -> None: ...


class MicroCallback(Protocol):
    """Called once per micro-batch (inside the gradient accumulation loop)."""

    def __call__(
        self,
        loss: torch.Tensor,
        input_ids: torch.Tensor,
        chain: int,
        step: int,
        micro_step: int,
    ) -> None: ...


def calculate_num_steps(
    *,
    num_draws: int,
    num_steps_bw_draws: int,
    num_burnin_steps: int,
) -> int:
    return num_draws * num_steps_bw_draws + num_burnin_steps


def set_seed(seed: int, device: str | None = None) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if device and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def _gc_and_empty_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_transformer_lens_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "cfg") and hasattr(model.cfg, "n_heads")


def _make_feed(
    loader: DataLoader, epoch_mode: EpochMode, num_batches_needed: int
) -> Iterator:
    """Create a batch iterator, cycling if needed.

    Raises RuntimeError (not StopIteration) if data runs out, to avoid
    silently terminating the sampling loop.
    """
    if epoch_mode == "cycle":
        # itertools.cycle caches the first pass and replays it — matches aether's
        # behavior, which means shuffle=True + cycle replays the first-epoch shuffle
        # rather than reshuffling each epoch.
        yield from cycle(loader)
    elif epoch_mode == "once":
        if len(loader) < num_batches_needed:
            raise ValueError(
                f"Dataset too small: need {num_batches_needed} batches, have {len(loader)}. "
                f"Use epoch_mode='cycle' or a larger dataset."
            )
        yield from loader
        raise RuntimeError(
            "Data loader exhausted during sampling (epoch_mode='once'). "
            "Use epoch_mode='cycle' or a larger dataset."
        )
    else:
        raise ValueError(f"Unknown epoch_mode: {epoch_mode!r}")


def sample_single_chain(
    ref_model: nn.Module,
    dataset: torch.utils.data.Dataset,
    evaluate: Callable[[nn.Module, torch.Tensor], tuple[torch.Tensor, dict[str, Any]]],
    param_masks: ParamMasks,
    num_draws: int = 100,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    gradient_accumulation_steps: int = 1,
    sampling_method: type[torch.optim.Optimizer] = SGLD,
    sampling_method_kwargs: dict[str, Any] | None = None,
    chain: int = 0,
    seed: int | None = None,
    dataloader_seed: int | None = None,
    device: str = "cpu",
    callbacks: list[Callable] | None = None,
    step_callback: StepCallback | None = None,
    micro_callback: MicroCallback | None = None,
    batch_size: int = 32,
    init_noise: float | None = None,
    shuffle: bool = True,
    epoch_mode: EpochMode = "cycle",
) -> None:
    """Sample a single SGLD chain."""
    dataloader_rng = torch.Generator(device="cpu")
    if dataloader_seed is not None:
        dataloader_rng.manual_seed(dataloader_seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=dataloader_rng,
        drop_last=True,
    )

    model = deepcopy(ref_model)
    if is_transformer_lens_model(model):
        model = model.to(device, print_details=False)  # pyright: ignore[reportCallIssue]
    else:
        model = model.to(device)

    if seed is not None:
        set_seed(seed, device=device)

    sampling_method_kwargs = sampling_method_kwargs or {}

    name_to_param = dict(model.named_parameters())
    for name, param in name_to_param.items():
        if name not in param_masks:
            param.requires_grad_(False)

    opt_params = [
        {
            "params": name_to_param[name],
            "mask": mask.to(device) if mask is not None else None,
        }
        for name, mask in param_masks.items()
    ]
    optimizer = sampling_method(
        opt_params,
        **sampling_method_kwargs,
    )

    if init_noise:
        for name, mask in param_masks.items():
            p = name_to_param[name]
            m = mask.to(p.device) if mask is not None else 1.0
            p.data.add_(torch.randn_like(p.data) * init_noise * m)

    num_steps = calculate_num_steps(
        num_draws=num_draws,
        num_steps_bw_draws=num_steps_bw_draws,
        num_burnin_steps=num_burnin_steps,
    )

    total_batches = num_steps * gradient_accumulation_steps
    feed = _make_feed(loader, epoch_mode, total_batches)

    model.train()
    _gc_and_empty_cache()

    try:
        with trange(0, num_steps, desc=f"Chain {chain}", leave=True) as pbar:
            for i in pbar:
                loss = torch.zeros((), device=device, dtype=torch.float32)
                for j in range(gradient_accumulation_steps):
                    data = next(feed)
                    input_ids = data["input_ids"]
                    data["input_ids"] = input_ids.to(device)
                    _loss, _ = evaluate(model, data)

                    if micro_callback is not None:
                        micro_callback(
                            loss=_loss.detach(),
                            input_ids=input_ids,
                            chain=chain,
                            step=i,
                            micro_step=j,
                        )

                    _loss = _loss.float()
                    (_loss.mean() / gradient_accumulation_steps).backward()
                    loss = loss + _loss.detach() / gradient_accumulation_steps

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    raise ChainHealthError(f"Chain {chain} failed: Loss is NaN or Inf")

                optimizer.step()

                if step_callback is not None:
                    step_callback(chain, i, optimizer)

                optimizer.zero_grad(set_to_none=True)

                draw, steps_since_draw = divmod(
                    i - num_burnin_steps, num_steps_bw_draws
                )
                if draw >= 0 and steps_since_draw == 0 and callbacks:
                    with torch.no_grad():
                        for callback in callbacks:
                            callback(
                                loss=loss.detach(),
                                draw=draw,
                                chain=chain,
                                model=model,
                                optimizer=optimizer,
                            )

    except (ChainHealthError, NonFiniteLogitsError) as e:
        warnings.warn(f"Chain failed: {e}")

    del model, optimizer
    _gc_and_empty_cache()
