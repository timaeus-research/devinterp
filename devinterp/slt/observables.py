from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import nn
from torchtyping import TensorType

from devinterp.utils import dict_compose


class MicroscopicObservable(Protocol):
    @staticmethod
    def __call__(
        xs: TensorType["batch_size", "input_dim"],
        ys: TensorType["batch_size", "output_dim"],
        yhats: TensorType["batch_size", "output_dim"],
        losses: TensorType["batch_size"],
        loss: TensorType[1],
        model: nn.Module,
    ) -> Any:
        raise NotImplementedError


def estimate_free_energy(losses: TensorType["num_draws"], num_samples: int, **_):
    """
    Estimate the free energy, $E[nL_n]$.
    """
    loss_avg = losses.mean()
    free_energy_estimate = loss_avg * num_samples

    return free_energy_estimate.item()


def estimate_rlct(
    loss_init: TensorType[1], losses: TensorType["num_draws"], num_samples: int, **_
):
    r"""
    Estimate the real log canonical ensemble (RLCT), $\hat\lambda$, using the WBIC.
    """
    loss_avg = losses.mean()
    rlct_estimate = (loss_avg - loss_init) * num_samples / np.log(num_samples)

    return rlct_estimate.item()


slt_summary_fn = dict_compose(
    free_energy=estimate_free_energy,
    rlct=estimate_rlct,
)
