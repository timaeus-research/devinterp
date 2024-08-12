import warnings
from typing import Callable, List, Union

import torch


class SamplerCallback:
    """
    Base class for creating callbacks used in :func:`devinterp.slt.sampler.get_results()`.
    Each instantiated callback gets its :python:`__call__` called every sample, and :python:`finalize` called at the end of sample (if it exists).
    Each callback method can access parameters in :python:`locals()`, so there's no need to pass variables along explicitly.

    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'.
    :raises NotImplementedError: if :python: `__call__` :python: `sample` are not overwritten.

    Note:
        - :python:`mps` devices might not work for all callbacks.
    """

    def __init__(self, device: Union[torch.device, str] = "cpu"):
        self.device = device

    def share_memory_(self):
        if self.device == "mps":
            warnings.warn("Cannot share memory with MPS device.")
            return self

        for attr in dir(self):
            if attr.startswith("_"):
                continue

            attr = getattr(self, attr)
            if hasattr(attr, "share_memory_"):
                attr.share_memory_()

        return self

    def __call__(self, *args, **kwargs):
        """Gets called at every (non burn-in) sample draw. Can access any variable in :python:`locals()` when called.
        Should be used for calucalting stats at every sample draw, for example running average chain loss.
        :raises NotImplementedError: if not overwritten for inherited classes.
        """
        raise NotImplementedError

    def finalize(self, *args, **kwargs):
        """Gets called at the end of sampling. Can access any variable in :python:`locals()` when called. Should be used for calucalting stats over chains, for example average chain loss."""
        pass

    def get_results(self, *args, **kwargs):
        """Does not get called automatically, but functions as an interface to easily access stats calculated by the callback."""
        raise NotImplementedError


def validate_callbacks(callbacks: List[Callable]):
    for i, callback in enumerate(callbacks):
        if isinstance(callback, SamplerCallback) and hasattr(callback, "base_callback"):
            base_callback = callback.base_callback
            base_callback_exists = False
            for j, other_callback in enumerate(callbacks):
                if other_callback is base_callback:
                    if j > i:
                        raise ValueError(
                            f"Derivative callback {callback} must be called after base callback {base_callback}."
                        )
                    base_callback_exists = True
            if not base_callback_exists:
                raise ValueError(
                    f"Base callback {base_callback} of derivative callback {callback} was not passed."
                )
    return True
