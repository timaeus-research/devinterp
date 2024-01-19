import warnings
from typing import Callable, List, Union
import torch


class SamplerCallback:
    """
    Base class for creating callbacks used in sample().
    Each instantiated callback gets its functions update(), finalize() and sample() called if those exist.
    update() happens at every (non burn-in) sample draw, finalize() happens after all draws are finished,
    and sample() is a helper function that allows access to whatever relevant parameters the callback has computed.
    For legibility, ach callback can also access parameters in locals() when a class function is called.

    Parameters:
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.

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
