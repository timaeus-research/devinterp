from typing import Callable

import torch
from torch import Tensor


def infer_data_format(t: torch.Tensor) -> str:
    # PyTorch generally uses the "NCHW" format for images, and this is common.
    # You may want to use some specific logic to infer the format if this is not the case for your data.

    shape = t.shape
    if len(shape) != 4:
        return None

    # Example: If the channel dimension is at the second position (index 1) and its size is 3, it may be "NCHW".
    if shape[1] == 3:
        return "NCHW"
    # If the channel dimension is at the last position (index 3) and its size is 3, it may be "NHWC".
    elif shape[3] == 3:
        return "NHWC"
    
    return None


def _dot(x: Tensor, y: Tensor) -> Tensor:
    return torch.sum(x * y, dim=-1)

def _dot_cossim(x: Tensor, y: Tensor, cossim_pow: float = 0) -> Tensor:
    eps = 1e-4
    xy_dot = _dot(x, y)
    if cossim_pow == 0:
        return torch.mean(xy_dot)
    x_mags = torch.sqrt(_dot(x, x))
    y_mags = torch.sqrt(_dot(y, y))
    cossims = xy_dot / (eps + x_mags) / (eps + y_mags)
    floored_cossims = torch.maximum(torch.tensor(0.1), cossims)
    return torch.mean(xy_dot * floored_cossims**cossim_pow)

def _extract_act_pos(acts: Tensor, x: int = None, y: int = None) -> Tensor:
    shape = acts.shape
    x_ = shape[1] // 2 if x is None else x
    y_ = shape[2] // 2 if y is None else y
    return acts[:, x_:x_+1, y_:y_+1]

def _make_arg_str(arg: str) -> str:
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return "..." if too_big else arg

def _T_force_NHWC(T: Callable) -> Callable:
    def T2(name: str) -> Tensor:
        t = T(name)
        shape = t.shape
        if len(shape) == 2:
            return t[:, None, None, :]
        elif len(shape) == 4:
            fmt = infer_data_format(t)
            if fmt == "NCHW":
                return t.permute(0, 2, 3, 1)
        return t
    return T2

def _T_handle_batch(T: Callable, batch: int = None) -> Callable:
    def T2(name: str) -> Tensor:
        t = T(name)
        if isinstance(batch, int):
            return t[batch:batch+1]
        else:
            return t
    return T2
