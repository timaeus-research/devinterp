import warnings
from typing import Iterable, Literal, Tuple, Union

import numpy as np
import torch

Reduction = Literal["mean", "sum", "none"]
ReturnTensor = Literal["pt", "tf", "np"]


def convert_tensor(x: Iterable, return_tensor: ReturnTensor):
    if return_tensor == "pt":
        return torch.tensor(x)
    elif return_tensor == "tf":
        raise NotImplementedError
        # return tf.convert_to_tensor(x)
    elif return_tensor == "np":
        return np.array(x)
    else:
        raise ValueError(f"Unknown return_tensor: {return_tensor}")


def reduce_tensor(xs: Union[np.ndarray, torch.Tensor], reduction: Reduction):
    if reduction == "mean":
        return xs.mean()
    elif reduction == "sum":
        return xs.sum()
    elif reduction == "none":
        return xs
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def to_tuple(x: Union[Tuple[int, ...], int]) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (x,)
    else:
        return x


def int_logspace(start, stop, num, return_type="list"):
    result = set(int(i) for i in np.logspace(np.log10(start), np.log10(stop), num))

    if len(result) != num:
        warnings.warn(f"Number of steps in int_logspace is not {num}, got {len(result)}.")

    if return_type == "set":
        return set(result)

    result = sorted(list(result))

    if return_type == "list":
        return result
    elif return_type == "np":
        return np.array(result)
    else:
        raise ValueError(
            f"return_type must be either 'list' or 'set', got {return_type}"
        )


def int_linspace(start, stop, num, return_type="list"):
    result = set(int(i) for i in np.linspace(start, stop, num))

    if len(result) != num:
        warnings.warn(f"Number of steps in int_linspace is not {num}, got {len(result)}.")

    if return_type == "set":
        return result

    result = sorted(list(result))

    if return_type == "list":
        return list(result)
    elif return_type == "np":
        return np.array(result)
    else:
        raise ValueError(
            f"return_type must be either 'list' or 'set', got {return_type}"
        )
