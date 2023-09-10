import warnings
from typing import Callable, Iterable, Literal, Optional, Protocol, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

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
        warnings.warn(
            f"Number of steps in int_logspace is not {num}, got {len(result)}."
        )

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
        warnings.warn(
            f"Number of steps in int_linspace is not {num}, got {len(result)}."
        )

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


def flatten_dict(dict_, prefix="", delimiter="/", flatten_lists=False):
    """
    Recursively flattens a nested dictionary of metrics into a single-level dictionary.

    Parameters:
        dict_ (dict): The dictionary to flatten. It can contain nested dictionaries and lists.
        prefix (str, optional): A string prefix to prepend to the keys in the flattened dictionary.
                                 This is used internally for the recursion and should not typically
                                 be set by the caller.
        delimiter (str, optional): The delimiter to use between keys in the flattened dictionary.
        flatten_lists (bool, optional): Whether to flatten lists in the dictionary. If True, list
                                        elements are treated as separate metrics.

    Returns:
        dict: A flattened dictionary where the keys are constructed by concatenating the keys from
              the original dictionary, separated by the specified delimiter.

    Example:
        Input:
            {
                "Train": {"Loss": "train_loss", "Accuracy": "train_accuracy"},
                "Test": {"Loss": "test_loss", "Details": {"Test/Accuracy": "test_accuracy"}},
                "List": [1, 2, [3, 4]]
            }

        Output (with flatten_lists=True):
            {
                'Train/Loss': 'train_loss',
                'Train/Accuracy': 'train_accuracy',
                'Test/Loss': 'test_loss',
                'Test/Details/Test/Accuracy': 'test_accuracy',
                'List/0': 1,
                'List/1': 2,
                'List/2/0': 3,
                'List/2/1': 4
            }
    """
    flattened = {}
    for key, value in dict_.items():
        if isinstance(value, dict):
            flattened.update(
                flatten_dict(
                    value,
                    prefix=f"{prefix}{key}{delimiter}",
                    delimiter=delimiter,
                    flatten_lists=flatten_lists,
                )
            )
        elif isinstance(value, list) and flatten_lists:
            for i, v in enumerate(value):
                if isinstance(v, (dict, list)):
                    flattened.update(
                        flatten_dict(
                            {str(i): v},
                            prefix=f"{prefix}{key}{delimiter}",
                            delimiter=delimiter,
                            flatten_lists=flatten_lists,
                        )
                    )
                else:
                    flattened[f"{prefix}{key}{delimiter}{i}"] = v
        else:
            flattened[f"{prefix}{key}"] = value
    return flattened


def map_nested(f, x):
    """Recursively applies a function to a nested dictionary or list."""
    if isinstance(x, dict):
        return {k: map_nested(f, v) for k, v in x.items()}
    elif isinstance(x, list):
        return [map_nested(f, v) for v in x]
    else:
        return f(x)


def dict_compose(**fns):
    """
    Composes multiple functions into a single function that applies each of them to its input.

    Parameters:
        fns: Keyword arguments where each key-value pair corresponds to the name and the function to be composed.

    Returns:
        fn: A new function that applies each input function to its arguments and stores the results in a dictionary.
    """

    def fn(**kwargs):
        output = {}
        for name, fn in fns.items():
            output[name] = fn(**kwargs)

        return output

    return fn


CriterionLiteral = Literal["mse_loss", "cross_entropy"]


class Criterion(Protocol):
    def __call__(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:  # type: ignore
        pass


def get_criterion(
    criterion: CriterionLiteral,
) -> Criterion:
    """
    Returns the criterion corresponding to the given string.
    """
    if isinstance(criterion, str):
        return getattr(F, criterion)
    return criterion


def nested_update(d1: dict, d2: dict):
    """
    Updates the values in d1 with the values in d2, recursively.
    """
    for k, v in d2.items():
        if isinstance(v, dict):
            d1[k] = nested_update(d1.get(k, {}), v)
        else:
            d1[k] = v
    return d1


def get_default_device():
    """
    Returns the default device for PyTorch.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
