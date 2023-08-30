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

<<<<<<< HEAD
def flatten_dict(dict_, prefix="", delimiter="/"):
=======

def flatten_dict(metrics, prefix="", delimiter="/"):
>>>>>>> main
    """
    Recursively flattens a nested dictionary of metrics into a single-level dictionary.

    Parameters:
        metrics (dict): The dictionary to flatten. It can contain nested dictionaries.
        prefix (str, optional): A string prefix to prepend to the keys in the flattened dictionary.
                                 This is used internally for the recursion and should not typically
                                 be set by the caller.

    Returns:
        dict: A flattened dictionary where the keys are constructed by concatenating the keys from
              the original dictionary, separated by slashes.

    Example:
        Input:
            {
                "Train": {"Loss": "train_loss", "Accuracy": "train_accuracy"},
                "Test": {"Loss": "test_loss", "Details": {"Test/Accuracy": "test_accuracy"}},
            }

        Output:
            {
                'Train/Loss': 'train_loss',
                'Train/Accuracy': 'train_accuracy',
                'Test/Loss': 'test_loss',
                'Test/Details/Test/Accuracy': 'test_accuracy'
            }
    """
    flattened = {}
    for key, value in dict_.items():
        if isinstance(value, dict):
            flattened.update(
                flatten_dict(
                    value, prefix=f"{prefix}{key}{delimiter}", delimiter=delimiter
                )
            )
        else:
            flattened[f"{prefix}{key}"] = value
    return flattened


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
