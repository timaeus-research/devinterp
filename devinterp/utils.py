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

