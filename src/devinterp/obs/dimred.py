from collections import defaultdict
from typing import (Dict, Generator, Iterable, List, Mapping, Optional,
                    Protocol, Tuple, Union)

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from devinterp.mechinterp.hooks import hook, run_with_hook


class DimReductionMethod(Protocol):
    def fit_transform(X: np.ndarray, y: Optional[np.ndarray] = None):
        ...


def generate_activations(models: Iterable[nn.Module], *args, paths: Optional[List[str]]=None, **kwargs) -> Generator[Dict[str, torch.Tensor], None, None]:
    paths = paths or [""]
    
    def eval_activations(model) -> Dict[str, Union[torch.Tensor, None]]:
        output, activations = run_with_hook(model, *args, paths=paths, **kwargs)

        if "" in paths:
            activations[""] = output

        return {k: v for k, v in activations.items() if v is not None}
    
    for model in models:
        yield eval_activations(model)


def reduce_dim_activations(method: DimReductionMethod, models: Iterable[nn.Module], *args, paths: Optional[List[str]]=None, **kwargs):
    """
    Run some inputs through a series of models, extract activations located at `paths`, and run dimensionality reduction on the vectorized result. 
    
    """

    activations_over_time = []

    for activations in generate_activations(models, *args, paths=paths, **kwargs):
        activations_over_time.append(np.array([a.flatten().detach().cpu().numpy() for a in activations.values() if a is not None]))

    activations_over_time = np.stack(activations_over_time)
    reduced_activations_over_time = method.fit_transform(activations_over_time)

    return reduced_activations_over_time
    