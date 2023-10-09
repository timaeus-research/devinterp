from collections import defaultdict
from typing import Dict, Generator, Iterable, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from devinterp.mechinterp.hooks import hook


def extract_activations_over_checkpoints(models: Iterable[nn.Module], xs, ys, *paths):
    def eval_activations(model) -> Dict[str, torch.Tensor]:
        hooked_model = hook(model, *paths)
        return hooked_model.run_with_cache(xs, ys)[1]
    
    for model in models:
        yield eval_activations(model)


def get_vectorized_activations_trace(models: Iterable[nn.Module], xs, ys, *paths):
    evals: Dict[str, list] = defaultdict(list)
    
    for activations in extract_activations_over_checkpoints(models, xs, ys, *paths):
        for path, activation in activations.items():
            evals[path].append(activation.flatten())

    return {
        k: torch.cat(v) for k, v in evals.items()
    }


def get_pca_activations_trace(models: Iterable[nn.Module], xs, ys, *paths, num_components=3) -> Dict[str, Tuple[PCA, np.ndarray]]:
    results = {}

    for path, activations in get_vectorized_activations_trace(models, xs, ys, *paths).items():
        pca = PCA(n_components=num_components)
        activations_reduced = pca.fit_transform(activations.detach().cpu().numpy())
        results[path] = pca, activations_reduced

    return results