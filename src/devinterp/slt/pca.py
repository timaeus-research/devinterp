import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import torch
from torch.utils.data import DataLoader
from torch import nn

from devinterp.utils import *


def get_output_for_model(model: nn.Module, dataloader: DataLoader, device):
    model.to(device)
    logits = []
    with torch.no_grad():
        for xs, ys in dataloader:
            xs, ys = xs.to(device), ys.to(device)
            outputs = model(xs, ys)
            logits.append(outputs)
    logits = torch.cat(logits).to("cpu")
    return logits


def get_output_for_models(models: nn.Module, dataloader: DataLoader, device):
    outputs = torch.stack(
        [get_output_for_model(model, dataloader, device) for model in models]
    )
    return outputs.reshape(len(outputs), -1)


def get_pca_components(samples: np.array, n_components: int):
    pca = PCA(n_components=n_components)
    transformed_samples = pca.fit_transform(samples)
    return pca, transformed_samples


def get_smoothed_pcs(
    samples, num_pca_components, early_smoothing, late_smoothing, late_smoothing_from
):
    smoothed_pcs = []
    for pca_component_i in range(0, num_pca_components):
        print(f"Processing smoothing for PC{pca_component_i+1}")
        smoothed_pc = np.copy(samples[:, 0])
        for z in range(len(samples)):
            sigma = sigma_helper(
                z, early_smoothing, late_smoothing, late_smoothing_from
            )
            smoothed_pc[z] = gaussian_filter1d(samples[:, pca_component_i], sigma)[z]
        smoothed_pcs.append(smoothed_pc)
    return smoothed_pcs


def plot_explained_variance(pca, title="Explained Variance", ax: plt.Axes = None):
    num_pca_components = pca.n_components
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 8))

    ax.bar(
        range(num_pca_components), pca.explained_variance_ratio_[:num_pca_components]
    )
    for i, ratio in enumerate(pca.explained_variance_ratio_[:num_pca_components]):
        ax.text(i, ratio, f"{ratio:.2f}", fontsize=12, ha="center", va="bottom")
    ax.set_title(title)
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained Variance")

    ax.set_xticks(range(num_pca_components), range(1, num_pca_components + 1))
