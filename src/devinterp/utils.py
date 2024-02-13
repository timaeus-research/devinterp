from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_trace(
    trace,
    y_axis,
    x_axis="step",
    title=None,
    plot_mean=True,
    plot_std=True,
    fig_size=(12, 9),
    true_lc=None,
):
    num_chains, num_draws = trace.shape
    sgld_step = list(range(num_draws))
    if true_lc:
        plt.axhline(y=true_lc, color="r", linestyle="dashed")
    # trace
    for i in range(num_chains):
        draws = trace[i]
        plt.plot(sgld_step, draws, linewidth=1, label=f"chain {i}")

    # mean
    mean = np.mean(trace, axis=0)
    plt.plot(
        sgld_step,
        mean,
        color="black",
        linestyle="--",
        linewidth=2,
        label="mean",
        zorder=3,
    )

    # std
    std = np.std(trace, axis=0)
    plt.fill_between(
        sgld_step, mean - std, mean + std, color="gray", alpha=0.3, zorder=2
    )

    if title is None:
        title = f"{y_axis} values over sampling draws"
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.figure(figsize=fig_size)
    plt.tight_layout()
    plt.show()


def optimal_temperature(data):
    if isinstance(data, DataLoader):
        return len(data.dataset) / np.log(len(data.dataset))
    elif isinstance(data, Dataset):
        return len(data) / np.log(len(data))
    elif isinstance(data, int):
        return data / np.log(data)


def get_init_loss_one_batch(dataloader, model, criterion, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        xs, ys = next(iter(dataloader))
        xs, ys = xs.to(device), ys.to(device)
        y_preds = model(xs)
        loss = criterion(y_preds, ys).detach().item()
    return loss


def get_init_loss_full_batch(dataloader, model, criterion, device):
    model = model.to(device)
    model.eval()
    loss = 0.0
    with torch.no_grad():
        xs, ys = next(iter(dataloader))
        xs, ys = xs.to(device), ys.to(device)
        y_preds = model(xs)
        loss += criterion(y_preds, ys).detach().item()
    return loss
