from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice


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


def optimal_temperature(dataloader: DataLoader):
    if isinstance(dataloader, DataLoader):
        return dataloader.batch_size / np.log(dataloader.batch_size)
    elif isinstance(dataloader, int):
        return dataloader / np.log(dataloader)
    else:
        raise NotImplementedError(
            f"Temperature for data type {type(dataloader)} not implemented, use DataLoader or int instead."
        )


def get_init_loss_one_batch(dataloader, model, criterion, device):
    model = model.to(device)
    model.train()  # to make sure we're using train loss, comparable to train loss of sampler()
    with torch.no_grad():
        xs, ys = next(iter(dataloader))
        xs, ys = xs.to(device), ys.to(device)
        y_preds = model(xs)
        loss = criterion(y_preds, ys).detach().item()
    return loss


def get_init_loss_multi_batch(dataloader, n_batches, model, criterion, device):
    model = model.to(device)
    model.train()
    loss = 0.0
    with torch.no_grad():
        for xs, ys in islice(dataloader, n_batches):
            xs, ys = xs.to(device), ys.to(device)
            y_preds = model(xs)
            loss += criterion(y_preds, ys).detach().item()
    return loss / n_batches


def get_init_loss_full_batch(dataloader, model, criterion, device):
    model = model.to(device)
    model.train()
    loss = 0.0
    with torch.no_grad():
        for xs, ys in dataloader:
            xs, ys = xs.to(device), ys.to(device)
            y_preds = model(xs)
            loss += criterion(y_preds, ys).detach().item()
    return loss / len(dataloader)
