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
    for i in range(num_chains):
        draws = trace[i]
        plt.plot(sgld_step, draws, linewidth=1, label=f"chain {i}")

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

def sigma_helper(z, sigma_early, sigma_late, sigma_interp_end, interp_range=0.2):
    sigma_interp_start = interp_range * sigma_interp_end
    if z < sigma_interp_start:
        return sigma_early
    elif z > sigma_interp_end:
        return sigma_late
    else:
        return sigma_early + (sigma_late - sigma_early) / (
            sigma_interp_end - sigma_interp_start
        ) * (z - sigma_interp_start)
