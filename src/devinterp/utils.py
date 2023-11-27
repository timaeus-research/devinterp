import numpy as np
import matplotlib.pyplot as plt


def plot_trace(trace, y_axis, x_axis='step', title=None, plot_mean=True, plot_std=True, fig_size=(12, 9)):
    num_chains, num_draws = trace.shape
    sgld_step = list(range(num_draws))

    # trace
    for i in range(num_chains):
        draws = trace[i]
        plt.plot(sgld_step, draws, linewidth=1, label=f'chain {i}')

    # mean
    mean = np.mean(trace, axis=0)
    plt.plot(sgld_step, mean, color='black', linestyle='--', linewidth=2, label='mean', zorder=3)
    
    # std
    std = np.std(trace, axis=0)
    plt.fill_between(sgld_step, mean - std, mean + std, color='gray', alpha=0.3, zorder=2)

    if title is None:
        title = f'{y_axis} values over sampling draws'
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))    
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.figure(figsize=fig_size)
    plt.tight_layout()
    plt.show()
