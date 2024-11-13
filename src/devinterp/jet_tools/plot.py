from copy import deepcopy

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from IPython.display import display
from PyMoments import kstat

from .diffs import ith_place_nth_diff, joint_ith_place_nth_diff


# plot stats for each dim (flattened)
def plot_second_order_one_place_stats(wt, n, title="zeros"):
    num_chains = len(wt)
    num_draws = len(wt[0])
    all_stds_per_i = []
    all_norms_per_i = []
    i_values = np.arange(1, num_draws // n - 2, 1)
    for i in i_values:
        diffs = ith_place_nth_diff(wt, i, n)
        # Calculate norm of differences across all chains
        diff_norms = np.linalg.norm(diffs, axis=2)  # Norm across parameter dimensions
        mean_norm = np.mean(diff_norms, axis=1)  # Mean across chains
        all_norms_per_i.append(mean_norm)
        std_per_i = np.std(diffs, axis=1)
        all_stds_per_i.append(std_per_i)
    fig, ax1 = plt.subplots()  # Create a figure and a primary axe
    for tensor_index, std in enumerate(np.array(all_stds_per_i).T):
        ax1.plot(i_values, std[0], label=f"std dim {tensor_index}")
    ax1.plot(i_values, all_norms_per_i, label=f"norm")
    plt.legend(loc="lower center")
    plt.title(title)
    plt.show()


def plot_second_order_one_place_dot_products(
    wt, gd, n, i_to_plot_draws_for=None, title="zeros"
):
    num_draws = len(wt[0])
    num_chains = len(wt)
    all_dot_products = []
    i_values = np.arange(1, num_draws // n - 2, 1)
    for i in i_values:
        # note the sign flip, np.diff returns a[i+1] - a[i] and we need the opposite
        diffs = -ith_place_nth_diff(wt, i, n)
        # Dot product across parameter dimensions
        dot_product = np.sum(diffs * gd[:, : len(diffs[0])], axis=2)
        if i == i_to_plot_draws_for:
            # cumulative average
            cum_avg_dot_up_to_draw = np.cumsum(dot_product, axis=1)[0] / np.arange(
                1, num_draws - i + 1
            )
        mean_dot = np.mean(dot_product, axis=1)  # Mean across draws
        all_dot_products.append(mean_dot)

    fig, ax1 = plt.subplots()
    ax1.plot(i_values, all_dot_products, label="dot product")
    plt.xlabel("ith diff")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()
    if i_to_plot_draws_for is not None:
        fig, ax1 = plt.subplots()
        ax1.plot(
            np.arange(0, num_draws - i_to_plot_draws_for, 1),
            cum_avg_dot_up_to_draw,
            label="",
        )
        plt.xlabel("draws")
        plt.title(title + f", i={i_to_plot_draws_for}")
        plt.show()


def plot_second_order_two_place_stats(wt, n, title="zeros"):
    num_chains = len(wt)
    num_draws = len(wt[0])
    all_stds_x_per_i = []
    all_stds_y_per_i = []
    cov_x_y_per_i = []
    all_norms_per_i = []
    i_values = np.arange(1, num_draws // n - 2, 1)

    for i in i_values:
        diffs = ith_place_nth_diff(wt, i, n)
        # Calculate norm of differences across all chains
        diff_norms = np.linalg.norm(diffs, axis=2)  # Norm across parameter dimensions
        mean_norm = np.mean(diff_norms, axis=1)  # Mean across chains
        all_norms_per_i.append(mean_norm)
        std_x_y = np.std(diffs, axis=1)[0]
        cov_x_y = kstat(diffs.reshape(-1, diffs.shape[-1]), (0, 1))
        all_stds_x_per_i.append(std_x_y[0])
        all_stds_y_per_i.append(std_x_y[1])
        cov_x_y_per_i.append(cov_x_y)

    fig, ax1 = plt.subplots()  # Create a figure and a primary axe
    # Plot data on the primary y-axis

    ax1.plot(i_values, all_stds_x_per_i, label="xx")
    ax1.plot(i_values, all_stds_y_per_i, label="yy")
    ax1.plot(i_values, cov_x_y_per_i, label="xy")
    ax1.plot(i_values, all_norms_per_i, label="norm")

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # Plot data on the secondary y-axis
    # ax2.plot(i_values, all_mean_dists_per_n, label='dist', color='r') # should be secondary_y

    # Add legends for both axes
    ax1.legend(loc="upper left")
    # ax2.legend(loc='upper right')
    plt.title(title)

    plt.show()


def plot_third_order_stats_per_dim(wt, n, title="zeros", up_to_dim=3):
    num_draws = len(wt[0])
    diff_third_cum_per_i_per_n = [[] for _ in range(up_to_dim)]

    i_values = np.arange(1, num_draws - 3, 1)
    for i in i_values:
        diffs = ith_place_nth_diff(wt, i, n)
        diffs = diffs.reshape(-1, diffs.shape[-1])
        for dim in range(up_to_dim):
            diff_third_cum_per_i_per_n[dim].append(kstat(diffs, (dim, dim, dim)))

    fig, ax1 = plt.subplots()  # Create a figure and a primary axes

    # Plot data on the primary y-axis
    for dim_number, diff_third_cum in enumerate(diff_third_cum_per_i_per_n):
        ax1.plot(i_values, diff_third_cum, label=f"dim={dim_number}")

    # Add legend
    ax1.legend(loc="upper left")
    plt.title(title)
    plt.show()


def plot_trajectories(weight_trajectories, names, model, n_bins=20):
    fig, axes = plt.subplots(1, len(weight_trajectories), figsize=(6, 6))
    if len(weight_trajectories) == 1:
        axes = [axes]
    model_copy = deepcopy(model)
    range_size = 5
    w1_range = np.linspace(-range_size, range_size, 21)
    w2_range = np.linspace(-range_size, range_size, 21)
    w1_vals, w2_vals = np.meshgrid(w1_range, w2_range)
    Z = np.zeros_like(w1_vals, dtype=float)

    for i in range(w1_vals.shape[0]):
        for j in range(w1_vals.shape[1]):
            w1 = w1_vals[i, j]
            w2 = w2_vals[i, j]
            model_copy.weights = nn.Parameter(
                torch.tensor([w1, w2], dtype=torch.float32).to("cpu")
            )
            Z[i, j] = (
                model_copy.to("cpu")(torch.tensor(1.0).to("cpu")).item() ** 2
            ) + 0.0001 * w1 * w2  # MSE, so square this

    custom_levels = np.linspace(Z.min(), Z.max() * 0.04, n_bins)

    for i, weight_trajectory in enumerate(weight_trajectories):
        # axes[i].contourf(
        #     w1_vals, w2_vals, Z, levels=custom_levels, cmap=contour_cmap, alpha=0.8
        # )
        draws_array = np.array(
            [
                weight
                for weight in weight_trajectory
                if w1_range[0] <= weight[0][0] <= w1_range[-1]
                and w2_range[0] <= weight[0][1] <= w2_range[-1]
            ]
        )
        print(np.shape(draws_array))
        sns.scatterplot(
            x=draws_array[:, 0, 0], y=draws_array[:, 0, 1], marker="x", ax=axes[i], s=10
        )
        axes[i].axhline(0, linestyle="--", color="gray")
        axes[i].axvline(0, linestyle="--", color="gray")
        axes[i].set_xlabel(r"$w_{1}$")
        axes[i].set_ylabel(r"$w_{2}$")
        axes[i].set_title(names[i])
        axes[i].grid(False)
    plt.show()


def plot_multi_trajectories(wt, i_range, diffs_range, legend, model, n_bins=20):
    num_chains = len(wt)
    for n in diffs_range:
        for i in i_range:
            diffs = ith_place_nth_diff(wt, i, n)
            plot_trajectories(
                diffs,
                names=[
                    (legend + f"_{chain} n={n} i={i}") for chain in range(num_chains)
                ],
                model=model,
                n_bins=n_bins,
            )


# Courtesy of ChatGPT
def compute_binned_averages(A, xmin, xmax, ymin, ymax, n):
    # Number of columns to average over
    m = A.shape[1] - 2

    # Initialize the output array B of shape (n, n, m) with zeros
    B = np.zeros((n, n, m))
    # Initialize a count array of shape (n, n) to keep track of the number of elements in each bin
    count = np.zeros((n, n))

    # Define bin width for x and y
    x_bin_width = (xmax - xmin) / n
    y_bin_width = (ymax - ymin) / n

    # Iterate over each element in A
    for k in range(A.shape[0]):
        x, y = A[k, 0], A[k, 1]
        values = A[k, 2:]  # Extract the values starting from the third column

        # Check which bin (i, j) the point (x, y) belongs to
        i = int((x - xmin) // x_bin_width)
        j = int((y - ymin) // y_bin_width)

        # Ensure the indices are within bounds
        if 0 <= i < n and 0 <= j < n:
            # Add the values to the corresponding bin (for each column l in A[k][2:])
            B[i, j, :] += values
            # Increment the count for that bin
            count[i, j] += 1

    # Compute the average for each bin and each column
    # Avoid division by zero by only averaging where count > 0
    for l in range(m):
        B[:, :, l] = np.divide(
            B[:, :, l], count, where=(count > 0), out=np.zeros_like(B[:, :, l])
        )

    return B


def compute_binned_averages_multi_with_std(A, xmin, xmax, ymin, ymax, n):
    # Number of columns to average over
    m = A.shape[1] - 2

    # Initialize the output array B of shape (n, n, m) for averages
    B_avg = np.zeros((n, n, m))
    # Initialize the array to hold the sum of squares
    B_sum_squares = np.zeros((n, n, m))
    # Initialize a count array of shape (n, n) to keep track of the number of elements in each bin
    count = np.zeros((n, n))

    # Define bin width for x and y
    x_bin_width = (xmax - xmin) / n
    y_bin_width = (ymax - ymin) / n

    # Iterate over each element in A
    for k in range(A.shape[0]):
        x, y = A[k, 0], A[k, 1]
        values = A[k, 2:]  # Extract the values starting from the third column

        # Check which bin (i, j) the point (x, y) belongs to
        i = int((x - xmin) // x_bin_width)
        j = int((y - ymin) // y_bin_width)

        # Ensure the indices are within bounds
        if 0 <= i < n and 0 <= j < n:
            # Add the values to the corresponding bin (for each column l in A[k][2:])
            B_avg[i, j, :] += values
            B_sum_squares[i, j, :] += values**2
            # Increment the count for that bin
            count[i, j] += 1

    # Compute the average for each bin and each column
    # Avoid division by zero by only averaging where count > 0
    for l in range(m):
        B_avg[:, :, l] = np.divide(
            B_avg[:, :, l], count, where=(count > 0), out=np.zeros_like(B_avg[:, :, l])
        )

    # Now compute the standard deviation for each bin and each column
    B_std = np.zeros((n, n, m))
    for l in range(m):
        # Use the formula std_dev = sqrt((sum_squares / N) - (mean^2))
        B_std[:, :, l] = np.sqrt(
            np.divide(B_sum_squares[:, :, l], count, where=(count > 0))
            - B_avg[:, :, l] ** 2
        )
    return B_avg, B_std


def plot_second_order_one_place_dot_products_2d(wt, gd, n, num_bins=15, title="zeros"):
    import ipywidgets as widgets
    from IPython.display import display

    # Create sliders
    i_slider = widgets.IntSlider(
        value=100,
        min=10,
        max=2000,
        step=10,
        description="i_to_plot_for:",
        continuous_update=False,
    )

    draws_slider = widgets.IntSlider(
        value=10000,
        min=100,
        max=10000,
        step=100,
        description="num_draws:",
        continuous_update=False,
    )
    # Get the range for both dimensions
    x_min, x_max = np.min(wt[..., 0]), np.max(wt[..., 0])
    y_min, y_max = np.min(wt[..., 1]), np.max(wt[..., 1])

    # Create bins
    x_bins = np.linspace(x_min, x_max, num_bins)
    y_bins = np.linspace(y_min, y_max, num_bins)

    def update_plot(i_to_plot_for, num_draws_to_plot_for):
        if num_draws_to_plot_for < i_to_plot_for:
            print("num_draws_to_plot_for must be greater than i_to_plot_for")
            return
        wt_to_plot_for = wt[:, :num_draws_to_plot_for]
        gd_to_plot_for = gd[:, :num_draws_to_plot_for]
        # note the sign flip, np.diff returns a[i+1] - a[i] and we need the opposite
        diffs_per_draw = -ith_place_nth_diff(wt_to_plot_for, i_to_plot_for, n)
        # Dot product across parameter dimensions
        dot_product_per_draw = np.sum(
            diffs_per_draw * gd_to_plot_for[:, : len(diffs_per_draw[0])],
            axis=2,
        )
        location_per_draw = wt_to_plot_for[:, : len(diffs_per_draw[0])]

        # Create 2D histogram with dot products as weights
        H, xedges, yedges = np.histogram2d(
            location_per_draw[0, :, 0],  # x coordinates from first chain
            location_per_draw[0, :, 1],  # y coordinates from first chain
            bins=[x_bins, y_bins],
            weights=dot_product_per_draw[0, :num_draws_to_plot_for],
            density=True,
        )
        plt.close("all")  # Clear any previous figures
        plt.figure(figsize=(10, 8))
        plt.imshow(
            H.T,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="auto",
            cmap="viridis",
        )
        plt.colorbar(label="Dot Product")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"{title}, i={i_to_plot_for}")
        plt.show()

    # Create interactive widget
    widget = widgets.interactive(
        update_plot, i_to_plot_for=i_slider, num_draws_to_plot_for=draws_slider
    )
    display(widget)


def plot_vector_field_with_colors(wts_flattened, num_bins=15, title="SGLD"):

    # Create sliders
    i_slider = widgets.IntSlider(
        value=100,
        min=10,
        max=2000,
        step=10,
        description="i_to_plot_for:",
        continuous_update=False,
    )

    diffs_slider = widgets.IntSlider(
        value=1, min=1, max=50, step=1, description="diff:", continuous_update=False
    )
    xmin = np.min(wts_flattened[:, :, 0])
    xmax = np.max(wts_flattened[:, :, 0])
    ymin = np.min(wts_flattened[:, :, 1])
    ymax = np.max(wts_flattened[:, :, 1])
    # Create the grid points for the plot
    x = np.linspace(xmin, xmax, num_bins)
    y = np.linspace(ymin, ymax, num_bins)

    # Create meshgrid for plotting
    Y, X = np.meshgrid(y, x)

    # Flatten the grid and the vectors for plotting
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    def update_plot(i_to_plot_for, diff_to_plot_for):
        diffs = np.concatenate(
            joint_ith_place_nth_diff(wts_flattened, i_to_plot_for, diff_to_plot_for)
        )
        vect_field, colors = compute_binned_averages_multi_with_std(
            diffs, xmin, xmax, ymin, ymax, num_bins
        )
        # Extract the vector components from A
        U = vect_field[:, :, 0]  # x-component of the vector field
        V = vect_field[:, :, 1]  # y-component of the vector field

        U_flat = U.flatten()
        V_flat = V.flatten()
        # Extract the values from C and flatten for coloring
        color_values = colors[:, :, 0].flatten()

        # Create the plot
        plt.close("all")  # Clear any previous figures
        plt.figure(figsize=(6, 6))

        # Use quiver to plot the vector field, with color based on the array C
        quiver = plt.quiver(
            X_flat, Y_flat, U_flat, V_flat, color_values, cmap="viridis"
        )

        # Add a colorbar to represent the color values from C
        plt.colorbar(quiver, label="Colored by standard deviations")

        # Set labels and plot title
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title + f" diff={diff_to_plot_for}, i={i_to_plot_for}")

        # Set limits for x and y axis
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.show()

    # Create interactive widget
    widget = widgets.interactive(
        update_plot, i_to_plot_for=i_slider, diff_to_plot_for=diffs_slider
    )
    display(widget)


def plot_combined_analysis(wt, gd, num_bins=15, title="Combined Analysis"):
    import ipywidgets as widgets
    from IPython.display import display

    # Create sliders
    i_slider = widgets.IntSlider(
        value=100,
        min=10,
        max=2000,
        step=10,
        description="i_to_plot_for:",
        continuous_update=False,
    )

    draws_slider = widgets.IntSlider(
        value=10000,
        min=100,
        max=10000,
        step=100,
        description="num_draws:",
        continuous_update=False,
    )

    diffs_slider = widgets.IntSlider(
        value=1, min=1, max=15, step=1, description="diff:", continuous_update=False
    )

    # Get the range for both dimensions
    x_min, x_max = np.min(wt[..., 0]), np.max(wt[..., 0])
    y_min, y_max = np.min(wt[..., 1]), np.max(wt[..., 1])

    # Create bins
    x_bins = np.linspace(x_min, x_max, num_bins)
    y_bins = np.linspace(y_min, y_max, num_bins)

    Y, X = np.meshgrid(y_bins, x_bins)
    X_flat, Y_flat = X.flatten(), Y.flatten()

    def update_plot(i_to_plot_for, num_draws_to_plot_for, diff_to_plot_for):
        if num_draws_to_plot_for < i_to_plot_for:
            print("num_draws_to_plot_for must be greater than i_to_plot_for")
            return

        # Left plot: Second order dot products
        wt_to_plot_for = wt[:, :num_draws_to_plot_for]
        gd_to_plot_for = gd[:, :num_draws_to_plot_for]
        diffs_per_draw = -ith_place_nth_diff(
            wt_to_plot_for, i_to_plot_for, diff_to_plot_for
        )
        dot_product_per_draw = np.sum(
            diffs_per_draw * gd_to_plot_for[:, : len(diffs_per_draw[0])], axis=2
        )
        location_per_draw = wt_to_plot_for[:, : len(diffs_per_draw[0])]

        H, _, _ = np.histogram2d(
            location_per_draw[0, :, 0],
            location_per_draw[0, :, 1],
            bins=[x_bins, y_bins],
            weights=dot_product_per_draw[0, :num_draws_to_plot_for],
            density=True,
        )

        diffs = np.concatenate(
            joint_ith_place_nth_diff(wt_to_plot_for, i_to_plot_for, diff_to_plot_for)
        )
        vect_field, colors = compute_binned_averages_multi_with_std(
            diffs, x_min, x_max, y_min, y_max, num_bins
        )

        U = vect_field[:, :, 0].flatten()  # x-component
        V = vect_field[:, :, 1].flatten()  # y-component
        color_values = colors[:, :, 0].flatten()

        plt.close("all")  # Clear any previous figures
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        im1 = ax1.imshow(
            H.T,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="auto",
            cmap="viridis",
        )
        plt.colorbar(im1, ax=ax1, label="Dot Product")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title(
            f"Sum of ({diff_to_plot_for}nd order diff) ⋅ grad over first {num_draws_to_plot_for} draws, i={i_to_plot_for}"
        )
        quiver = ax2.quiver(X_flat, Y_flat, U, V, color_values, cmap="viridis")
        plt.colorbar(quiver, ax=ax2, label="Standard deviations")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title(
            f"Vector Field of {diff_to_plot_for}nd order diff over first {num_draws_to_plot_for} draws,  i={i_to_plot_for}"
        )
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    # Create interactive widget
    widget = widgets.interactive(
        update_plot,
        i_to_plot_for=i_slider,
        num_draws_to_plot_for=draws_slider,
        diff_to_plot_for=diffs_slider,
    )
    display(widget)
