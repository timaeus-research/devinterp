from copy import deepcopy

from .diffs import ith_place_nth_diff, joint_ith_place_nth_diff
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from PyMoments import kstat


# plot stats for each dim (flattened)
# todo rename, this is second order but 1-place stats
def plot_second_order_one_place_stats(wt, n, method="zeros"):
    wt = np.array([wt[0]])  # TODO average across the three chains
    num_chains = len(wt)
    num_draws = len(wt[0])
    all_stds_per_i = []
    all_mean_dists_per_n = []
    i_values = np.arange(1, num_draws // n - 2, 1)
    for i in i_values:
        diffs = ith_place_nth_diff(wt, i, n)
        std_per_i = np.std(diffs, axis=1)[0]
        print
        all_stds_per_i.append(std_per_i)
    fig, ax1 = plt.subplots()  # Create a figure and a primary axe
    for tensor_index, std in enumerate(np.array(all_stds_per_i).T):
        ax1.plot(i_values, std, label=f"std {tensor_index}")
    plt.title(method)
    plt.show()


# todo rename this is 2-place stats
def plot_second_order_two_place_stats(wt, n, method="zeros"):
    num_chains = len(wt)
    num_draws = len(wt[0])
    all_stds_x_per_i = []
    all_stds_y_per_i = []
    cov_x_y_per_i = []
    i_values = np.arange(1, num_draws // n - 2, 1)
    for i in i_values:
        diffs = ith_place_nth_diff(wt, i, n)
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

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # Plot data on the secondary y-axis
    # ax2.plot(i_values, all_mean_dists_per_n, label='dist', color='r') # should be secondary_y

    # Add legends for both axes
    ax1.legend(loc="upper left")
    # ax2.legend(loc='upper right')

    plt.show()


def plot_third_order_stats(wt, n, method="zeros"):
    num_chains = len(wt)
    num_draws = len(wt[0])
    diff_third_cum_xxx_per_i = []
    diff_third_cum_yyy_per_i = []
    diff_third_cum_xxy_per_i = []
    diff_third_cum_xyy_per_i = []
    i_values = np.arange(1, num_draws - 3, 1)
    for i in i_values:
        diffs = ith_place_nth_diff(wt, i, n)
        diffs = diffs.reshape(-1, diffs.shape[-1])
        diff_third_cum_xxx_per_i.append(kstat(diffs, (0, 0, 0)))
        diff_third_cum_yyy_per_i.append(kstat(diffs, (1, 1, 1)))
        diff_third_cum_xxy_per_i.append(kstat(diffs, (0, 0, 1)))
        diff_third_cum_xyy_per_i.append(kstat(diffs, (0, 1, 1)))
    fig, ax1 = plt.subplots()  # Create a figure and a primary axes


def plot_third_order_stats_per_dim(wt, n, method="zeros"):
    num_chains = len(wt)
    num_draws = len(wt[0])
    diff_third_cum_xxx_per_i = []
    diff_third_cum_yyy_per_i = []
    diff_third_cum_zzz_per_i = []
    diff_third_cum_aaa_per_i = []
    i_values = np.arange(1, num_draws - 3, 1)
    for i in i_values:
        diffs = ith_place_nth_diff(wt, i, n)
        diffs = diffs.reshape(-1, diffs.shape[-1])
        diff_third_cum_xxx_per_i.append(kstat(diffs, (0, 0, 0)))
        diff_third_cum_yyy_per_i.append(kstat(diffs, (1, 1, 1)))
        diff_third_cum_zzz_per_i.append(kstat(diffs, (2, 2, 2)))
        diff_third_cum_aaa_per_i.append(kstat(diffs, (3, 3, 3)))

    fig, ax1 = plt.subplots()  # Create a figure and a primary axes

    # Plot data on the primary y-axis
    ax1.plot(i_values, diff_third_cum_xxx_per_i, label="xxx")
    ax1.plot(i_values, diff_third_cum_yyy_per_i, label="yyy")
    ax1.plot(i_values, diff_third_cum_zzz_per_i, label="zzz")
    ax1.plot(i_values, diff_third_cum_aaa_per_i, label="aaa")

    # Add legend
    ax1.legend(loc="upper left")

    plt.show()


def plot_trajectories(weight_trajectories, names, model, n_bins):
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
                if w1_range[0] <= weight[0] <= w1_range[-1]
                and w2_range[0] <= weight[1] <= w2_range[-1]
            ]
        )
        sns.scatterplot(
            x=draws_array[:, 0], y=draws_array[:, 1], marker="x", ax=axes[i], s=10
        )
        axes[i].axhline(0, linestyle="--", color="gray")
        axes[i].axvline(0, linestyle="--", color="gray")
        axes[i].set_xlabel(r"$w_{1}$")
        axes[i].set_ylabel(r"$w_{2}$")
        axes[i].set_title(names[i])
        axes[i].grid(False)
    plt.show()


def plot_multi_trajectories(wt, i_range, diffs_range, legend):
    num_chains = len(wt)
    for n in diffs_range:
        for i in i_range:
            diffs = ith_place_nth_diff(wt, i, n)
            plot_trajectories(
                diffs,
                names=[legend + f"_{chain} n={n} i={i}" for chain in range(num_chains)],
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


import numpy as np


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


def plot_vector_field(A, xmin, xmax, ymin, ymax):
    n = A.shape[0]  # Assuming A is of shape (n, n, 2)

    # Create the grid points for the plot
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    # Extract the vector components from A
    U = A[:, :, 0]  # x-component of the vector field
    V = A[:, :, 1]  # y-component of the vector field

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, U, V)

    # Set labels and plot title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Vector Field")

    # Set limits for x and y axis
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.gca().set_aspect("equal", adjustable="box")  # Equal scaling for axes
    plt.show()


def plot_vector_field_with_colors(A, C, xmin, xmax, ymin, ymax, i, n, legend):
    num_bins = A.shape[0]

    # Create the grid points for the plot
    x = np.linspace(xmin, xmax, num_bins)
    y = np.linspace(ymin, ymax, num_bins)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    # Extract the vector components from A
    U = A[:, :, 0]  # x-component of the vector field
    V = A[:, :, 1]  # y-component of the vector field

    # Flatten the grid and the vectors for plotting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    U_flat = U.flatten()
    V_flat = V.flatten()

    # Extract the values from C and flatten for coloring
    color_values = C[:, :].flatten()

    # Create the plot
    plt.figure(figsize=(6, 6))

    # Use quiver to plot the vector field, with color based on the array C
    quiver = plt.quiver(X_flat, Y_flat, U_flat, V_flat, color_values, cmap="viridis")

    # Add a colorbar to represent the color values from C
    plt.colorbar(quiver, label="Colored by standard deviations")

    # Set labels and plot title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(legend + f" n={n}, i={i}")

    # Set limits for x and y axis
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)


def plot_vector_field_jets(wt, i_range, diffs_range, num_bins, legend):
    num_chains = len(wt)
    xmin = np.min(wt[:, :, 0])
    xmax = np.max(wt[:, :, 0])
    ymin = np.min(wt[:, :, 1])
    ymax = np.max(wt[:, :, 1])
    for n in diffs_range:
        for i in i_range:
            diffs = np.concatenate(joint_ith_place_nth_diff(wt, i, n))
            vect_field, colors = compute_binned_averages_multi_with_std(
                diffs, xmin, xmax, ymin, ymax, num_bins
            )
            plot_vector_field_with_colors(
                vect_field, colors[:, :, 0], xmin, xmax, ymin, ymax, i, n, legend
            )
