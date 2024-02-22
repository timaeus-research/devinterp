import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import warnings

from devinterp.slt.pca import *


def get_osculating_circle(curve, t_index):
    # Handle edge cases
    if t_index == 0:
        t_index = 1
    if t_index == len(curve) - 1:
        t_index = len(curve) - 2

    # Central differences for first and second derivatives
    r_prime = (curve[t_index + 1] - curve[t_index - 1]) / 2
    r_double_prime = curve[t_index + 1] - 2 * curve[t_index] + curve[t_index - 1]

    # Append a zero for 3D cross product
    r_prime_3d = np.append(r_prime, [0])
    r_double_prime_3d = np.append(r_double_prime, [0])

    # Curvature calculation and normal vector direction
    cross_product = np.cross(r_prime_3d, r_double_prime_3d)
    curvature = np.linalg.norm(cross_product) / np.linalg.norm(r_prime) ** 3
    signed_curvature = np.sign(cross_product[2])  # Sign of z-component of cross product
    radius_of_curvature = 1 / (curvature + 1e-12)

    # Unit tangent vector
    tangent = r_prime / np.linalg.norm(r_prime)

    # Unit normal vector, direction depends on the sign of the curvature
    if signed_curvature >= 0:
        norm_perp = np.array(
            [-tangent[1], tangent[0]]
        )  # Rotate tangent by 90 degrees counter-clockwise
    else:
        norm_perp = np.array(
            [tangent[1], -tangent[0]]
        )  # Rotate tangent by 90 degrees clockwise

    # Center of the osculating circle
    center = curve[t_index] + radius_of_curvature * norm_perp

    return center, radius_of_curvature


def get_osculate_plot_data(osculating_circles, skip, num_sharp_points, num_vertices):
    dcenters = []
    radii = []
    circles = []
    prev_center = None
    for center, radius in osculating_circles[::skip]:
        if prev_center is not None:
            dcenters.append(np.linalg.norm(center - prev_center))
        else:
            dcenters.append(1000)
        radii.append(radius)
        circles.append(
            plt.Circle(center, radius, alpha=0.5, color="lightgray", lw=0.5, fill=False)
        )
        prev_center = center
    # Sharp turn = small radius, so sort lo->hi and take first
    top_n_smallest_radii = np.argsort(radii)[:num_sharp_points]
    # Big cusp = High dcenter = big change, so sort lo->hi and take last
    top_n_sharpest_cusps = np.argsort(dcenters)[-num_vertices:]
    # Convert from index w/ skips to index w/o
    top_n_smallest_radii *= skip
    top_n_sharpest_cusps *= skip
    return top_n_sharpest_cusps, top_n_smallest_radii, circles


def plot_essential_dynamics_grid(
    samples,
    transitions=[],
    colors=[],
    marked_cusp_data=[],
    num_pca_components=3,
    plot_caustic=True,
    plot_vertex_influence=True,
    figsize=(20, 6),
    num_sharp_points=20,
    num_vertices=35,
    osculate_start=1,
    osculate_end_offset=0,
    osculate_skip=8,
    early_smoothing=10,
    late_smoothing=60,
    late_smoothing_from=200,
):
    OSCULATE_START = osculate_start
    OSCULATE_END = len(samples) - osculate_end_offset
    OSCULATE_SKIP = osculate_skip

    if len(colors) != len(transitions) != 0:
        warnings.warn("len(colors) != len(transitions), using rainbow palette.")
        cm = plt.get_cmap("gist_rainbow")
        colors = [cm(1.0 * i / len(transitions)) for i in range(len(transitions))]
    if OSCULATE_END - OSCULATE_START < 10:
        warnings.warn("< 10 data points for osculates, this will not be a useful plot.")
    if plot_vertex_influence and not len(marked_cusp_data):
        warnings.warn("Can't plot vertex influence when cusp data not provided")
    print(f"Number of samples: {len(samples)}")

    # Make sure we have the smoothed data for each PC
    pc_combo_indices = combinations(range(num_pca_components), 2)
    smoothed_pc_combos = combinations(
        get_smoothed_pcs(
            samples,
            num_pca_components,
            early_smoothing,
            late_smoothing,
            late_smoothing_from,
        ),
        2,
    )
    num_subplots = num_pca_components * (num_pca_components - 1) // 2
    fig, axes = plt.subplots(1, num_subplots, figsize=figsize)

    for (
        ax,
        (second_pc_index, first_pc_index),
        (second_smoothed_pc, first_smoothed_pc),
    ) in zip(axes, pc_combo_indices, smoothed_pc_combos):

        # For each PC pair we first get the osculating circles, and plot those
        print(f"Calculating osculates of PC{second_pc_index+1} vs PC{first_pc_index+1}")
        osculating_circles = [
            get_osculating_circle(
                np.column_stack((first_smoothed_pc, second_smoothed_pc)), z
            )
            for z in range(OSCULATE_START, OSCULATE_END)
        ]
        # Get all circles, highest curvature points & caustic cusps
        dcenter_indices, radius_indices, circles_to_plot = get_osculate_plot_data(
            osculating_circles, OSCULATE_SKIP, num_sharp_points, num_vertices
        )
        print(f"Plotting")
        for circle in circles_to_plot:
            ax.add_artist(circle)
        for i in radius_indices:
            ax.scatter(first_smoothed_pc[i], second_smoothed_pc[i], color="red")
        for i in dcenter_indices:
            ax.scatter(first_smoothed_pc[i], second_smoothed_pc[i], color="gold")

        # Draw the evolute
        if plot_caustic:
            x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
            for (center_x, center_y), _ in osculating_circles:
                if (x_lim[0] < center_x < x_lim[1]) and (
                    y_lim[0] < center_y < y_lim[1]
                ):
                    ax.scatter(center_x, center_y, color="black", s=0.5)
        # Plot marked cusps & vertex influence, if supplied & set (resp.)
        for marked_cusp in marked_cusp_data:
            marked_cusp_idx = marked_cusp["idx"]
            ax.scatter(
                first_smoothed_pc[marked_cusp_idx],
                second_smoothed_pc[marked_cusp_idx],
                color="green",
                marker="x",
                s=40,
            )
            center, _ = osculating_circles[marked_cusp_idx]
            ax.scatter(*center, color="green", marker="x", s=60)

            if plot_vertex_influence:
                vertex_influence_start = marked_cusp["influence_start"]
                vertex_influence_end = marked_cusp["influence_end"]
                ax.scatter(
                    first_smoothed_pc[vertex_influence_start],
                    second_smoothed_pc[vertex_influence_start],
                    color="blue",
                    marker="x",
                    s=40,
                )
                ax.scatter(
                    first_smoothed_pc[vertex_influence_end],
                    second_smoothed_pc[vertex_influence_end],
                    color="blue",
                    marker="x",
                    s=40,
                )
        # Plot un-smoothed points in the background
        ax.scatter(
            x=samples[OSCULATE_START:OSCULATE_END, first_pc_index],
            y=samples[OSCULATE_START:OSCULATE_END, second_pc_index],
            alpha=0.5,
            color="lightgray",
            s=10,
        )
        # Plot transitions & transition color legend
        if transitions:
            legend_ax = fig.add_axes([0.1, -0.03, 0.95, 0.05])
            handles = [
                plt.Line2D([0], [0], color=color, linestyle="-") for color in colors
            ]
            labels = [transition[2] for transition in transitions]
            legend_ax.legend(
                handles, labels, loc="center", ncol=len(labels), frameon=False
            )
            legend_ax.axis("off")

            for color, (start, end, _) in zip(colors, transitions):
                ax.plot(
                    first_smoothed_pc[start:end],
                    second_smoothed_pc[start:end],
                    color=color,
                    lw=2,
                )
        ax.set_xlabel(f"PC {first_pc_index+1}")
        ax.set_ylabel(f"PC {second_pc_index+1}")
    # plot_explained_variance(pca, title="Explained Variance", axes[-1])
    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.set_facecolor("white")
    return fig
