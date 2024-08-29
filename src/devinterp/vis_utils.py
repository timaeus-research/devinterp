import typing
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Container, List, Optional, Type, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm, trange

from devinterp.slt.mala import MalaAcceptanceRate
from devinterp.utils import default_nbeta


# Sampling config validates input parameters while allowing us to use **kwargs later on
class SweepConfig(BaseModel):
    epsilon_range: List[float]
    beta_range: List[float]
    llc_estimator: Callable
    llc_estimator_kwargs: dict

    # Pydantic-recognized field for custom settings
    class Config:
        arbitrary_types_allowed = True  # Allows Pydantic to accept pytorch models

    # Build epsilon_range and beta_range given different user input formats for beta and epsilon ranges
    @classmethod
    def setup(
        cls,
        llc_estimator,
        llc_estimator_kwargs,
        min_beta,
        max_beta,
        beta_samples,
        beta_range,
        min_epsilon,
        max_epsilon,
        epsilon_samples,
        epsilon_range,
        dataloader=None,
    ):
        if epsilon_range is not None:
            assert isinstance(
                epsilon_range, Sequence
            ), "epsilon_range must be a list-like object (e.g list or numpy array)"
            if min_epsilon is not None or max_epsilon is not None:
                warnings.warn(
                    "min_epsilon and max_epsilon will be ignored as epsilon_range is provided"
                )
        else:
            epsilon_range = np.power(
                10,
                np.linspace(
                    np.log10(min_epsilon), np.log10(max_epsilon), epsilon_samples
                ),
            )

        if beta_range is not None:
            assert isinstance(
                beta_range, Sequence
            ), "beta_range must be a list-like object (e.g list or numpy array)"
            if min_beta is not None or max_beta is not None:
                warnings.warn(
                    "min_beta and max_beta will be ignored as beta_range is provided"
                )
        else:
            if dataloader is not None:
                # Calculate default beta (inverse temperature) range.
                default_beta = default_nbeta(dataloader)
                if min_beta is None:
                    min_beta = 1e-2 * default_beta
                if max_beta is None:
                    max_beta = 1e3 * default_beta
            else:
                if min_beta is None or max_beta is None:
                    raise ValueError(
                        "min_beta and max_beta must be provided if dataloader is not provided."
                    )
            beta_range = np.power(
                10, np.linspace(np.log10(min_beta), np.log10(max_beta), beta_samples)
            )

        assert min(beta_range) > 0, "All beta values must be greater than 0"
        assert min(epsilon_range) > 0, "All epsilon values must be greater than 0"
        if max(epsilon_range) > 1e-2:
            warnings.warn(
                "Epsilon values greater than 1e-2 typically lead to instability in the sampling process. Consider reducing epsilon to between 1e-6 and 1e-2."
            )

        return cls(
            epsilon_range=epsilon_range,
            beta_range=beta_range,
            llc_estimator=llc_estimator,
            llc_estimator_kwargs=llc_estimator_kwargs,
        )


class EpsilonBetaAnalyzer:
    """
    A class for analyzing and visualizing the local learning coefficient (LLC) across different epsilon and beta values.

    Includes methods to configure, run, and visualize sweeps of the local learning coefficient over epsilon and beta.
    """

    def __init__(self):
        self.sweep_config = None
        self.plotting_config = None
        self.sweep_df = None
        self.fig = None

    def configure_sweep(
        self,
        llc_estimator: Callable,
        llc_estimator_kwargs: dict,
        min_epsilon: Optional[float] = 1e-6,
        max_epsilon: Optional[float] = 1e-2,
        epsilon_samples: float = 8,
        epsilon_range: Optional[List[float]] = None,
        min_beta: Optional[float] = None,
        max_beta: Optional[float] = None,
        beta_samples: float = 8,
        beta_range: Optional[List[float]] = None,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> None:
        """
        Configure the sampling parameters for the LLC analysis.
        :param llc_estimator: Callable function to estimate the local learning coefficient.
            Note: The estimator function expected by EpsilonBetaAnalyzer must have the following signature:
            def estimator(epsilon: float, beta: float, **kwargs) -> dict
            where kwargs are the arguments to estimate_learning_coeff_with_summary
            The return value must be a dict with a "llc/trace" key corresponding to a numpy array of shape (num_chains, num_draws)
            Additional keys can represent other values of interest (e.g. acceptance rates, true LLC.)
        :param llc_estimator_kwargs: Keyword arguments for the llc_estimator function.
        :param min_epsilon: Minimum value for epsilon range (if epsilon_range not provided).
        :param max_epsilon: Maximum value for epsilon range (if epsilon_range not provided).
        :param epsilon_samples: Number of samples in epsilon range (if epsilon_range not provided).
        :param epsilon_range: Explicit range of epsilon values to use (overrides min/max_epsilon).
        :param min_beta: Minimum value for beta range (if beta_range not provided).
        :param max_beta: Maximum value for beta range (if beta_range not provided).
        :param beta_samples: Number of samples in beta range (if beta_range not provided).
        :param beta_range: Explicit range of beta values to use (overrides min/max_beta).
        :param dataloader: Optional dataloader for calculating optimal beta.
        """

        self.sweep_config = SweepConfig.setup(
            llc_estimator,
            llc_estimator_kwargs,
            min_beta,
            max_beta,
            beta_samples,
            beta_range,
            min_epsilon,
            max_epsilon,
            epsilon_samples,
            epsilon_range,
            dataloader,
        )

    def sweep(self, add_to_existing=False) -> None:
        """
        Perform the LLC sweep using the configured parameters.

        This method runs the LLC estimator for each combination of epsilon and beta values
        and stores the results in self.sweep_df.

        :param add_to_existing: If True, adds new sweep results to existing ones. If False, replaces existing results.
            Useful for sweeping over multiple models or datasets.
        """
        assert (
            self.sweep_config is not None
        ), "Sweep configuration is not set. Please call configure_sweep() first."

        epsilon_range = self.sweep_config.epsilon_range
        beta_range = self.sweep_config.beta_range
        llc_estimator = self.sweep_config.llc_estimator
        llc_estimator_kwargs = self.sweep_config.llc_estimator_kwargs

        if "device" in llc_estimator_kwargs:
            if torch.cuda.is_available() and (
                llc_estimator_kwargs["device"] == "cpu"
                or torch.device(llc_estimator_kwargs["device"]).type == "cpu"
            ):
                warnings.warn(
                    "CUDA is available but not being used. Consider setting device='cuda' for faster computation."
                )

        all_sweep_stats = []
        with tqdm(total=len(epsilon_range) * len(beta_range)) as pbar:
            for epsilon in epsilon_range:
                for beta in beta_range:
                    try:
                        sweep_stats = llc_estimator(
                            epsilon=epsilon, beta=beta, **llc_estimator_kwargs
                        )
                        sweep_stats = dict(sweep_stats, epsilon=epsilon, beta=beta)
                        all_sweep_stats.append(sweep_stats)
                    except RuntimeError as e:
                        warnings.warn(
                            f"Error encountered for epsilon={epsilon}, beta={beta}. Skipping. Warning: {e}"
                        )
                    pbar.update(1)

        sweep_df = pd.DataFrame(all_sweep_stats)
        # If there's only one sweep, there'll only be one trace, so we need to add an extra dimension
        sweep_df["llc/trace"] = sweep_df["llc/trace"].apply(
            lambda x: x if len(x.shape) == 2 else x[np.newaxis, :]
        )
        if add_to_existing:
            if self.sweep_df is not None:
                self.sweep_df = pd.concat([self.sweep_df, sweep_df], ignore_index=True)
            else:
                self.sweep_df = sweep_df
        else:
            self.sweep_df = sweep_df

    def plot(
        self,
        true_lambda: Union[float, int, str, Container] = None,
        num_last_steps_to_average: int = 50,
        color: Optional[str] = None,
        slider: Optional[str] = None,
        slider_plane: Optional[str] = False,
        **kwargs,
    ) -> go.Figure:
        """
        Plot the results of the LLC sweep.

        :param true_lambda: True value of lambda for comparison (optional). Can be a scalar, a list, or a string column name of sweep_df.
            Will plot a horizontal plane at the true_lambda value.
        :param num_last_steps_to_average: Number of last steps to average for final LLC value.
        :param color: Column name to use for coloring the scatter points.
        :param slider: Column name to use for creating a slider in the plot.
        :param slider_plane: If True, adds a plane for each slider value.
        :param kwargs: Additional keyword arguments to pass to the plotting function.
            Example: range_color=[0, 0.15] to set the color range.
        :return: A plotly Figure object containing the LLC sweep visualization.
        """
        plot_config = {
            "title": "Local learning coefficient vs. epsilon and beta",
            "z": "llc/final",
            "log_y": True,
            "log_x": True,
            "log_z": True,
        }

        assert (
            self.sweep_df is not None
        ), "No data to plot. Please call get_results() first."

        sweep_df = self.sweep_df.copy()
        # Calculate additional statistics
        sweep_df["llc/std_over_mean"] = sweep_df["llc/trace"].apply(
            lambda x: x[:, -num_last_steps_to_average:].std()
            / x[:, -num_last_steps_to_average:].mean()
        )
        sweep_df["llc/final"] = sweep_df["llc/trace"].apply(
            lambda x: x[:, -num_last_steps_to_average].mean()
        )

        if true_lambda is not None:
            if type(true_lambda) in [int, float]:
                sweep_df["true_lambda"] = sweep_df["llc/trace"].apply(
                    lambda x: true_lambda
                )
            elif type(true_lambda) == str:
                sweep_df["true_lambda"] = sweep_df[true_lambda]
            else:
                sweep_df["true_lambda"] = true_lambda
            sweep_df["log_true_lambda"] = sweep_df["true_lambda"].apply(
                lambda x: np.log10(np.abs(x))
            )
            sweep_df["log_lambda_hat"] = sweep_df["llc/final"].apply(
                lambda x: np.log10(np.abs(x))
            )
            sweep_df["log_delta_lambda"] = (
                sweep_df["log_lambda_hat"] - sweep_df["log_true_lambda"]
            )
            sweep_df["lambda_delta"] = sweep_df["llc/final"] - sweep_df["true_lambda"]
            sweep_df["log_lambda_delta"] = sweep_df["lambda_delta"].apply(
                lambda x: np.log10(np.abs(x)) * np.sign(x)
            )
            if color is None:
                color = "log_lambda_delta"  # default color when true_lambda is provided
                plot_config["range_color"] = [-4, 4]

        if color is None:
            color = (
                "llc/std_over_mean"  # default color when true_lambda is not provided
            )

        if color == "llc/std_over_mean":
            plot_config["range_color"] = [0, 0.15]

        # Add any additional kwargs to the plot_config
        plot_config.update(kwargs)

        if slider is None:
            fig = px.scatter_3d(
                sweep_df, x="epsilon", y="beta", color=color, **plot_config
            )

            if true_lambda is not None:
                # Grid (to easily plot horizontal planes)
                epsilon_range = self.sweep_config.epsilon_range
                beta_range = self.sweep_config.beta_range
                broadcast_grid = np.ones((len(epsilon_range), len(beta_range)))
                # Place a horizontal plane at height true_lambda
                plane = go.Surface(
                    x=epsilon_range,
                    y=beta_range,
                    z=true_lambda * broadcast_grid,
                    opacity=0.4,
                    surfacecolor=true_lambda * broadcast_grid,
                    showscale=False,
                    name=f"RLCT={true_lambda}",
                )
                fig.add_trace(plane)

        else:
            # Create base figure.
            fig = None

            # Determine fixed ranges for axes and colorbar
            x_range = [
                sweep_df["epsilon"].min() / 2,
                sweep_df["epsilon"].max() / 1.5,
            ]  # Add some margin
            y_range = [sweep_df["beta"].min(), sweep_df["beta"].max()]
            z_range = [
                max(1e-2, sweep_df["llc/final"].min()),
                sweep_df["llc/final"].max(),
            ]
            color_range = [sweep_df[color].min(), sweep_df[color].max()]
            plot_config["range_color"] = color_range
            if color == "llc/std_over_mean":
                plot_config["range_color"] = [0, 0.15]

            plot_config["range_z"] = z_range
            plot_config["range_x"] = x_range
            plot_config["range_y"] = y_range

            unique_slider_vals = sweep_df[slider].unique()
            unique_slider_vals = sorted(unique_slider_vals)
            # Add traces for each unique slider value
            for slider_val in unique_slider_vals:
                df_filtered = sweep_df[sweep_df[slider] == slider_val]
                plot_ = px.scatter_3d(
                    df_filtered, x="epsilon", y="beta", color=color, **plot_config
                )
                if fig is None:
                    fig = plot_
                else:
                    trace = plot_.data[0]
                    fig.add_trace(trace)

                # Grid (to easily plot horizontal planes)
                epsilon_range = self.sweep_config.epsilon_range
                beta_range = self.sweep_config.beta_range
                broadcast_grid = np.ones((len(epsilon_range), len(beta_range)))

                if slider_plane:
                    # Place a horizontal plane with height = slider_val
                    plane = go.Surface(
                        x=epsilon_range,
                        y=beta_range,
                        z=slider_val * broadcast_grid,
                        opacity=0.4,
                        surfacecolor=slider_val * broadcast_grid,
                        showscale=False,
                        name=f"{slider}={slider_val}",
                    )
                    fig.add_trace(plane)

            # Slider
            steps = []
            for i, slider_val in enumerate(unique_slider_vals):
                step = dict(
                    method="update",
                    args=[
                        {"visible": [False] * len(fig.data)},
                        {
                            "title": f"Local learning coefficient vs. epsilon and beta ({slider} = {slider_val})"
                        },
                    ],
                    label=str(slider_val),
                )

                if slider_plane:
                    step["args"][0]["visible"][
                        2 * i
                    ] = True  # Toggle i'th scatter trace to "visible"
                    step["args"][0]["visible"][
                        2 * i + 1
                    ] = True  # Toggle i'th plane trace to "visible"
                else:
                    step["args"][0]["visible"][i] = True
                steps.append(step)

            sliders = [
                dict(
                    active=0,
                    currentvalue={"prefix": f"{slider}: "},
                    pad={"t": 50},
                    steps=steps,
                )
            ]

            # Axes and layout
            fig.update_layout(
                scene=dict(
                    xaxis_title="epsilon",
                    yaxis_title="beta",
                    zaxis_title="lambdahat",
                    xaxis_type="log",
                    yaxis_type="log",
                    zaxis_type="log",
                    xaxis_range=np.log10(x_range),
                    yaxis_range=np.log10(y_range),
                    zaxis_range=np.log10(z_range),
                    aspectmode="manual",
                    aspectratio=dict(x=0.7, y=1.2, z=1),
                ),
                sliders=sliders,
                title="Local learning coefficient vs. epsilon and beta",
            )

        self.fig = fig
        return fig

    def save_fig(self, path: str) -> None:
        """
        Save the current figure to a file.
        :param path: Path to save the figure to.
        """
        assert self.fig is not None, "No figure to save. Please call plot() first."
        self.fig.write_html(path)
