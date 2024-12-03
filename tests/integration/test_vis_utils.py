import sys
from unittest import mock

import pytest


# Importing EpsilonBetaAnalyzer within a mock context to simulate Plotly absence
def test_plot_without_plotly():
    with mock.patch.dict(
        sys.modules, {"plotly.express": None, "plotly.graph_objects": None}
    ):
        with mock.patch("devinterp.vis_utils.warnings") as mock_warn:
            from devinterp.vis_utils import EpsilonBetaAnalyzer

            analyzer = EpsilonBetaAnalyzer()
            fig = analyzer.plot()

            # Assert that Plotly was not imported
            assert (
                analyzer.sweep_config is not None
            )  # Ensuring other functionalities work
            assert fig is None  # Plot should return None when Plotly is unavailable

            # Check that a warning was issued
            mock_warn.warn.assert_called_with(
                "Plotting is unavailable because Plotly is not installed. "
                "Install devinterp[vis] to enable visualization."
            )


def test_plot_with_plotly():
    # This test assumes Plotly is installed and ensures that plotting works as expected
    from devinterp.vis_utils import EpsilonBetaAnalyzer

    analyzer = EpsilonBetaAnalyzer()

    # Mock data for plotting
    analyzer.sweep_config = mock.Mock()
    analyzer.sweep_config.epsilon_range = [1e-6, 1e-5]
    analyzer.sweep_config.beta_range = [1e-2, 1e-1]
    analyzer.beta_range = [1e-2, 1e-1]

    # Since actual plotting would require more setup, we'll focus on ensuring no exceptions are raised
    try:
        fig = analyzer.plot()
        assert fig is not None  # A Plotly figure should be returned
    except Exception as e:
        pytest.fail(f"Plotting raised an exception unexpectedly: {e}")
