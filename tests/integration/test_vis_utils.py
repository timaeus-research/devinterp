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

            assert fig is None  # Plot should return None when Plotly is unavailable

            # Check that a warning was issued
            mock_warn.warn.assert_called_with(
                "Plotting is unavailable because Plotly is not installed. Install with `pip install devinterp[vis]` to enable visualization."
            )


def test_plot_with_plotly():
    # This test assumes Plotly is installed
    from devinterp.vis_utils import EpsilonBetaAnalyzer

    analyzer = EpsilonBetaAnalyzer()

    # Mock data for plotting
    analyzer.sweep_config = mock.Mock()
    analyzer.sweep_config.epsilon_range = [1e-6, 1e-5]
    analyzer.sweep_config.beta_range = [1e-2, 1e-1]
    analyzer.beta_range = [1e-2, 1e-1]

    # with plotly plotting should raise an AssertionError (no data), which is past the early plotly return
    with pytest.raises(AssertionError):
        analyzer.plot()
