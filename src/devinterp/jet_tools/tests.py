from .diffs import ith_place_first_diff
import numpy as np


def test_ith_place_first_diff():
    test_array = np.array([[[0.0], [1.0], [3.0]]])
    should_be = np.array([1.0, 2.0])
    assert np.array_equal(
        ith_place_first_diff(test_array, 1).reshape(-1), should_be.reshape(-1)
    )
