import numpy as np
import pytest
import torch
from devinterp.slt.mala import mala_acceptance_probability


MALA_CALC_TESTCASES = [
    [[0.0, 0.0], [0.0, 0.0], [0.0], [1.5, 5.5], [1.5, 5.5], [16.25], 0.1, 0.6661436],
    [[1.0, 1.0], [1.0, 1.0], 1.0, [1.5, 0.5], [1.5, 0.5], 1.25, 0.5, 0.9692332],
    [[0.0, 0.0], [0.0, 0.0], 0.0, [10.5, 5.5], [10.5, 5.5], 70.25, 0.1, 0.17268492],
    [[0.0, 0.0], [0.0, 0.0], 0.0, [10.5, 5.5], [10.5, 5.5], 70.25, 0.5, 0.00015359],
]


@pytest.mark.parametrize(
    "prev_point,prev_grad,prev_loss,current_point,current_grad,current_loss,learning_rate,benchmark_accept_prob",
    MALA_CALC_TESTCASES,
)
def test_mala_calc(
    prev_point,
    prev_grad,
    prev_loss,
    current_point,
    current_grad,
    current_loss,
    learning_rate,
    benchmark_accept_prob,
):
    mala_accept_prob = mala_acceptance_probability(
        torch.tensor(prev_point),
        torch.tensor(prev_grad),
        torch.tensor(prev_loss),
        torch.tensor(current_point),
        torch.tensor(current_grad),
        torch.tensor(current_loss),
        torch.tensor(learning_rate),
    )
    assert np.isclose(mala_accept_prob, benchmark_accept_prob, atol=0.000001), (
        f"MALA accept prob {mala_accept_prob}, not close to benchmark value {benchmark_accept_prob:.2f}"
    )
