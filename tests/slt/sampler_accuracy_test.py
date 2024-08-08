import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from devinterp.slt import sample
from devinterp.backends.default.slt.llc import LLCEstimator, OnlineLLCEstimator
from devinterp.test_utils import *
from devinterp.utils import evaluate_mse, optimal_nbeta


@pytest.fixture
def generated_normalcrossing_dataset():
    torch.manual_seed(42)
    np.random.seed(42)
    sigma = 0.25
    num_samples = 1000
    x = torch.normal(0, 2, size=(num_samples,))
    y = sigma * torch.normal(0, 1, size=(num_samples,))
    train_data = TensorDataset(x, y)
    train_dataloader = DataLoader(train_data, batch_size=num_samples, shuffle=True)
    return train_dataloader, train_data, x, y


TRUE_LCS_PER_POWER = [
    [[0, 1], 0.5],
    [[1, 1], 0.5],
    [[0, 2], 0.25],
    [[1, 2], 0.25],
    [[2, 2], 0.25],
    [[0, 3], 0.166],
    [[1, 3], 0.166],
    [[2, 3], 0.166],
    [[3, 3], 0.166],
]


@pytest.mark.parametrize("sampling_method", [SGLD, SGNHT])
@pytest.mark.parametrize("powers, true_lc", TRUE_LCS_PER_POWER)
def test_accuracy_normalcrossing(
    generated_normalcrossing_dataset, sampling_method, powers, true_lc
):
    seed = 42
    model = Polynomial(powers)
    train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
    lr = 0.0002
    num_chains = 10
    num_draws = 5_000
    llc_estimator = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=optimal_nbeta(train_dataloader),
    )
    sample(
        model,
        train_dataloader,
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(lr=lr, bounding_box_size=1.0),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator],
        verbose=False,
        seed=seed,
    )
    llc_mean = llc_estimator.get_results()["llc/mean"]
    llc_std_dev = llc_estimator.get_results()["llc/std"]
    assert (
        llc_mean - 2 * llc_std_dev < true_lc < llc_mean + 2 * llc_std_dev
    ), f"LLC mean {llc_mean:.3f} +- {2*llc_std_dev:.3f} does not contain true value {true_lc:.3f} for powers {powers} using {sampling_method}"
