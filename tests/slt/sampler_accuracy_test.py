import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from devinterp.slt import sample
from devinterp.slt.llc import LLCEstimator
from devinterp.zoo.test_utils import *


@pytest.fixture
def generated_normalcrossing_dataset():
    torch.manual_seed(42)
    np.random.seed(42)
    sigma = 0.25
    num_train_samples = 1000
    batch_size = num_train_samples
    x = torch.normal(0, 2, size=(num_train_samples,))
    y = sigma * torch.normal(0, 1, size=(num_train_samples,))
    train_data = TensorDataset(x, y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader, train_data, x, y


# not a fixture as we're generating data for several m, n combinations
# and I couldn't figure out how to fit that into the fixture mold
def generated_rrr_dataset(m, n):
    torch.manual_seed(42)
    np.random.seed(42)
    num_samples = 1000
    batch_size = num_samples
    x = torch.randn(num_samples, m)
    y = torch.randn(num_samples, n)
    train_data = TensorDataset(x, y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader, train_data, x, y


TRUE_LCS_PER_POWER = {
    ((0, 1), 1 / 2),
    ((1, 1), 1 / 2),
    ((0, 2), 1 / 4),
    ((1, 2), 1 / 4),
    ((2, 2), 1 / 4),
    ((0, 3), 1 / 6),
    ((1, 3), 1 / 6),
    ((2, 3), 1 / 6),
    ((3, 3), 1 / 6),
}


@pytest.mark.parametrize("sampling_method", [SGLD, SGNHT])
@pytest.mark.parametrize("powers, true_lc", TRUE_LCS_PER_POWER)
def test_accuracy_normalcrossing(
    generated_normalcrossing_dataset, sampling_method, powers, true_lc
):
    seed = 42
    model = Polynomial(powers)
    train_loader, train_data, _, _ = generated_normalcrossing_dataset
    criterion = F.mse_loss
    lr = 0.0001
    num_chains = 10
    num_draws = 5_000
    llc_estimator = LLCEstimator(
        num_chains=num_chains, num_draws=num_draws, n=len(train_data)
    )
    sample(
        model,
        train_loader,
        criterion=criterion,
        optimizer_kwargs=dict(
            lr=lr,
            bounding_box_size=1.0,
            num_samples=len(train_data),
        ),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator],
        verbose=False,
        seed=seed,
    )
    llc_mean = llc_estimator.sample()["llc/mean"]
    llc_std_dev = llc_estimator.sample()["llc/std"]
    assert (
        llc_mean - 2 * llc_std_dev < true_lc < llc_mean + 2 * llc_std_dev
    ), f"LLC mean {llc_mean:.3f} +- {2*llc_std_dev:.3f} does not contain true value {true_lc:.3f} for powers {powers} using {sampling_method}"


@pytest.mark.parametrize("sampling_method", [SGLD, SGNHT])
@pytest.mark.parametrize(
    "m,h,n",
    [
        (5, 3, 5),  # case 1, odd
        (2, 1, 2),  # case 1, odd
        (5, 4, 5),  # case 1, even
        (3, 2, 3),  # case 1, even
        (4, 3, 8),  # case 2
        (2, 1, 4),  # case 2
        (8, 3, 4),  # case 3
        (4, 1, 2),  # case 3
        (3, 8, 4),  # case 4
        (1, 4, 2),  # case 4
    ],
)
def test_accuracy_rrr(sampling_method, m, h, n):
    # see "The Generalization Error of Reduced Rank Regression in Bayesian Estimation", M. Aoyagi & S. Watanabe, 2004.
    # Note: RRR is kind of an odd fit for pytorch, being a two-layer no-bias linear model.
    # We train this model long enough to (hopefully) not end up in a local min
    torch.manual_seed(42)
    np.random.seed(42)
    criterion = F.mse_loss
    train_loader, train_data, x, y = generated_rrr_dataset(m, n)
    #  m -> h (rank) -> n
    m = x.size(1)
    n = y.size(1)
    model = ReducedRankRegressor(m, h, n)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(5000):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    num_chains = 10
    num_draws = 1_000
    llc_estimator = LLCEstimator(
        num_chains=num_chains, num_draws=num_draws, n=len(train_data)
    )
    sample(
        model,
        train_loader,
        criterion=criterion,
        optimizer_kwargs=dict(
            lr=0.0005,
            bounding_box_size=2.0,
            num_samples=len(train_data),
        ),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator],
        verbose=False,
        seed=42,
    )
    llc_mean = llc_estimator.sample()["llc/mean"]
    llc_std_dev = llc_estimator.sample()["llc/std"]
    case_1_even = (m + h + n) % 2 == 0
    case_1_odd = (m + h + n) % 2 == 1
    case_2 = m + h < n
    case_3 = n + h < m
    case_4 = m + n < h
    if case_2:
        case = "2"
        true_lc = m * h / 2
    elif case_3:
        case = "3"
        true_lc = h * n / 2
    elif case_4:
        case = "4"
        true_lc = m * n / 2
    elif case_1_even:
        case = "1_even"
        true_lc = (2 * m * n + 2 * h * n + 2 * m * h - n**2 - m**2 - h**2) / 8
    elif case_1_odd:
        case = "1_odd"
        true_lc = (1 + 2 * m * n + 2 * h * n + 2 * m * h - n**2 - m**2 - h**2) / 8
    assert (
        llc_mean - 2 * llc_std_dev < true_lc < llc_mean + 2 * llc_std_dev
    ), f"DLN case {case} LLC mean {llc_mean:.3f} +- {2*llc_std_dev:.3f} does not contain true value {true_lc:.3f} for (M, H, N)={(m, h, n)} using {sampling_method}"
