import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from devinterp.slt import sample
from devinterp.slt.llc import LLCEstimator
from devinterp.test_utils import *
from devinterp.utils import *


def make_pop_loss_fn(true_model):
    assert true_model.m == true_model.n
    d = true_model.m
    true_A, true_B = (
        true_model.fc1.weight.detach().clone(),
        true_model.fc2.weight.detach().clone(),
    )
    true_prod = true_B @ true_A

    def loss_fn(model):
        Q = true_prod - (model.fc2.weight @ model.fc1.weight)
        loss = ((d / (d + 2)) * (torch.sum(Q * Q) / d)) / d
        return loss

    return loss_fn


def make_emp_loss_fn(true_model, num_samples):
    assert true_model.m == true_model.n
    d = true_model.m
    x = (torch.rand(num_samples, 1) ** (1 / d)) * torch.nn.functional.normalize(
        torch.randn(num_samples, d)
    )
    y = true_model(x)

    def loss_fn(model):
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        return loss

    return loss_fn


# not a fixture as we're generating data for several m, n combinations
# and I couldn't figure out how to fit that into the fixture mold
def generated_rrr_dataset(m, n):
    torch.manual_seed(42)
    np.random.seed(42)
    num_samples = 1000
    x = torch.randn(num_samples, m)
    y = torch.randn(num_samples, n)
    train_data = TensorDataset(x, y)
    train_dataloader = DataLoader(train_data, batch_size=num_samples, shuffle=True)
    return train_dataloader, train_data, x, y


@pytest.mark.parametrize("sampling_method", [SGLD])
@pytest.mark.parametrize(
    "m,h,n",
    [
        (5, 3, 5),  # case 1, odd
        (5, 4, 5),  # case 1, even
        (4, 3, 8),  # case 2
        (8, 3, 4),  # case 3
        (3, 8, 4),  # case 4
    ],
)
def test_accuracy_rrr(sampling_method, m, h, n):
    # see "The Generalization Error of Reduced Rank Regression in Bayesian Estimation", M. Aoyagi & S. Watanabe, 2004.
    # Note: RRR is kind of an odd fit for pytorch, being a two-layer no-bias linear model.
    # We train this model long enough to (hopefully) not end up in a local min
    torch.manual_seed(42)
    np.random.seed(42)
    criterion = F.mse_loss
    train_dataloader, train_data, x, y = generated_rrr_dataset(m, n)
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
    num_draws = 2_000
    llc_estimator = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
    )
    sample(
        model,
        train_dataloader,
        criterion=criterion,
        optimizer_kwargs=dict(lr=0.0006, localization=1.0),
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
    ), f"DLN case {case} estimated LLC mean {llc_mean:.3f} +- {2*llc_std_dev:.3f} vs True LC {true_lc:.3f} for (M, H, N)={(m, h, n)} using {sampling_method}"


# TODO:
# Scale up these estimates like in Furman & Lau (2024), also to DLNs more generally
#
# For models with a closed-form population loss, like DLNs:
# compare SGLD on empirical loss with SGLD on population loss (results should agree)
# SGLD on population loss should be able to get the LLC exactly correct,
# assuming beta is sufficiently high (using population loss here instead of empirical loss allows very high beta without prohibitively large training set size)
