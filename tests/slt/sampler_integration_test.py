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


class Polynomial(nn.Module):
    def __init__(self, powers=[1, 1]):
        super(Polynomial, self).__init__()
        self.powers = torch.tensor(powers)
        self.weights = nn.Parameter(
            torch.tensor(
                torch.zeros_like(self.powers, dtype=torch.float32), requires_grad=True
            )
        )

    def forward(self, x):
        return x * torch.prod(self.weights**self.powers)


class LinePlusDot(nn.Module):
    def __init__(self, dim=2):
        super(LinePlusDot, self).__init__()
        self.weights = nn.Parameter(
            torch.zeros(dim, dtype=torch.float32), requires_grad=True
        )

    def forward(self, x):
        return x * (self.weights[0] - 2) * (torch.sum(self.weights**2) ** 2)


@pytest.fixture
def generated_dataset():
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


@pytest.mark.parametrize("sampling_method", [SGLD, SGNHT])
@pytest.mark.parametrize("model", [LinePlusDot, Polynomial])
@pytest.mark.parametrize("dim", [2])
def test_llc_ordinality(generated_dataset, sampling_method, model, dim):
    if model == Polynomial:
        model = model([2 for _ in range(dim)])
    else:
        model = model(dim)
    train_loader, train_data, _, _ = generated_dataset
    torch.manual_seed(42)
    np.random.seed(42)
    criterion = F.mse_loss
    lr = 0.0001
    num_chains = 5
    num_draws = 1_000
    llcs = []
    sample_points = [
        [0.0 for _ in range(dim)],
        [2.0 if i == 0 else 0.0 for i in range(dim)],
    ] + ([[0.0 if i == dim - 1 else 2.0 for i in range(dim)]] if dim > 2 else [])
    for sample_point in sample_points:
        model.weights = nn.Parameter(
            torch.tensor(sample_point, dtype=torch.float32, requires_grad=True)
        )
        llc_estimator = LLCEstimator(
            num_chains=num_chains, num_draws=num_draws, n=len(train_data)
        )
        sample(
            model,
            train_loader,
            criterion=criterion,
            optimizer_kwargs=dict(
                lr=lr,
                bounding_box_size=.5,
                num_samples=len(train_data),
            ),
            sampling_method=sampling_method,
            num_chains=num_chains,
            num_draws=num_draws,
            callbacks=[llc_estimator],
            verbose=False,
        )
        llcs += [llc_estimator.sample()["llc/mean"]]
    assert (
        np.diff(llcs) >= 0
    ).all(), f"Ordinality not preserved for sampler {sampling_method} on {dim}-d {model}: llcs {llcs:.3f} are not in ascending order."


TRUE_LCS = {
    (0, 1): 1 / 2,
    (1, 1): 1 / 2,
    (0, 2): 1 / 4,
    (1, 2): 1 / 4,
    (2, 2): 1 / 4,
    (0, 3): 1 / 6,
    (1, 3): 1 / 6,
    (2, 3): 1 / 6,
    (3, 3): 1 / 6,
}


@pytest.mark.parametrize("sampling_method", [SGLD, SGNHT])
@pytest.mark.parametrize("powers", list(TRUE_LCS.keys()))
def test_llc_accuracy(generated_dataset, sampling_method, powers):
    true_lc = TRUE_LCS[powers]
    model = Polynomial(powers)
    train_loader, train_data, _, _ = generated_dataset
    torch.manual_seed(42)
    np.random.seed(42)
    criterion = F.mse_loss
    lr = 0.0001
    num_chains = 5
    num_draws = 10_000
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
    )
    llc_mean = llc_estimator.sample()["llc/mean"]
    llc_std_dev = llc_estimator.sample()["llc/std"]
    assert (
        llc_mean - llc_std_dev < true_lc < llc_mean + llc_std_dev
    ), f"LLC mean {llc_mean:.3f} +- {llc_std_dev:.3f} does not contain true value {true_lc:.3f} for powers {powers} using {sampling_method}"
