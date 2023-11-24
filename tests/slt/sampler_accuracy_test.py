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
        return x * (self.weights[0] - 1) * (torch.sum(self.weights**2) ** 2)


@pytest.fixture
def generated_linedot_normalcrossing_dataset():
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
def test_accuracy_linedot_normalcrossing(
    generated_linedot_normalcrossing_dataset, sampling_method, powers
):
    seed = 42
    true_lc = TRUE_LCS[powers]
    model = Polynomial(powers)
    train_loader, train_data, _, _ = generated_linedot_normalcrossing_dataset
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
        llc_mean - llc_std_dev < true_lc < llc_mean + llc_std_dev
    ), f"LLC mean {llc_mean:.3f} +- {llc_std_dev:.3f} does not contain true value {true_lc:.3f} for powers {powers} using {sampling_method}"


# @pytest.fixture
def generated_rrr_dataset():
    # Sample data (replace these with your actual data)
    torch.manual_seed(42)
    np.random.seed(42)
    num_samples = 1000
    batch_size = num_samples
    d_in = 5
    d_out = 5
    x = torch.randn(num_samples, d_in)
    y = torch.randn(num_samples, d_out)
    train_data = TensorDataset(x, y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader, train_data, x, y


class ReducedRankRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(ReducedRankRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, rank, bias=False)
        self.fc2 = nn.Linear(rank, output_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)


def test_accuracy_rrr(generated_rrr_dataset):
    torch.manual_seed(42)
    np.random.seed(42)
    criterion = F.mse_loss
    rank = 3
    train_loader, train_data, x, y = generated_rrr_dataset
    #  M -> H (rank) -> N
    M = x.size(1)
    H = rank
    N = y.size(1)
    print(M, N, H)
    print(M + H < N)  # case 2
    print(N + H < M)  # case 3
    print(M + N < H)  # case 4
    print((M + H + N) % 2 == 0)  # case 1, even
    print((M + H + N) % 2 == 1)  # case 1, odd
    lambda_case_2 = M * H / 2
    lambda_case_3 = H * N / 2
    lambda_case_4 = M * N / 2
    lambda_if_even = (2 * M * N + 2 * H * N + 2 * M * H - N**2 - M**2 - H**2) / 8
    lambda_if_odd = (
        1 + 2 * M * N + 2 * H * N + 2 * M * H - N**2 - M**2 - H**2
    ) / 8
    print(lambda_case_2, lambda_case_3, lambda_case_4, lambda_if_even, lambda_if_odd)
    model = ReducedRankRegressor(M, N, H)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    num_samples = len(train_loader)
    batch_size = num_samples
    num_chains = 20
    num_draws = 1_000
    for epoch in range(25000):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    print(loss)
    llc_estimator = LLCEstimator(
        num_chains=num_chains, num_draws=num_draws, n=len(train_data)
    )
    sample(
        model,
        train_loader,
        criterion=criterion,
        optimizer_kwargs=dict(
            lr=0.005,
            bounding_box_size=2.0,
            num_samples=len(train_data),
        ),
        sampling_method=SGLD,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator],
        verbose=False,
        seed=42,
    )
    llc_mean = llc_estimator.sample()["llc/mean"]
    llc_std_dev = llc_estimator.sample()["llc/std"]
    print(llc_mean, llc_std_dev)


test_accuracy_rrr(generated_rrr_dataset())
