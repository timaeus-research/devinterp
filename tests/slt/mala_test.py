import numpy as np
import pytest
import torch
import torch.nn.functional as F
from devinterp.optim.sgld import SGLD
from devinterp.slt.mala import MalaAcceptanceRate, mala_acceptance_probability
from devinterp.slt.sampler import sample
from devinterp.test_utils import *
from devinterp.utils import default_nbeta, make_evaluate
from torch.utils.data import DataLoader, TensorDataset


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
        return torch.sum(torch.pow(self.weights, self.powers))


@pytest.fixture
def generated_normalcrossing_dataset():
    torch.manual_seed(42)
    np.random.seed(42)
    num_samples = 1000
    x = torch.zeros(num_samples)
    y = torch.zeros(num_samples)
    train_data = TensorDataset(x, y)
    train_dataloader = DataLoader(train_data, batch_size=num_samples, shuffle=True)
    return train_dataloader, train_data, x, y


def linear_loss(y_preds, ys):
    return torch.mean(y_preds)


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
    assert np.isclose(
        mala_accept_prob, benchmark_accept_prob, atol=0.000001
    ), f"MALA accept prob {mala_accept_prob}, not close to benchmark value {benchmark_accept_prob:.2f}"


SETS_TO_TEST = [
    [[2], 1.0e-2, 100.0, 0.59],
    [[2], 1.0e-2, 0.1, 0.73],
    [[2], 1.0e-3, 100.0, 0.98],
    [[4], 1.5e-2, 100.0, 0.85],
    [[4], 1.5e-2, 1.0, 0.935],
    [[4], 1.5e-5, 100.0, 0.999],
    [[8], 1.5e-2, 100.0, 0.87],
    [[8], 1.0e-2, 0.1, 0.97],
    [[8], 1.0e-5, 100.0, 0.999],
    [[2, 2], 1.0e-3, 100.0, 0.971],
    [[2, 2], 1.0e-2, 100.0, 0.54],
    [[2, 2], 3e-3, 0.01, 0.91],
    [[0, 2], 2.0e-3, 100.0, 0.95],
    [[4, 0], 1.0e-3, 100.0, 0.995],
    [[4, 4], 1.0e-2, 100.0, 0.87],
]


@pytest.mark.slow
@pytest.mark.parametrize("powers,lr,localization,accept_prob", SETS_TO_TEST)
def test_mala_callback_closeness(
    generated_normalcrossing_dataset,
    powers,
    lr,
    localization,
    accept_prob,
):
    seed = 0
    for seed in range(10):
        seed += 1
        model = Polynomial(powers)
        train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
        evaluate = make_evaluate(linear_loss)
        num_draws = 5_000
        num_chains = 1
        mala_estimator = MalaAcceptanceRate(
            num_chains=num_chains,
            num_draws=num_draws,
            nbeta=default_nbeta(train_dataloader),
            learning_rate=lr,
        )
        sample(
            model,
            train_dataloader,
            evaluate=evaluate,
            optimizer_kwargs=dict(
                lr=lr,
                localization=localization,
                nbeta=default_nbeta(train_dataloader),
            ),
            sampling_method=SGLD,
            num_chains=num_chains,
            num_draws=num_draws,
            callbacks=[mala_estimator],
            verbose=False,
            seed=seed,
        )
        mala_acceptance_rate_mean = mala_estimator.get_results()["mala_accept/mean"]
        if not np.isnan(mala_acceptance_rate_mean):
            break
    assert np.isclose(
        mala_acceptance_rate_mean, accept_prob, atol=0.01
    ), f"MALA Rate mean {mala_acceptance_rate_mean:.2f}, not close to benchmark value {accept_prob:.2f}, lr {lr} elas {localization}"
