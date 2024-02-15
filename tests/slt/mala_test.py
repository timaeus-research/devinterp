import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from devinterp.optim.sgld import SGLD
from devinterp.slt import sample
from devinterp.slt.mala import MalaAcceptanceRate
from devinterp.test_utils import *
from devinterp.utils import *


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


# @pytest.fixture
def generated_normalcrossing_dataset():
    torch.manual_seed(42)
    np.random.seed(42)
    num_samples = 1000
    x = torch.zeros(num_samples)
    y = torch.zeros(num_samples)
    train_data = TensorDataset(x, y)
    train_dataloader = DataLoader(train_data, batch_size=num_samples, shuffle=True)
    return train_dataloader, train_data, x, y


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
    [[2, 2], 1.0e-3, 100.0, 0.971],  # linear_loss
]


def quaternary_loss(y_preds, ys):
    return torch.mean(torch.pow((y_preds), 4))


def linear_loss(y_preds, ys):
    return torch.mean(y_preds)


# @pytest.mark.parametrize("sampling_method", [SGLD])
# @pytest.mark.parametrize("powers,lr,elasticity,accept_prob", SETS_TO_TEST)
def test_mala_closeness(
    generated_normalcrossing_dataset,
    sampling_method,
    powers,
    lr,
    elasticity,
    accept_prob,
):
    seed = 0
    for seed in range(10):
        seed += 1
        model = Polynomial(powers)
        # model.weights = nn.Parameter(torch.tensor([-0.01065028, -0.03249225]))
        train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
        criterion = linear_loss
        num_draws = 5_000
        num_chains = 1
        mala_estimator = MalaAcceptanceRate(
            num_chains=1,
            num_draws=num_draws,
            num_samples=len(train_data),
            model=model,
            elasticity=elasticity,
            learning_rate=lr,
        )
        sample(
            model,
            train_dataloader,
            criterion=criterion,
            optimizer_kwargs=dict(
                lr=lr, elasticity=elasticity, num_samples=len(train_data)
            ),
            sampling_method=sampling_method,
            num_chains=num_chains,
            num_draws=num_draws,
            callbacks=[mala_estimator],
            verbose=False,
            seed=seed,
        )
        mala_acceptance_rate_mean = mala_estimator.sample()["mala_accept/mean"]
        if not np.isnan(mala_acceptance_rate_mean):
            break
    plt.plot(mala_estimator.sample()["mala_accept/trace"][-1])
    plt.show()
    print(
        f"hyperparams: loss = weight^{powers[0]*2:<2}, lr = {lr:<6.5f}, gamma = {elasticity:<7.3f}, zach's value = {accept_prob}, ours = {mala_acceptance_rate_mean:.2f}"
    )
    assert np.isclose(
        mala_acceptance_rate_mean, accept_prob, rtol=0.01
    ), f"MALA Rate mean {mala_acceptance_rate_mean:.3f}, not close to benchmark value {accept_prob:.3f}, lr {lr} elas {elasticity}"


for _ in SETS_TO_TEST:
    test_mala_closeness(
        generated_normalcrossing_dataset(),
        SGLD,
        *_,
    )
