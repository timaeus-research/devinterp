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


# @pytest.fixture
def generated_normalcrossing_dataset():
    torch.manual_seed(42)
    np.random.seed(42)
    sigma = 0.25
    num_samples = 1000
    x = torch.normal(0, 1., size=(num_samples,))
    y = torch.zeros(num_samples)
    train_data = TensorDataset(x, y)
    train_dataloader = DataLoader(train_data, batch_size=num_samples, shuffle=True)
    return train_dataloader, train_data, x, y


# TRUE_LCS_PER_POWER = [[2]]


# @pytest.mark.parametrize("sampling_method", [SGLD, SGNHT])
# @pytest.mark.parametrize("powers", TRUE_LCS_PER_POWER)
def test_accuracy_normalcrossing(
    generated_normalcrossing_dataset, sampling_method, powers
):
    seed = 42
    model = Polynomial(powers)
    train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
    criterion = F.mse_loss
    lr = 1.5e-2
    num_draws = 5_000
    num_chains = 1
    mala_estimator = MalaAcceptanceRate(
        num_chains=1,
        num_draws=num_draws,
        num_samples=len(train_data),
        model=model,
        elasticity=100.0,
        learning_rate=lr,
    )
    sample(
        model,
        train_dataloader,
        criterion=criterion,
        optimizer_kwargs=dict(lr=lr, elasticity=100.0, num_samples=len(train_data)),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[mala_estimator],
        verbose=False,
        seed=seed,
    )
    mala_acceptance_rate_mean = mala_estimator.sample()["mala_accept/mean"]
    mala_acceptance_rate_std = mala_estimator.sample()["mala_accept/std"]
    print(mala_estimator.sample()["mala_accept/trace"])
    plt.plot(mala_estimator.sample()["mala_accept/trace"][0])
    plt.show()
    print(
        f"MALA Rate mean {mala_acceptance_rate_mean:.3f}, std {mala_acceptance_rate_std:.3f}  for powers {powers} using {sampling_method}"
    )


test_accuracy_normalcrossing(generated_normalcrossing_dataset(), SGLD, [2])
