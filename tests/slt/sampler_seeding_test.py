import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from devinterp.slt import sample
from devinterp.slt.llc import LLCEstimator
from devinterp.zoo.test_utils import *
from devinterp.utils import *


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


@pytest.mark.parametrize("sampling_method", [SGLD, SGNHT])
def test_seeding(generated_normalcrossing_dataset, sampling_method):
    torch.manual_seed(42)
    seed = 42

    model = Polynomial()

    train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
    criterion = F.mse_loss
    lr = 0.0001
    num_chains = 3
    num_draws = 100
    llc_estimator_1 = LLCEstimator(
        num_chains=num_chains, num_draws=num_draws, n=len(train_data)
    )
    llc_estimator_2 = LLCEstimator(
        num_chains=num_chains, num_draws=num_draws, n=len(train_data)
    )
    torch.manual_seed(42)

    sample(
        model,
        train_dataloader,
        criterion=criterion,
        optimizer_kwargs=dict(lr=lr, num_samples=len(train_data)),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator_1],
        verbose=False,
        seed=seed,
    )
    torch.manual_seed(42)

    sample(
        model,
        train_dataloader,
        criterion=criterion,
        optimizer_kwargs=dict(lr=lr, num_samples=len(train_data)),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator_2],
        verbose=False,
        seed=seed,
    )
    llc_mean_1 = llc_estimator_1.sample()["llc/mean"]
    llc_mean_2 = llc_estimator_2.sample()["llc/mean"]
    assert np.array_equal(
        llc_mean_1, llc_mean_2
    ), f"LLC mean {llc_mean_1:.8f}!={llc_mean_2:.8f} for same seed for sampler {SGLD}!"
