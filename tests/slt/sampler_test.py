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


# @pytest.mark.parametrize("batch_sizes", [[1, 10, 100, 1000]])
# @pytest.mark.parametrize("sampling_method", [SGLD, SGNHT])
# @pytest.mark.parametrize("model", [Polynomial])
def test_batch_size_convergence(batch_sizes, sampling_method, model):
    seed = 42
    model = model([2, 2])
    criterion = F.mse_loss
    lr = 0.0000002
    num_chains = 5
    means = []
    stds = []
    train_data, _, _ = generated_normalcrossing_dataset()
    for batch_size in batch_sizes:
        num_draws = 5_000 * 1000 // batch_size
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        llc_estimator = LLCEstimator(
            num_chains=num_chains, num_draws=num_draws, n=len(train_data)
        )

        online_llc_estimator = OnlineLLCEstimator(
            num_chains=num_chains, num_draws=num_draws, n=len(train_data)
        )
        sample(
            model,
            train_dataloader,
            criterion=criterion,
            optimizer_kwargs=dict(
                lr=lr,
                bounding_box_size=1.0,
                num_samples=len(train_data),
            ),
            sampling_method=sampling_method,
            num_chains=num_chains,
            num_draws=num_draws,
            callbacks=[online_llc_estimator],
            verbose=False,
        )
        # means += [llc_estimator.sample()["llc/mean"]]
        # stds += [llc_estimator.sample()["llc/std"]]
        trace = online_llc_estimator.sample()["llc/trace"]
        plot_trace(trace, f"LLC {batch_size}")
        # print(llc_estimator.sample()["llc/mean"])
    overall_mean = np.mean(means)
    std_dev_of_means = np.std(means)
    assert (
        False
    ), f"mean {overall_mean}, std_dev_of_means {std_dev_of_means}, {means}, {stds}"
