import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from devinterp.slt import sample
from devinterp.slt.llc import LLCEstimator, OnlineLLCEstimator
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


POWERS = [
    [
        [1, 1, 0],
        [1, 1, 10],
    ],
    [
        [2, 2, 10],
        [2, 2, 3],
    ],
    [
        [3, 3, 6.1],
        [3, 3, 1.2],
    ],
]

SAMPLE_POINTS = [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]


@pytest.mark.parametrize("sampling_method", [SGLD, SGNHT])
@pytest.mark.parametrize("powers", POWERS)
@pytest.mark.parametrize("sample_point", SAMPLE_POINTS)
def test_rllc_normalcrossing_between_powers(
    generated_normalcrossing_dataset, sampling_method, powers, sample_point
):
    seed = 42
    torch.manual_seed(seed)

    model1 = Polynomial(powers[0])
    model1.weights = torch.nn.Parameter(torch.tensor(sample_point))
    model2 = Polynomial(powers[1])
    model2.weights = torch.nn.Parameter(torch.tensor(sample_point))

    train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
    criterion = F.mse_loss
    lr = 0.0002
    num_chains = 1
    num_draws = 100
    llc_estimator_1 = LLCEstimator(
        num_chains=num_chains, num_draws=num_draws, n=len(train_data)
    )
    llc_estimator_2 = LLCEstimator(
        num_chains=num_chains, num_draws=num_draws, n=len(train_data)
    )
    torch.manual_seed(seed)

    sample(
        model1,
        train_dataloader,
        criterion=criterion,
        optimizer_kwargs=dict(
            lr=lr,
            num_samples=len(train_data),
        ),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator_1],
        verbose=False,
        seed=seed,
        optimize_over_per_model_param={"weights": torch.tensor([1, 1, 0])},
    )
    torch.manual_seed(seed)

    sample(
        model2,
        train_dataloader,
        criterion=criterion,
        optimizer_kwargs=dict(
            lr=lr,
            num_samples=len(train_data),
        ),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator_2],
        verbose=False,
        seed=seed,
        optimize_over_per_model_param={"weights": torch.tensor([1, 1, 0])},
    )
    llc_mean_1 = llc_estimator_1.sample()["llc/mean"]
    llc_mean_2 = llc_estimator_2.sample()["llc/mean"]
    assert np.isclose(
        llc_mean_1, llc_mean_2, atol=1e-5
    ), f"LLC mean {llc_mean_1:.3f}!={llc_mean_2:.3f} for powers {powers} using {sampling_method}"


POWERS = [
    [1, 1],
    [2, 10],
]
EXTRA_DIM_POWER = [3, 10, 100]


@pytest.mark.parametrize("sampling_method", [SGLD])
@pytest.mark.parametrize("relevant_powers", POWERS)
@pytest.mark.parametrize("extra_dim_power", EXTRA_DIM_POWER)
@pytest.mark.parametrize("sample_point", SAMPLE_POINTS)
def test_rllc_gradient_normalcrossing_between_dims(
    generated_normalcrossing_dataset,
    sampling_method,
    relevant_powers,
    extra_dim_power,
    sample_point,
):
    torch.manual_seed(42)
    seed = 42

    model1 = Polynomial(relevant_powers)
    model2 = Polynomial(relevant_powers + [extra_dim_power])

    model1.weights = torch.nn.Parameter(torch.tensor(sample_point[:-1]))
    model2.weights = torch.nn.Parameter(torch.tensor(sample_point))

    train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
    criterion = F.mse_loss
    lr = 0.0001
    num_chains = 1
    num_draws = 50
    llc_estimator_2d = LLCEstimator(  # TODO look at the weights instead
        num_chains=num_chains, num_draws=num_draws, n=len(train_data)
    )
    llc_estimator_3d = LLCEstimator(  # TODO look at the weights instead
        num_chains=num_chains, num_draws=num_draws, n=len(train_data)
    )

    sample(
        model1,
        train_dataloader,
        criterion=criterion,
        optimizer_kwargs=dict(lr=lr, num_samples=len(train_data), noise_level=0.0),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator_2d],
        verbose=False,
        seed=seed,
    )
    sample(
        model2,
        train_dataloader,
        criterion=criterion,
        optimizer_kwargs=dict(lr=lr, num_samples=len(train_data), noise_level=0.0),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator_3d],
        verbose=False,
        seed=seed,
        optimize_over_per_model_param={"weights": torch.tensor([1, 1, 0])},
    )
    llc_mean_2d = llc_estimator_2d.sample()["llc/mean"]
    llc_mean_3d_restricted = llc_estimator_3d.sample()["llc/mean"]
    assert np.isclose(
        llc_mean_2d, llc_mean_3d_restricted, atol=1e-5
    ), f"LLC mean {llc_mean_2d:.3f}!={llc_mean_3d_restricted:.3f} for powers {relevant_powers + [extra_dim_power]} using {sampling_method}, {model2.weights}"

@pytest.mark.parametrize("sampling_method", [SGLD, SGNHT])
@pytest.mark.parametrize("relevant_powers", POWERS)
@pytest.mark.parametrize("extra_dim_power", EXTRA_DIM_POWER)
@pytest.mark.parametrize("sample_point", SAMPLE_POINTS)
def test_rllc_full_normalcrossing_between_dims(
    generated_normalcrossing_dataset,
    sampling_method,
    relevant_powers,
    extra_dim_power,
    sample_point,
):
    torch.manual_seed(42)
    seed = 42

    model1 = Polynomial(relevant_powers)
    model2 = Polynomial(relevant_powers + [extra_dim_power])

    model1.weights = torch.nn.Parameter(torch.tensor(sample_point[:-1]))
    model2.weights = torch.nn.Parameter(torch.tensor(sample_point))

    train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
    criterion = F.mse_loss
    lr = 0.0001
    num_chains = 1
    num_draws = 2 if sampling_method == SGLD else 1
    llc_estimator_2d = LLCEstimator(  # TODO look at the weights instead
        num_chains=num_chains, num_draws=num_draws, n=len(train_data)
    )
    llc_estimator_3d = LLCEstimator(  # TODO look at the weights instead
        num_chains=num_chains, num_draws=num_draws, n=len(train_data)
    )

    sample(
        model1,
        train_dataloader,
        criterion=criterion,
        optimizer_kwargs=dict(lr=lr, num_samples=len(train_data)),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator_2d],
        verbose=False,
        seed=seed,
    )
    sample(
        model2,
        train_dataloader,
        criterion=criterion,
        optimizer_kwargs=dict(lr=lr, num_samples=len(train_data)),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator_3d],
        verbose=False,
        seed=seed,
        optimize_over_per_model_param={"weights": torch.tensor([1, 1, 0])},
    )
    llc_mean_2d = llc_estimator_2d.sample()["llc/mean"]
    llc_mean_3d_restricted = llc_estimator_3d.sample()["llc/mean"]
    assert np.array_equal(
        llc_mean_2d, llc_mean_3d_restricted
    ), f"LLC mean {llc_mean_2d:.8f}!={llc_mean_3d_restricted:.8f} for powers {relevant_powers + [extra_dim_power]} using {sampling_method}, {model2.weights}"
