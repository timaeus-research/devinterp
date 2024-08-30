import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from devinterp.optim.sgld import SGLD
from devinterp.slt.llc import LLCEstimator
from devinterp.slt.sampler import sample
from devinterp.test_utils import *
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


@pytest.mark.parametrize("sampling_method", [SGLD])
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

    train_dataloader, _, _, _ = generated_normalcrossing_dataset
    lr = 0.0002
    num_chains = 1
    num_draws = 100
    init_loss_1 = get_init_loss_multi_batch(
        train_dataloader, num_chains, model1, evaluate_mse, device="cpu"
    )
    init_loss_2 = get_init_loss_multi_batch(
        train_dataloader, num_chains, model2, evaluate_mse, device="cpu"
    )
    llc_estimator_1 = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss_1,
    )
    llc_estimator_2 = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss_2,
    )
    torch.manual_seed(seed)

    sample(
        model1,
        train_dataloader,
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(
            lr=lr,
            nbeta=default_nbeta(train_dataloader),
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
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(
            lr=lr,
            nbeta=default_nbeta(train_dataloader),
        ),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator_2],
        verbose=False,
        seed=seed,
        optimize_over_per_model_param={"weights": torch.tensor([1, 1, 0])},
    )
    llc_mean_1 = llc_estimator_1.get_results()["llc/mean"]
    llc_mean_2 = llc_estimator_2.get_results()["llc/mean"]
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
def test_restricted_gradient_normalcrossing_between_dims(
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
    lr = 0.0001
    num_chains = 1
    num_draws = 1000
    init_loss_1 = get_init_loss_multi_batch(
        train_dataloader, num_chains, model1, evaluate_mse, device="cpu"
    )
    init_loss_2 = get_init_loss_multi_batch(
        train_dataloader, num_chains, model2, evaluate_mse, device="cpu"
    )
    llc_estimator_2d = LLCEstimator(  # TODO look at the weights instead
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss_1,
    )
    llc_estimator_3d = LLCEstimator(  # TODO look at the weights instead
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss_2,
    )

    sample(
        model1,
        train_dataloader,
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(
            lr=lr, nbeta=default_nbeta(train_dataloader), noise_level=0.0
        ),
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
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(
            lr=lr, nbeta=default_nbeta(train_dataloader), noise_level=0.0
        ),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator_3d],
        verbose=False,
        seed=seed,
        optimize_over_per_model_param={"weights": torch.tensor([1, 1, 0])},
    )
    llc_mean_2d = llc_estimator_2d.get_results()["llc/mean"]
    llc_mean_3d_restricted = llc_estimator_3d.get_results()["llc/mean"]
    assert np.isclose(
        llc_mean_2d, llc_mean_3d_restricted, atol=1e-5
    ), f"LLC mean {llc_mean_2d:.3f}!={llc_mean_3d_restricted:.3f} for powers {relevant_powers + [extra_dim_power]} using {sampling_method}, {model2.weights}"


SAMPLE_POINTS = [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]


@pytest.mark.parametrize("sampling_method", [SGLD])
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
    lr = 0.001
    num_chains = 1
    num_draws = 500
    model1 = Polynomial(relevant_powers)
    model2 = Polynomial(relevant_powers + [extra_dim_power])

    model1.weights = torch.nn.Parameter(torch.tensor(sample_point[:-1]))
    model2.weights = torch.nn.Parameter(torch.tensor(sample_point))

    train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
    init_loss_1 = get_init_loss_multi_batch(
        train_dataloader, num_chains, model1, evaluate_mse, device="cpu"
    )
    init_loss_2 = get_init_loss_multi_batch(
        train_dataloader, num_chains, model2, evaluate_mse, device="cpu"
    )

    llc_estimator_2d = LLCEstimator(  # TODO look at the weights instead
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss_1,
    )
    llc_estimator_3d = LLCEstimator(  # TODO look at the weights instead
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss_2,
    )

    sample(
        model1,
        train_dataloader,
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(lr=lr, nbeta=default_nbeta(train_dataloader)),
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
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(lr=lr, nbeta=default_nbeta(train_dataloader)),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator_3d],
        verbose=False,
        seed=seed,
        optimize_over_per_model_param={"weights": torch.tensor([1, 1, 0])},
    )
    llc_mean_2d = llc_estimator_2d.get_results()["llc/mean"]
    llc_mean_3d_restricted = llc_estimator_3d.get_results()["llc/mean"]
    assert np.isclose(
        llc_mean_2d, llc_mean_3d_restricted, atol=8e-2
    ), f"LLC mean {llc_mean_2d:.8f}!={llc_mean_3d_restricted:.8f} for powers {relevant_powers + [extra_dim_power]} using {sampling_method}, {model2.weights}"


POWERS = [[0, 1], [1, 2], [0, 3]]


@pytest.mark.parametrize("sampling_method", [SGLD])
@pytest.mark.parametrize("relevant_powers", POWERS)
def test_rllc_different_from_full_llc_between_dims(
    generated_normalcrossing_dataset, sampling_method, relevant_powers
):
    torch.manual_seed(42)
    seed = 42

    model = Polynomial(relevant_powers)
    model.weights = torch.nn.Parameter(torch.tensor([0.3, 1.5]))

    train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
    lr = 0.001
    num_chains = 1
    num_draws = 200
    init_loss = get_init_loss_multi_batch(
        train_dataloader, num_chains, model, evaluate_mse, device="cpu"
    )
    llc_estimator = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss,
    )
    rllc_estimator = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss,
    )

    sample(
        model,
        train_dataloader,
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(lr=lr, nbeta=default_nbeta(train_dataloader)),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator],
        verbose=False,
        seed=seed,
    )
    sample(
        model,
        train_dataloader,
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(lr=lr, nbeta=default_nbeta(train_dataloader)),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[rllc_estimator],
        verbose=False,
        seed=seed,
        optimize_over_per_model_param={"weights": torch.tensor([1, 0])},
    )
    llc_mean = llc_estimator.get_results()["llc/mean"]
    rllc_mean = rllc_estimator.get_results()["llc/mean"]
    assert not np.isclose(
        llc_mean, rllc_mean, atol=1e-2
    ), f"LLC {llc_mean:.3f} too close to RLLC {rllc_mean:.3f} for powers {relevant_powers} using {sampling_method}"
