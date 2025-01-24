import numpy as np
import pytest
import torch
import torch.nn.functional as F
from devinterp.optim import SGLD, SGMCMC
from devinterp.optim.sgnht import SGNHT
from devinterp.slt.llc import LLCEstimator
from devinterp.slt.sampler import sample
from devinterp.test_utils import *
from devinterp.utils import default_nbeta, evaluate_mse, get_init_loss_multi_batch
from torch.utils.data import DataLoader, TensorDataset


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


@pytest.mark.parametrize("sampling_method", [SGLD, SGNHT, SGMCMC.sgld])
def test_seeding(generated_normalcrossing_dataset, sampling_method):
    torch.manual_seed(42)
    seed = 42

    model = Polynomial()

    train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
    lr = 0.0001
    num_chains = 3
    num_draws = 100
    init_loss = get_init_loss_multi_batch(
        train_dataloader, num_chains, model, evaluate_mse, device="cpu"
    )
    llc_estimator_1 = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss,
    )
    llc_estimator_2 = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss,
    )
    torch.manual_seed(42)

    sample(
        model,
        train_dataloader,
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(lr=lr, nbeta=default_nbeta(train_dataloader)),
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
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(lr=lr, nbeta=default_nbeta(train_dataloader)),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[llc_estimator_2],
        verbose=False,
        seed=seed,
    )
    llc_mean_1 = llc_estimator_1.get_results()["llc/mean"]
    llc_mean_2 = llc_estimator_2.get_results()["llc/mean"]
    assert np.array_equal(
        llc_mean_1, llc_mean_2
    ), f"LLC mean {llc_mean_1:.8f}!={llc_mean_2:.8f} for same seed for sampler {SGLD}!"


# @pytest.mark.parametrize("batch_sizes", [[1, 10, 100, 1000]])
@pytest.mark.parametrize("sampling_method", [SGLD, SGMCMC.sgld])
# @pytest.mark.parametrize("model", [Polynomial])
def unused_test_batch_size_convergence(
    generated_normalcrossing_dataset, batch_sizes, sampling_method, model
):
    model = model([2, 2])
    criterion = F.mse_loss
    lr = 0.0002
    num_chains = 1
    means = []
    stds = []
    _, train_data, _, _ = generated_normalcrossing_dataset
    for batch_size in batch_sizes:

        num_draws = 5_000
        torch.manual_seed(42)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        init_loss = get_init_loss_multi_batch(
            train_dataloader, num_chains, model, evaluate_mse, device="cpu"
        )
        llc_estimator = LLCEstimator(
            num_chains=num_chains,
            num_draws=num_draws,
            nbeta=default_nbeta(train_dataloader),
            init_loss=init_loss,
        )
        sample(
            model,
            train_dataloader,
            evaluate=evaluate_mse,
            optimizer_kwargs=dict(lr=lr, localization=1.0),
            sampling_method=sampling_method,
            num_chains=num_chains,
            num_draws=num_draws,
            callbacks=[llc_estimator],
            verbose=False,
        )
        means += [llc_estimator.get_results()["llc/mean"]]
        stds += [llc_estimator.get_results()["llc/std"]]
    overall_mean = np.mean(means)
    std_dev_of_means = np.std(means)
    assert (
        False
    ), f"mean {overall_mean}, std_dev_of_means {std_dev_of_means}, {means}, {stds}"
