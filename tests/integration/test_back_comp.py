import numpy as np
import pytest
import torch
from devinterp.optim.sgld import SGLD
from devinterp.slt.llc import LLCEstimator, OnlineLLCEstimator
from devinterp.slt.sampler import sample
from devinterp.test_utils import *
from devinterp.utils import evaluate_mse, get_init_loss_multi_batch
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


@pytest.mark.parametrize("sampling_method", [SGLD])
@pytest.mark.parametrize("estimator", [LLCEstimator, OnlineLLCEstimator])
def test_pass_in_temperature(
    generated_normalcrossing_dataset, sampling_method, estimator
):
    seed = 0
    torch.manual_seed(seed)
    model = Polynomial()
    train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
    lr = 0.001
    num_chains = 2
    num_draws = 10
    temperature = nbeta = 10.0
    init_loss = get_init_loss_multi_batch(
        train_dataloader, num_chains, model, evaluate_mse, device="cpu"
    )
    temp_llc_estimator = estimator(
        num_chains=num_chains,
        num_draws=num_draws,
        temperature=temperature,
        init_loss=init_loss,
    )
    nbeta_llc_estimator = estimator(
        num_chains=num_chains, num_draws=num_draws, nbeta=nbeta, init_loss=init_loss
    )
    torch.manual_seed(seed)
    sample(
        model,
        train_dataloader,
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(lr=lr, temperature=temperature),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[temp_llc_estimator],
        verbose=False,
        seed=seed,
    )
    torch.manual_seed(seed)
    sample(
        model,
        train_dataloader,
        evaluate=evaluate_mse,
        optimizer_kwargs=dict(lr=lr, nbeta=nbeta),
        sampling_method=sampling_method,
        num_chains=num_chains,
        num_draws=num_draws,
        callbacks=[nbeta_llc_estimator],
        verbose=False,
        seed=seed,
    )
    nbeta_llc_estimator = nbeta_llc_estimator.get_results()
    temp_llc_estimator = temp_llc_estimator.get_results()
    for k, v in nbeta_llc_estimator.items():
        if isinstance(v, torch.Tensor):
            assert torch.allequal(
                v, temp_llc_estimator[k]
            ), f"Evaluation failed for {k}"
        elif isinstance(v, np.ndarray):
            assert np.equal(
                v, temp_llc_estimator[k]
            ).all(), f"Evaluation failed for {k}"
        else:
            assert np.equal(v, temp_llc_estimator[k]), f"Evaluation failed for {k}"


@pytest.mark.parametrize("sampling_method", [SGLD])
@pytest.mark.parametrize("estimator", [LLCEstimator, OnlineLLCEstimator])
def test_dont_allow_both_temp_and_nbeta(
    generated_normalcrossing_dataset, sampling_method, estimator
):
    model = Polynomial([2, 2])
    with pytest.raises(AssertionError):
        train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
        lr = 0.0004
        num_chains = 1
        num_draws = 2
        init_loss = get_init_loss_multi_batch(
            train_dataloader, num_chains, model, evaluate_mse, device="cpu"
        )
        llc_estimator = estimator(
            num_chains=num_chains,
            num_draws=num_draws,
            nbeta=2.0,
            init_loss=init_loss,
        )
        sample(
            model,
            train_dataloader,
            evaluate=evaluate_mse,
            optimizer_kwargs=dict(
                lr=lr,
                temperature=2.0,
            ),
            sampling_method=sampling_method,
            num_chains=num_chains,
            num_draws=num_draws,
            callbacks=[llc_estimator],
            verbose=False,
        )
    with pytest.raises(AssertionError):

        llc_estimator = estimator(
            num_chains=num_chains,
            num_draws=num_draws,
            temperature=2.0,
            init_loss=init_loss,
        )
        sample(
            model,
            train_dataloader,
            evaluate=evaluate_mse,
            optimizer_kwargs=dict(
                lr=lr,
                nbeta=2.0,
            ),
            sampling_method=sampling_method,
            num_chains=num_chains,
            num_draws=num_draws,
            callbacks=[llc_estimator],
            verbose=False,
        )
