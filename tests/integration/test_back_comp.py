from unittest import mock

import numpy as np
import pytest
import torch
from devinterp.optim.sgld import SGLD
from devinterp.slt.llc import LLCEstimator, OnlineLLCEstimator
from devinterp.slt.sampler import sample
from devinterp.utils import default_nbeta, evaluate_mse, get_init_loss_multi_batch
from torch.utils.data import DataLoader, TensorDataset


@pytest.mark.parametrize("sampling_method", [SGLD])
@pytest.mark.parametrize("estimator", [LLCEstimator, OnlineLLCEstimator])
def test_dont_allow_both_temp_and_nbeta(
    generated_normalcrossing_dataset, sampling_method, estimator, Polynomial
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
            sampling_method_kwargs=dict(
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
            sampling_method_kwargs=dict(
                lr=lr,
                nbeta=2.0,
            ),
            sampling_method=sampling_method,
            num_chains=num_chains,
            num_draws=num_draws,
            callbacks=[llc_estimator],
            verbose=False,
        )


def test_warn_on_default_nbeta():
    with mock.patch("devinterp.utils.warnings") as mock_warn:
        _ = default_nbeta(
            DataLoader(TensorDataset(torch.randn(100, 10)), batch_size=1),
            gradient_accumulation_steps=1,
        )
        # Check that a warning was issued
        mock_warn.warn.assert_called_with(
            "default nbeta is undefined for batch_size * gradient_accumulation_steps == 1, falling back to default value of 1"
        )
    with mock.patch("devinterp.utils.warnings") as mock_warn:
        _ = default_nbeta(1, gradient_accumulation_steps=1)

        # Check that a warning was issued
        mock_warn.warn.assert_called_with(
            "default nbeta is undefined for batch_size * gradient_accumulation_steps == 1, falling back to default value of 1"
        )
