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

import itertools

DEVICE = "cuda"


def make_pop_loss_fn(true_model):
    assert true_model.m == true_model.n
    d = true_model.m
    true_A, true_B = (
        true_model.fc1.weight.detach().clone(),
        true_model.fc2.weight.detach().clone(),
    )
    true_prod = true_B @ true_A

    def loss_fn(model):
        Q = true_prod - (model.fc2.weight @ model.fc1.weight)
        loss = ((d / (d + 2)) * (torch.sum(Q * Q) / d)) / d
        return loss

    return loss_fn


def make_emp_loss_fn(true_model, num_samples):
    assert true_model.m == true_model.n
    d = true_model.m
    x = (torch.rand(num_samples, 1) ** (1 / d)) * torch.nn.functional.normalize(
        torch.randn(num_samples, d)
    )
    y = true_model(x)

    def loss_fn(model):
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        return loss

    return loss_fn


# not a fixture as we're generating data for several m, n combinations
# and I couldn't figure out how to fit that into the fixture mold
def generated_rrr_dataset(layer_widths, true_model):
    m = layer_widths[0]
    n = layer_widths[-1]
    torch.manual_seed(42)
    np.random.seed(42)
    num_samples = 10000
    x = torch.randn(num_samples, m)
    true_model.eval()
    with torch.no_grad():
        y = true_model(x)
    train_data = TensorDataset(x, y)
    train_dataloader = DataLoader(train_data, batch_size=500, shuffle=True)
    return train_dataloader, train_data, x, y


# @pytest.mark.parametrize("sampling_method", [SGLD, SGNHT])
# @pytest.mark.parametrize(
#     "m,h,n",
#     [
#         (5, 3, 5),  # case 1, odd
#         (2, 1, 2),  # case 1, odd
#         (5, 4, 5),  # case 1, even
#         (3, 2, 3),  # case 1, even
#         (4, 3, 8),  # case 2
#         (2, 1, 4),  # case 2
#         (8, 3, 4),  # case 3
#         (4, 1, 2),  # case 3
#         (3, 8, 4),  # case 4
#         (1, 4, 2),  # case 4
#     ],
# )
def true_dln_learning_coefficient(true_rank, layer_widths, input_dim, verbose=False):
    M_list = np.array([input_dim] + list(layer_widths)) - true_rank
    indices = brute_force_search_subset(M_list, early_return=verbose)
    M_subset = M_list[indices]
    if verbose:
        print(f"M_list: {M_list}, indices: {indices}, M_subset: {M_subset}")
    M_subset_sum = np.sum(M_subset)
    ell = len(M_subset) - 1
    M = np.ceil(M_subset_sum / ell)
    a = M_subset_sum - (M - 1) * ell
    output_dim = layer_widths[-1]

    term1 = (-(true_rank**2) + true_rank * (output_dim + input_dim)) / 2
    term2 = a * (ell - a) / (4 * ell)
    term3 = -ell * (ell - 1) / 4 * (M_subset_sum / ell) ** 2
    term4 = (
        1
        / 2
        * np.sum(
            [
                M_subset[i] * M_subset[j]
                for i in range(ell + 1)
                for j in range(i + 1, ell + 1)
            ]
        )
    )
    learning_coefficient = term1 + term2 + term3 + term4
    return learning_coefficient


def _condition(indices, intlist, verbose=False):
    intlist = np.array(intlist)
    ell = len(indices) - 1
    subset = intlist[indices]
    complement = intlist[[i for i in range(len(intlist)) if i not in indices]]
    has_complement = len(complement) > 0
    # print(indices, subset, complement)
    if has_complement and not (np.max(subset) < np.min(complement)):
        if verbose:
            print(
                f"max(subset) = {np.max(subset)}, min(complement) = {np.min(complement)}"
            )
        return False
    if not (np.sum(subset) >= ell * np.max(subset)):
        if verbose:
            print(
                f"sum(subset) = {sum(subset)}, ell * max(subset) = {ell * np.max(subset)}"
            )
        return False
    if has_complement and not (np.sum(subset) < ell * np.min(complement)):
        if verbose:
            print(
                f"sum(subset) = {sum(subset)}, ell * min(complement) = {ell * np.min(complement)}"
            )
        return False
    return True


def generate_indices_subsets(length):
    indices = list(range(length))
    for size in range(1, length + 1):
        for subset in itertools.combinations(indices, size):
            subset = np.array(subset)
            yield subset


def brute_force_search_subset(intlist, early_return=False):
    candidates = []
    for indices in generate_indices_subsets(len(intlist)):
        if _condition(indices, intlist):
            if early_return:
                return indices
            candidates.append(indices)
    if len(candidates) == 0:
        raise RuntimeError("No candidates")
    if len(candidates) > 1:
        print("More than one candidate")
    return candidates[0]


def to_float_or_list(x):
    if isinstance(x, (float, int)):
        return float(x)
    elif isinstance(x, (list, tuple)):
        return [float(el) for el in x]
    elif hasattr(x, "tolist"):  # For JAX or numpy arrays
        return x.tolist()
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def test_accuracy_rrr(sampling_method, layer_widths):
    # see "The Generalization Error of Reduced Rank Regression in Bayesian Estimation", M. Aoyagi & S. Watanabe, 2004.
    # Note: RRR is kind of an odd fit for pytorch, being a two-layer no-bias linear model.
    # We train this model long enough to (hopefully) not end up in a local min
    torch.manual_seed(42)
    np.random.seed(42)
    criterion = F.mse_loss
    factors = np.array([1.0, 1.0])
    true_model = ReducedRankRegressor(layer_widths, factors)

    train_dataloader, train_data, x, y = generated_rrr_dataset(layer_widths, true_model)
    true_matrix = true_model.to_numpy_matrix()
    true_rank = np.linalg.matrix_rank(true_matrix)
    true_lc = true_dln_learning_coefficient(
        true_rank, layer_widths[1:], layer_widths[0], verbose=False
    )
    num_chains = 1
    num_draws = 10
    for mult_fact in [0.001, 1.0, 1000.0]:
        print("____")
        factors2 = torch.Tensor([mult_fact, 1 / mult_fact]).to(DEVICE)
        model = ReducedRankRegressor(layer_widths, factors2).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(500):
            optimizer.zero_grad()
            outputs = model(x.to(DEVICE))
            loss = criterion(outputs, y.to(DEVICE))
            loss.backward()
            optimizer.step()
        print(list(model.parameters()))

        for precond_factor in [0, 1, 2]:
            llc_estimator = LLCEstimator(
                num_chains=num_chains, num_draws=num_draws, n=len(train_data)
            )

            sample(
                model,
                train_dataloader,
                criterion=criterion,
                optimizer_kwargs=dict(
                    lr=0.0005,
                    num_samples=len(train_data),
                    precond=factors2.detach() ** precond_factor,
                ),
                sampling_method=sampling_method,
                num_chains=num_chains,
                num_draws=num_draws,
                callbacks=[llc_estimator],
                verbose=False,
                seed=42,
                device=DEVICE,
            )
            llc_mean = llc_estimator.sample()["llc/mean"]

            input(
                f"estimated {llc_mean:.3f} vs {true_lc:.3f} {factors2**precond_factor} {factors2} {layer_widths}"
            )
    # assert (
    #     llc_mean - 2 * llc_std_dev < true_lc < llc_mean + 2 * llc_std_dev
    # ), f"DLN case {case} estimated LLC mean {llc_mean:.3f} +- {2*llc_std_dev:.3f} vs True LC {true_lc:.3f} for (M, H, N)={(m, h, n)} using {sampling_method}"


# TODO: For models with a closed-form population loss, like DLNs:
# compare SGLD on empirical loss with SGLD on population loss (results should agree)
# SGLD on population loss should be able to get the LLC exactly correct,
# assuming beta is sufficiently high (using population loss here instead of empirical loss allows very high beta without prohibitively large training set size)

for layer_widths in [
    (1, 2, 1),
    (5, 10, 5),
    (10, 20, 20, 10),
    (20, 40, 40, 20),
    (40, 80, 80, 40),
]:
    test_accuracy_rrr(SGLD, layer_widths)
