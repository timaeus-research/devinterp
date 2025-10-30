import platform

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from devinterp.backends.default.slt.sampler import sample
from devinterp.optim import SGLD, SGMCMC
from devinterp.slt.llc import LLCEstimator
from devinterp.utils import default_nbeta, evaluate_mse, get_init_loss_multi_batch
from torch.utils.data import DataLoader, TensorDataset

# Test configuration constants
TOLERANCE_RTOL = 0.4
CONFIDENCE_MULTIPLIER = 2.5
FULL_SAMPLING_DRAWS = 400
SNAPSHOT_DRAWS = 5
LEARNING_RATE = 0.0006
LOCALIZATION = 1.0
NUM_CHAINS = 3
RANDOM_SEED = 42
NUM_SAMPLES = 1000


# not a fixture as we're generating data for several m, n combinations
# and I couldn't figure out how to fit that into the fixture mold
def generated_rrr_dataset(m, n):
    """Generate synthetic dataset for reduced rank regression testing.

    Args:
        m: Input dimension
        n: Output dimension

    Returns:
        Tuple of (dataloader, dataset, input_tensor, output_tensor)
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    x = torch.randn(NUM_SAMPLES, m)
    y = torch.randn(NUM_SAMPLES, n)
    train_data = TensorDataset(x, y)

    # Add deterministic generator
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    train_dataloader = DataLoader(train_data, batch_size=NUM_SAMPLES, shuffle=True)
    return train_dataloader, train_data, x, y


# Test case mapping for theoretical scenarios from Aoyagi & Watanabe (2004)
TEST_CASES = [
    (5, 3, 5, "case_1_odd", "General case with odd sum of dimensions"),
    (5, 4, 5, "case_1_even", "General case with even sum of dimensions"),
    (4, 3, 8, "case_2", "Input + hidden < output dimension"),
    (8, 3, 4, "case_3", "Output + hidden < input dimension"),
    (3, 8, 4, "case_4", "Input + output < hidden dimension"),
]


@pytest.mark.skip("This test is flaky and old.")
@pytest.mark.skipif(
    platform.machine() != "x86_64",
    reason=f"Differences in results between ARM and x86_64. Your arch is {platform.machine()}",
)
@pytest.mark.parametrize("sampling_method", [SGLD, SGMCMC.sgld])
@pytest.mark.parametrize(
    "m,h,n,case_name,description",
    TEST_CASES,
    ids=[f"{case[3]}_{case[0]}x{case[1]}x{case[2]}" for case in TEST_CASES],
)
@pytest.mark.parametrize("perturb", [True, False], ids=["perturbed", "unperturbed"])
def test_accuracy_rrr(
    sampling_method,
    m,
    h,
    n,
    case_name,
    description,
    perturb,
    ReducedRankRegressor,
    snapshot,
    is_snapshot_update,
):
    """Test reduced rank regression LLC estimation accuracy.

    Tests the Local Learning Coefficient (LLC) estimation for different
    dimensional configurations of reduced rank regression models.

    Based on "The Generalization Error of Reduced Rank Regression in Bayesian Estimation",
    M. Aoyagi & S. Watanabe, 2004.

    Args:
        sampling_method: SGLD or SGMCMC sampling method
        m: Input dimension
        h: Hidden/rank dimension
        n: Output dimension
        case_name: Theoretical case identifier
        description: Human-readable description of the test case
        perturb: Whether to perturb the model away from optimal parameters
        ReducedRankRegressor: Model class fixture
        snapshot: Expected LLC value for regression testing
        is_snapshot_update: Whether to run full accuracy test against theory
    """
    # Set up test data and model
    model, train_dataloader = _setup_rrr_model(m, h, n, perturb, ReducedRankRegressor)

    # Run the snapshot test first, so our random seed is consistent.
    llc_mean, llc_std_dev = do_sampling(
        sampling_method, train_dataloader, model, num_draws=SNAPSHOT_DRAWS
    )

    # Test against theoretical values when updating snapshots
    if is_snapshot_update:
        # Verify that the calculated case matches the expected case from pytest parameters
        calculated_case, _ = calc_true_lc(m, h, n)
        assert calculated_case == case_name, (
            f"Calculated theoretical case '{calculated_case}' does not match expected case '{case_name}' "
            f"for dimensions (M={m}, H={h}, N={n})"
        )

        _test_theoretical_accuracy(
            sampling_method, train_dataloader, model, m, h, n, perturb
        )

    # Always test against snapshot for consistency.
    assert llc_mean == snapshot


def _setup_rrr_model(m, h, n, perturb, ReducedRankRegressor):
    """Set up the reduced rank regression model for testing."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    criterion = F.mse_loss
    train_dataloader, train_data, x, y = generated_rrr_dataset(m, n)
    # Update dimensions based on actual data
    m = x.size(1)
    n = y.size(1)

    model = ReducedRankRegressor(m, h, n, x, y, criterion)

    if perturb:
        # We repeat the Litany Against Non-Determinism:
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        model.perturb()

    return model, train_dataloader


def _test_theoretical_accuracy(
    sampling_method, train_dataloader, model, m, h, n, perturb
):
    """Test LLC estimation against theoretical values with full sampling."""
    llc_mean, llc_std_dev = do_sampling(
        sampling_method, train_dataloader, model, num_draws=FULL_SAMPLING_DRAWS
    )
    case, true_lc = calc_true_lc(m, h, n)

    if not perturb:
        _assert_close_to_theory(
            llc_mean, llc_std_dev, true_lc, case, m, h, n, sampling_method
        )
    else:
        _assert_not_close_to_theory(
            llc_mean, llc_std_dev, true_lc, case, m, h, n, sampling_method
        )


def _assert_close_to_theory(
    llc_mean, llc_std_dev, true_lc, case, m, h, n, sampling_method
):
    """Assert that LLC estimate is close to theoretical value."""
    confidence_interval = CONFIDENCE_MULTIPLIER * llc_std_dev
    error_msg = (
        f"DLN case {case}: LLC estimate {llc_mean:.3f} ± {confidence_interval:.3f} "
        f"should be close to theoretical LC {true_lc:.3f} "
        f"for dimensions (M={m}, H={h}, N={n}) using {sampling_method}"
    )
    assert np.isclose(llc_mean, true_lc, rtol=TOLERANCE_RTOL), error_msg


def _assert_not_close_to_theory(
    llc_mean, llc_std_dev, true_lc, case, m, h, n, sampling_method
):
    """Assert that perturbed model LLC estimate is not close to theoretical value."""
    confidence_interval = CONFIDENCE_MULTIPLIER * llc_std_dev
    error_msg = (
        f"Perturbed model should not match theory. "
        f"DLN case {case}: LLC estimate {llc_mean:.3f} ± {confidence_interval:.3f} "
        f"vs theoretical LC {true_lc:.3f} "
        f"for dimensions (M={m}, H={h}, N={n}) using {sampling_method}"
    )
    assert not np.isclose(llc_mean, true_lc, rtol=TOLERANCE_RTOL), error_msg


def do_sampling(sampling_method, train_dataloader, model, num_draws):
    """Perform MCMC sampling to estimate LLC.

    Args:
        sampling_method: SGLD or SGMCMC sampling method
        train_dataloader: DataLoader for training data
        model: Model to sample from
        num_draws: Number of MCMC draws to perform

    Returns:
        Tuple of (llc_mean, llc_std_dev)
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    init_loss = get_init_loss_multi_batch(
        train_dataloader, NUM_CHAINS, model, evaluate_mse, device="cpu"
    )

    llc_estimator = LLCEstimator(
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss,
    )

    sample(
        model,
        train_dataloader,
        evaluate=evaluate_mse,
        sampling_method_kwargs=dict(
            lr=LEARNING_RATE,
            localization=LOCALIZATION,
            nbeta=default_nbeta(train_dataloader),
        ),
        sampling_method=sampling_method,
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        callbacks=[llc_estimator],
        verbose=False,
        seed=RANDOM_SEED,
    )

    results = llc_estimator.get_results()
    return results["llc/mean"], results["llc/std"]


def calc_true_lc(m, h, n):
    """Calculate theoretical learning coefficient for reduced rank regression.

    Based on the theoretical results from Aoyagi & Watanabe (2004).

    Args:
        m: Input dimension
        h: Hidden/rank dimension
        n: Output dimension

    Returns:
        Tuple of (case_name, theoretical_lc_value)
    """
    # Determine which theoretical case applies
    case_2 = m + h < n
    case_3 = n + h < m
    case_4 = m + n < h
    case_1_even = not (case_2 or case_3 or case_4) and (m + h + n) % 2 == 0
    case_1_odd = not (case_2 or case_3 or case_4) and (m + h + n) % 2 == 1

    if case_2:
        return "case_2", m * h / 2
    elif case_3:
        return "case_3", h * n / 2
    elif case_4:
        return "case_4", m * n / 2
    elif case_1_even:
        numerator = 2 * m * n + 2 * h * n + 2 * m * h - n**2 - m**2 - h**2
        return "case_1_even", numerator / 8
    elif case_1_odd:
        numerator = 1 + 2 * m * n + 2 * h * n + 2 * m * h - n**2 - m**2 - h**2
        return "case_1_odd", numerator / 8
    else:
        raise ValueError(
            f"Unknown theoretical case for dimensions (M={m}, H={h}, N={n})"
        )


# TODO:
# Scale up these estimates like in Furman & Lau (2024), also to DLNs more generally
#
# For models with a closed-form population loss, like DLNs:
# compare SGLD on empirical loss with SGLD on population loss (results should agree)
# SGLD on population loss should be able to get the LLC exactly correct,
# assuming beta is sufficiently high (using population loss here instead of empirical loss allows very high beta without prohibitively large training set size)
