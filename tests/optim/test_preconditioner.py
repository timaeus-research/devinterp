import pytest
import torch
from devinterp.optim.preconditioner import (
    CompositePreconditioner,
    IdentityPreconditioner,
    MaskPreconditioner,
    NHTPreconditioning,
    PreconditionerCoefs,
    RMSpropPreconditioner,
)


def test_preconditioner_coefs_multiplication():
    coef1 = PreconditionerCoefs(2.0, 3.0, 4.0, 5.0, None)
    coef2 = PreconditionerCoefs(1.0, 2.0, 3.0, 4.0, torch.tensor([1.0, 2.0]))

    result = coef1.combine_with(coef2)
    assert result.grad_coef == 2.0
    assert result.prior_coef == 6.0
    assert result.noise_coef == 12.0
    assert result.overall_coef == 20.0
    assert torch.equal(result.grad_correction, torch.tensor([1.0, 2.0]))

    result_2 = PreconditionerCoefs.combine(coef1, coef2)
    assert result_2.grad_coef == 2.0
    assert result_2.prior_coef == 6.0
    assert result_2.noise_coef == 12.0
    assert result_2.overall_coef == 20.0
    assert torch.equal(result_2.grad_correction, torch.tensor([1.0, 2.0]))


def test_identity_preconditioner():
    precond = IdentityPreconditioner()
    param = torch.randn(3, 4)
    grad = torch.randn(3, 4)
    state = {}

    coefs = precond.get_coefficients(param, grad, state)
    assert coefs.grad_coef == 1.0
    assert coefs.prior_coef == 1.0
    assert coefs.noise_coef == 1.0
    assert coefs.overall_coef == 1.0
    assert coefs.grad_correction is None


def test_mask_preconditioner():
    mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    precond = MaskPreconditioner([mask])
    param = torch.randn(2, 2)
    grad = torch.randn(2, 2)
    state = {"param_idx": 0}

    coefs = precond.get_coefficients(param, grad, state)
    assert coefs.grad_coef == 1.0
    assert coefs.prior_coef == 1.0
    assert coefs.noise_coef == 1.0
    assert torch.equal(coefs.overall_coef, mask)
    assert coefs.grad_correction is None


def test_composite_preconditioner():
    # Test optimization that returns identity when no non-identity preconditioners
    precond = CompositePreconditioner(
        [IdentityPreconditioner(), IdentityPreconditioner()]
    )
    assert isinstance(precond, IdentityPreconditioner)

    # Test single preconditioner optimization
    mask_precond = MaskPreconditioner([torch.tensor(0.5)])
    precond = CompositePreconditioner([IdentityPreconditioner(), mask_precond])
    assert precond is mask_precond


def test_composite_preconditioner_with_identity():
    # Test that CompositePreconditioner returns IdentityPreconditioner when only IdentityPreconditioners are provided
    composite = CompositePreconditioner(
        [IdentityPreconditioner(), IdentityPreconditioner()]
    )
    assert isinstance(composite, IdentityPreconditioner)


def test_composite_preconditioner_empty():
    # Test that CompositePreconditioner returns IdentityPreconditioner when given empty list
    assert isinstance(CompositePreconditioner([]), IdentityPreconditioner)


def test_rmsprop_preconditioner():
    precond = RMSpropPreconditioner(alpha=0.9, eps=1e-8)
    param = torch.ones(2, 2)
    grad = torch.ones(2, 2)
    state = {}

    coefs = precond.get_coefficients(param, grad, state)
    assert "square_avg" in state
    assert torch.allclose(state["square_avg"], torch.ones_like(param) * 0.1)

    # Test that coefficients are properly computed
    expected_precond = 1.0 / (torch.sqrt(state["square_avg"]) + 1e-8)
    assert torch.allclose(coefs.grad_coef, expected_precond)
    assert torch.allclose(coefs.prior_coef, expected_precond)
    assert torch.allclose(coefs.noise_coef, torch.sqrt(expected_precond))
    assert coefs.overall_coef == 1.0


def test_nht_preconditioning():
    precond = NHTPreconditioning(diffusion_factor=0.01)
    param = torch.ones(2, 2)
    grad = torch.ones(2, 2)
    state = {}

    coefs = precond.get_coefficients(param, grad, state)
    assert "thermostat" in state
    assert "momentum" in state
    assert state["momentum"].shape == param.shape

    # Test coefficient properties
    assert coefs.grad_coef == 1.0
    assert coefs.prior_coef == 1.0
    assert torch.allclose(coefs.noise_coef, torch.sqrt(torch.tensor(0.02)))
    assert coefs.overall_coef == 1.0
    assert coefs.grad_correction is not None
    assert coefs.grad_correction.shape == param.shape
