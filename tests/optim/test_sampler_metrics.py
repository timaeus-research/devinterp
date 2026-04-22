"""
Tests for sampler metrics accuracy in SGLD and SGMCMC.

Verifies that tracked metrics accurately reflect the actual parameter update
components. Each test isolates a single component with hand-computed expected
values (using the 3-4-5 Pythagorean triple for clean norms).

All tests use float64 to ensure exact equality between test expectations
and the float32 metric accumulators for these small integer-friendly values.
"""

import warnings

import pytest
import torch
import torch.nn as nn
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgmcmc import SGMCMC

pytestmark = [
    # SGLD emits DeprecationWarning on construction; SGMCMC warns about
    # nbeta=1 and non-default noise_level. These are expected in test
    # configs and would clutter output.
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:.*nbeta.*"),
    pytest.mark.filterwarnings("ignore:.*noise_level.*"),
]


def _make_optimizer(sampler_cls, params, **kwargs):
    """Create optimizer, suppressing expected warnings during construction."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sampler_cls(params, **kwargs)


@pytest.fixture(params=[SGLD, SGMCMC.sgld], ids=["sgld", "sgmcmc"])
def sampler_cls(request):
    return request.param


# ---------------------------------------------------------------------------
# Component isolation tests
#
# Each verifies one metric component with all others disabled.
# Expected values are literal, derived from the SGLD update equation:
#   scaled_grad = (lr/2) * nbeta * grad
#   localization = (lr/2) * gamma * (w - w0)
#   weight_decay = (lr/2) * lambda * w
#   noise = sqrt(lr) * eta
# ---------------------------------------------------------------------------


class TestScaledGrad:
    def test_metric_value(self, sampler_cls):
        """scaled_grad = ||(lr/2) * nbeta * grad||_2"""
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls, [w], lr=1.0, nbeta=2.0, noise_level=0.0, save_metrics=True
        )
        w.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        # (1.0/2) * 2.0 * [3, 4] = [3, 4], norm = 5
        torch.testing.assert_close(m.scaled_grad, torch.tensor([5.0]), atol=0, rtol=0)

    def test_other_components_zero(self, sampler_cls):
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls, [w], lr=1.0, nbeta=2.0, noise_level=0.0, save_metrics=True
        )
        w.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        torch.testing.assert_close(m.localization, torch.tensor([0.0]), atol=0, rtol=0)
        torch.testing.assert_close(m.weight_decay, torch.tensor([0.0]), atol=0, rtol=0)
        torch.testing.assert_close(m.noise, torch.tensor([0.0]), atol=0, rtol=0)


class TestLocalization:
    def test_metric_value(self, sampler_cls):
        """localization = ||(lr/2) * gamma * (w - w0)||_2"""
        w = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls,
            [w],
            lr=2.0,
            nbeta=2.0,
            localization=1.0,
            noise_level=0.0,
            save_metrics=True,
        )
        # Shift w away from w0=[0,0] to create a known distance
        w.data = torch.tensor([3.0, 4.0], dtype=torch.float64)
        w.grad = torch.zeros(2, dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        # (2.0/2) * 1.0 * [3, 4] = [3, 4], norm = 5
        torch.testing.assert_close(m.localization, torch.tensor([5.0]), atol=0, rtol=0)


class TestWeightDecay:
    def test_metric_value(self, sampler_cls):
        """weight_decay = ||(lr/2) * lambda * w||_2"""
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls,
            [w],
            lr=2.0,
            nbeta=2.0,
            weight_decay=1.0,
            noise_level=0.0,
            save_metrics=True,
        )
        w.grad = torch.zeros(2, dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        # (2.0/2) * 1.0 * [3, 4] = [3, 4], norm = 5
        torch.testing.assert_close(m.weight_decay, torch.tensor([5.0]), atol=0, rtol=0)


class TestNoise:
    def test_metric_equals_param_change(self, sampler_cls):
        """When only noise is active, noise metric = ||param change||_2."""
        w = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        opt = _make_optimizer(sampler_cls, [w], lr=0.5, nbeta=2.0, save_metrics=True)
        w.grad = torch.zeros(2, dtype=torch.float64)
        w_before = w.data.clone()
        opt.step(noise_generator=torch.Generator().manual_seed(42))

        m = opt.get_metrics()
        actual_change_norm = (w.data - w_before).norm().unsqueeze(0)
        # Metrics accumulate in float32, so allow float32-level tolerance
        torch.testing.assert_close(
            m.noise.to(torch.float64),
            actual_change_norm,
            atol=1e-6,
            rtol=0,
        )

    def test_zero_when_noise_disabled(self, sampler_cls):
        w = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls, [w], lr=0.5, nbeta=2.0, noise_level=0.0, save_metrics=True
        )
        w.grad = torch.zeros(2, dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(42))

        m = opt.get_metrics()
        torch.testing.assert_close(m.noise, torch.tensor([0.0]), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Combined and structural tests
# ---------------------------------------------------------------------------


class TestAllComponents:
    def test_all_metrics_correct(self, sampler_cls):
        """With all components active, each metric matches its expected value."""
        w = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls,
            [w],
            lr=2.0,
            nbeta=2.0,
            localization=1.0,
            weight_decay=1.0,
            noise_level=0.0,
            save_metrics=True,
        )
        w.data = torch.tensor([3.0, 4.0], dtype=torch.float64)
        w.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        # scaled_grad: (2/2)*2*[3,4] = [6,8], norm = 10
        torch.testing.assert_close(m.scaled_grad, torch.tensor([10.0]), atol=0, rtol=0)
        # localization: (2/2)*1*[3,4] = [3,4], norm = 5
        torch.testing.assert_close(m.localization, torch.tensor([5.0]), atol=0, rtol=0)
        # weight_decay: (2/2)*1*[3,4] = [3,4], norm = 5
        torch.testing.assert_close(m.weight_decay, torch.tensor([5.0]), atol=0, rtol=0)


class TestNumel:
    def test_counts_parameter_elements(self, sampler_cls):
        w = nn.Parameter(torch.randn(3, 4, dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls, [w], lr=0.5, nbeta=2.0, noise_level=0.0, save_metrics=True
        )
        w.grad = torch.zeros_like(w)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        assert m.numel == 12

    def test_sums_across_params(self, sampler_cls):
        w1 = nn.Parameter(torch.randn(3, dtype=torch.float64))
        w2 = nn.Parameter(torch.randn(5, dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls, [w1, w2], lr=0.5, nbeta=2.0, noise_level=0.0, save_metrics=True
        )
        w1.grad = torch.zeros_like(w1)
        w2.grad = torch.zeros_like(w2)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        assert m.numel == 8

    @pytest.mark.parametrize(
        "factory",
        [
            lambda p, m, **kw: _make_optimizer(SGLD, [{"params": p, "mask": m}], **kw),
            lambda p, m, **kw: _make_optimizer(SGMCMC.sgld, p, mask=m, **kw),
            lambda p, m, **kw: _make_optimizer(SGMCMC.rmsprop_sgld, p, mask=m, **kw),
        ],
        ids=["sgld", "sgmcmc", "rmsprop"],
    )
    def test_counts_masked_in_only(self, factory):
        w = nn.Parameter(torch.randn(6, dtype=torch.float64))
        mask = torch.tensor([True, True, False, False, True, False])
        opt = factory([w], mask, lr=0.5, nbeta=2.0, noise_level=0.0, save_metrics=True)
        w.grad = torch.zeros_like(w)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        assert m.numel == 3


def _make_masked_sgld(params: list[nn.Parameter], mask, **kwargs):
    """SGLD: mask via param group dict."""
    return _make_optimizer(SGLD, [{"params": params, "mask": mask}], **kwargs)


def _make_masked_sgmcmc(params: list[nn.Parameter], mask, **kwargs):
    """SGMCMC: mask via constructor kwarg (→ MaskPreconditioner)."""
    return _make_optimizer(SGMCMC.sgld, params, mask=mask, **kwargs)


def _make_masked_rmsprop(params: list[nn.Parameter], mask, **kwargs):
    """SGMCMC.rmsprop_sgld: mask via constructor kwarg (→ CompositePreconditioner)."""
    return _make_optimizer(SGMCMC.rmsprop_sgld, params, mask=mask, **kwargs)


@pytest.fixture(
    params=[_make_masked_sgld, _make_masked_sgmcmc, _make_masked_rmsprop],
    ids=["sgld", "sgmcmc", "rmsprop"],
)
def make_masked_optimizer(request):
    return request.param


class TestMask:
    @pytest.mark.parametrize(
        "factory, expected",
        [
            # (ε/2)·nβ·grad[0] = (1/2)·2·3 = 3
            (_make_masked_sgld, 3.0),
            (_make_masked_sgmcmc, 3.0),
            # RMSProp first step (α=0.99, ε_rms=0.1):
            #   v = (1-α)·g₀² = 0.01·9 = 0.09,  G = 1/(√v+ε) = 1/0.4 = 2.5
            #   (ε/2)·nβ·G·g₀ = (1/2)·2·2.5·3 = 7.5
            (_make_masked_rmsprop, 7.5),
        ],
        ids=["sgld", "sgmcmc", "rmsprop"],
    )
    def test_mask_restricts_scaled_grad(self, factory, expected):
        """Mask zeroes out second element, so only first contributes."""
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        mask = torch.tensor([True, False])
        opt = factory(
            [w],
            mask,
            lr=1.0,
            nbeta=2.0,
            noise_level=0.0,
            save_metrics=True,
        )
        w.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        torch.testing.assert_close(
            m.scaled_grad, torch.tensor([expected]), atol=0, rtol=0
        )

    def test_masked_params_unchanged(self, make_masked_optimizer):
        """Masked parameter elements should not change."""
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        mask = torch.tensor([True, False])
        opt = make_masked_optimizer(
            [w],
            mask,
            lr=1.0,
            nbeta=2.0,
            save_metrics=True,
        )
        w.grad = torch.randn(2, dtype=torch.float64)
        w_before = w.data.clone()
        opt.step(noise_generator=torch.Generator().manual_seed(42))

        torch.testing.assert_close(w.data[1:], w_before[1:], atol=0, rtol=0)


class TestGetMetricsInterface:
    def test_raises_when_not_enabled(self, sampler_cls):
        w = nn.Parameter(torch.tensor([1.0]))
        opt = _make_optimizer(sampler_cls, [w], lr=0.5, nbeta=2.0, save_metrics=False)
        with pytest.raises(RuntimeError, match="Metrics not enabled"):
            opt.get_metrics()

    def test_returns_cpu_tensors(self, sampler_cls):
        w = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        opt = _make_optimizer(sampler_cls, [w], lr=0.5, nbeta=2.0, save_metrics=True)
        w.grad = torch.zeros(1, dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        assert m.scaled_grad.device == torch.device("cpu")
        assert m.noise.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# unscaled_grad, distance, and dot product tests
# ---------------------------------------------------------------------------


class TestUnscaledGrad:
    def test_equals_scaled_grad_for_sgld(self, sampler_cls):
        """SGLD has no preconditioner, so unscaled_grad == scaled_grad."""
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls, [w], lr=1.0, nbeta=2.0, noise_level=0.0, save_metrics=True
        )
        w.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        torch.testing.assert_close(m.unscaled_grad, m.scaled_grad, atol=0, rtol=0)

    def test_differs_from_scaled_grad_with_rmsprop(self):
        """RMSprop preconditioner makes scaled_grad != unscaled_grad."""
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        opt = _make_optimizer(
            SGMCMC.rmsprop_sgld,
            [w],
            lr=1.0,
            nbeta=2.0,
            noise_level=0.0,
            save_metrics=True,
        )
        w.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        # unscaled_grad = ||(lr/2)*nbeta*grad|| = ||(0.5)*2*[3,4]|| = ||[3,4]|| = 5
        torch.testing.assert_close(m.unscaled_grad, torch.tensor([5.0]), atol=0, rtol=0)
        # scaled_grad includes the RMSprop preconditioner, so != 5
        assert not torch.equal(m.scaled_grad, m.unscaled_grad)


class TestDistance:
    def test_distance_finite(self, sampler_cls):
        """distance metric is finite and non-negative."""
        w = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls,
            [w],
            lr=2.0,
            nbeta=2.0,
            localization=1.0,
            noise_level=0.0,
            save_metrics=True,
        )
        w.data = torch.tensor([3.0, 4.0], dtype=torch.float64)
        w.grad = torch.zeros(2, dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        assert torch.isfinite(m.distance)
        assert m.distance.item() >= 0

    def test_distance_zero_at_init(self, sampler_cls):
        """distance = 0 when w hasn't moved from w0."""
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls,
            [w],
            lr=2.0,
            nbeta=2.0,
            localization=1.0,
            noise_level=0.0,
            save_metrics=True,
        )
        # Don't move w from init, only set gradient
        w.grad = torch.zeros(2, dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        # No gradient, no noise => w didn't move, distance = 0
        torch.testing.assert_close(m.distance, torch.tensor([0.0]), atol=0, rtol=0)

    def test_distance_without_localization(self, sampler_cls):
        """distance is tracked even when localization=0 (if save_metrics=True)."""
        w = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls,
            [w],
            lr=2.0,
            nbeta=2.0,
            localization=0.0,
            noise_level=0.0,
            save_metrics=True,
        )
        w.data = torch.tensor([3.0, 4.0], dtype=torch.float64)
        w.grad = torch.zeros(2, dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        assert m.distance.item() > 0

    def test_distance_exact_value(self, sampler_cls):
        """distance = ||w - w0|| measured before the step update.

        With w0=[0,0], w=[3,4]: distance = ||[3,4] - [0,0]|| = 5.
        """
        w = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls,
            [w],
            lr=2.0,
            nbeta=2.0,
            localization=1.0,
            noise_level=0.0,
            save_metrics=True,
        )
        w.data = torch.tensor([3.0, 4.0], dtype=torch.float64)
        w.grad = torch.zeros(2, dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        torch.testing.assert_close(m.distance, torch.tensor([5.0]), atol=0, rtol=0)


class TestDotProducts:
    def test_dot_products_with_known_vectors(self, sampler_cls):
        """Verify dot products with hand-computed values.

        With lr=2, nbeta=2, localization=1, weight_decay=1, noise=0:
          scaled_grad = (2/2)*2*[3,4] = [6,8]
          loc = (2/2)*1*[3,4] = [3,4]  (w-w0 = [3,4])
          wd = (2/2)*1*[3,4] = [3,4]   (w = [3,4])
          prior = loc + wd = [6,8]
          noise = 0

        dot_grad_prior = <[6,8], [6,8]> = 36+64 = 100
        dot_grad_noise = 0
        dot_prior_noise = 0
        """
        w = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls,
            [w],
            lr=2.0,
            nbeta=2.0,
            localization=1.0,
            weight_decay=1.0,
            noise_level=0.0,
            save_metrics=True,
        )
        w.data = torch.tensor([3.0, 4.0], dtype=torch.float64)
        w.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        torch.testing.assert_close(
            m.dot_grad_prior, torch.tensor([100.0]), atol=0, rtol=0
        )
        torch.testing.assert_close(
            m.dot_grad_noise, torch.tensor([0.0]), atol=0, rtol=0
        )
        torch.testing.assert_close(
            m.dot_prior_noise, torch.tensor([0.0]), atol=0, rtol=0
        )

    def test_dot_products_orthogonal_components(self, sampler_cls):
        """When grad and prior are orthogonal, dot_grad_prior = 0."""
        w = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls,
            [w],
            lr=2.0,
            nbeta=2.0,
            localization=1.0,
            noise_level=0.0,
            save_metrics=True,
        )
        # w - w0 = [1, 0] (localization direction)
        w.data = torch.tensor([1.0, 0.0], dtype=torch.float64)
        # grad = [0, 1] (orthogonal to localization)
        w.grad = torch.tensor([0.0, 1.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        # scaled_grad = (2/2)*2*[0,1] = [0,2]
        # loc = (2/2)*1*[1,0] = [1,0]
        # dot = 0*1 + 2*0 = 0
        torch.testing.assert_close(
            m.dot_grad_prior, torch.tensor([0.0]), atol=0, rtol=0
        )

    def test_dot_products_with_noise(self, sampler_cls):
        """Dot products involving noise should be nonzero with nonzero noise."""
        w = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls,
            [w],
            lr=2.0,
            nbeta=2.0,
            localization=1.0,
            save_metrics=True,
        )
        w.data = torch.tensor([3.0, 4.0], dtype=torch.float64)
        w.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(42))

        m = opt.get_metrics()
        # With noise_level=1 (default), noise terms should be nonzero
        assert m.dot_grad_noise.item() != 0
        assert m.dot_prior_noise.item() != 0

    def test_dot_product_matches_manual_computation(self, sampler_cls):
        """Dot products match manual computation from reconstructed vectors."""
        torch.manual_seed(0)
        w = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))
        lr, nbeta, loc, wd, noise_level = 2.0, 2.0, 1.0, 0.5, 1.0

        opt = _make_optimizer(
            sampler_cls,
            [w],
            lr=lr,
            nbeta=nbeta,
            localization=loc,
            weight_decay=wd,
            noise_level=noise_level,
            save_metrics=True,
        )
        w.data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        w.grad = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)

        gen = torch.Generator().manual_seed(99)
        opt.step(noise_generator=gen)

        m = opt.get_metrics()

        # Reconstruct the component vectors manually
        half_lr = lr / 2
        _sg = half_lr * nbeta * torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        _loc = half_lr * loc * torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        _wd = half_lr * wd * torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        _prior = _loc + _wd

        gen2 = torch.Generator().manual_seed(99)
        raw_noise = torch.normal(0.0, noise_level, size=(3,), generator=gen2)
        _noise = raw_noise * (lr**0.5)

        expected_dot_gp = torch.dot(_sg.float(), _prior.float()).unsqueeze(0)
        expected_dot_gn = torch.dot(_sg.float(), _noise.float()).unsqueeze(0)
        expected_dot_pn = torch.dot(_prior.float(), _noise.float()).unsqueeze(0)

        torch.testing.assert_close(m.dot_grad_prior, expected_dot_gp, atol=1e-5, rtol=0)
        torch.testing.assert_close(m.dot_grad_noise, expected_dot_gn, atol=1e-5, rtol=0)
        torch.testing.assert_close(
            m.dot_prior_noise, expected_dot_pn, atol=1e-5, rtol=0
        )

    def test_dot_products_accumulate_across_params(self, sampler_cls):
        """Dot products sum across multiple parameters in a single param group.

        Two parameters [w1, w2] share one optimizer group. The step() loop
        calls _update_metrics for each param, accumulating into the same
        group["metrics"] object. Cross-group aggregation (Metrics.aggregate)
        is tested separately in test_metrics_multi_gpu.

        w0_1=[1,0], w_1=[3,0], grad_1=[3,0]:
          sg_1 = (2/2)*2*[3,0] = [6,0]
          loc_grad_1 = gamma*(w-w0) = 1*[2,0]; prior_1 = (2/2)*1*[2,0] = [2,0]
          dot_1 = 6*2 + 0*0 = 12

        w0_2=[0,1], w_2=[0,4], grad_2=[0,4]:
          sg_2 = (2/2)*2*[0,4] = [0,8]
          loc_grad_2 = 1*[0,3]; prior_2 = (2/2)*1*[0,3] = [0,3]
          dot_2 = 0*0 + 8*3 = 24

        Total dot_grad_prior = 36.
        """
        w1 = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float64))
        w2 = nn.Parameter(torch.tensor([0.0, 1.0], dtype=torch.float64))
        opt = _make_optimizer(
            sampler_cls,
            [w1, w2],
            lr=2.0,
            nbeta=2.0,
            localization=1.0,
            noise_level=0.0,
            save_metrics=True,
        )
        w1.data = torch.tensor([3.0, 0.0], dtype=torch.float64)
        w2.data = torch.tensor([0.0, 4.0], dtype=torch.float64)
        w1.grad = torch.tensor([3.0, 0.0], dtype=torch.float64)
        w2.grad = torch.tensor([0.0, 4.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        torch.testing.assert_close(
            m.dot_grad_prior, torch.tensor([36.0]), atol=0, rtol=0
        )

    def test_dot_products_with_rmsprop(self):
        """Dot products work with RMSprop preconditioner."""
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        opt = _make_optimizer(
            SGMCMC.rmsprop_sgld,
            [w],
            lr=1.0,
            nbeta=2.0,
            localization=1.0,
            noise_level=0.0,
            save_metrics=True,
        )
        w.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        m = opt.get_metrics()
        # With RMSprop, the preconditioner scales the components differently.
        # The dot product should still be well-defined and finite.
        assert torch.isfinite(m.dot_grad_prior)
        # Since noise=0, noise-related dots should be 0
        torch.testing.assert_close(
            m.dot_grad_noise, torch.tensor([0.0]), atol=0, rtol=0
        )


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_metrics_on_gpu():
    w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64, device="cuda"))
    opt = _make_optimizer(
        SGMCMC.sgld, [w], lr=1.0, nbeta=2.0, noise_level=0.0, save_metrics=True
    )
    w.grad = torch.tensor([3.0, 4.0], dtype=torch.float64, device="cuda")
    opt.step(noise_generator=torch.Generator("cuda").manual_seed(0))

    m = opt.get_metrics()
    assert m.scaled_grad.device == torch.device("cpu")
    torch.testing.assert_close(m.scaled_grad, torch.tensor([5.0]), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.only_multi_gpu
def test_metrics_multi_gpu():
    """Metrics aggregate correctly with params on different GPUs.

    No torch.distributed initialization is needed here: we're just placing
    individual parameters on separate devices, not using DDP or collective ops.
    """
    w0 = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64, device="cuda:0"))
    w1 = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64, device="cuda:1"))
    opt = _make_optimizer(
        SGMCMC.sgld,
        [{"params": [w0]}, {"params": [w1]}],
        lr=1.0,
        nbeta=2.0,
        noise_level=0.0,
        save_metrics=True,
    )
    w0.grad = torch.tensor([3.0, 4.0], dtype=torch.float64, device="cuda:0")
    w1.grad = torch.tensor([3.0, 4.0], dtype=torch.float64, device="cuda:1")
    opt.step(noise_generator=torch.Generator("cuda").manual_seed(0))

    m = opt.get_metrics()
    assert m.scaled_grad.device == torch.device("cpu")
    # Each group contributes norm 5, combined: sqrt(5^2 + 5^2) = sqrt(50)
    expected = torch.tensor([50.0]).sqrt()
    torch.testing.assert_close(m.scaled_grad, expected, atol=0, rtol=0)
    assert m.numel == 4
