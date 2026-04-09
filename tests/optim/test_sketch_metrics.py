"""Tests for count sketch integration with SGMCMC and legacy SGLD optimizers.

Verifies that sketches tracked during optimizer steps correctly reflect the
actual parameter update components. The key cross-check (Andy's test):
the L2 norm of a sketched vector should approximate the exact L2 norm
recorded by Metrics.
"""

import warnings

import pytest
import torch
import torch.nn as nn
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgmcmc import SGMCMC
from devinterp.optim.sketch import SKETCH_QUANTITIES, SketchBuffer

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:.*nbeta.*"),
    pytest.mark.filterwarnings("ignore:.*noise_level.*"),
]


def _make_optimizer(factory, params, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return factory(params, **kwargs)


class TestSketchInit:
    def test_creates_sketch_buffer(self):
        w = nn.Parameter(torch.randn(10, dtype=torch.float64))
        opt = _make_optimizer(SGMCMC.sgld, [w], lr=0.5, nbeta=2.0, sketch_dim=32)
        assert opt.save_sketches
        assert isinstance(opt._sketch_buf, SketchBuffer)

    def test_correct_output_dim(self):
        w = nn.Parameter(torch.randn(10, dtype=torch.float64))
        opt = _make_optimizer(SGMCMC.sgld, [w], lr=0.5, nbeta=2.0, sketch_dim=64)
        assert opt._sketch_buf.scaled_grad.shape == (64,)

    def test_not_enabled_by_default(self):
        w = nn.Parameter(torch.randn(10, dtype=torch.float64))
        opt = _make_optimizer(SGMCMC.sgld, [w], lr=0.5, nbeta=2.0)
        assert not opt.save_sketches
        assert opt._sketch is None
        assert opt._sketch_buf is None


class TestGetSketches:
    def test_raises_when_not_enabled(self):
        w = nn.Parameter(torch.randn(5))
        opt = _make_optimizer(SGMCMC.sgld, [w], lr=0.5, nbeta=2.0)
        with pytest.raises(RuntimeError, match="Sketches not enabled"):
            opt.get_sketches()

    def test_returns_all_quantities(self):
        w = nn.Parameter(torch.randn(10, dtype=torch.float64))
        opt = _make_optimizer(
            SGMCMC.sgld, [w], lr=0.5, nbeta=2.0, noise_level=0.0, sketch_dim=32
        )
        w.grad = torch.randn_like(w)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        sketches = opt.get_sketches()
        for q in SKETCH_QUANTITIES:
            t = getattr(sketches, q)
            assert t.shape == (32,), q
            assert t.device == torch.device("cpu"), q

    def test_correct_output_dim(self):
        w = nn.Parameter(torch.randn(20, dtype=torch.float64))
        opt = _make_optimizer(
            SGMCMC.sgld, [w], lr=0.5, nbeta=2.0, noise_level=0.0, sketch_dim=128
        )
        w.grad = torch.randn_like(w)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        sketches = opt.get_sketches()
        assert sketches.scaled_grad.shape == (128,)


class TestSketchZeroComponents:
    """When a component is zero, its sketch should also be zero."""

    def test_noise_zero_when_disabled(self):
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        opt = _make_optimizer(
            SGMCMC.sgld, [w], lr=1.0, nbeta=2.0, noise_level=0.0, sketch_dim=32
        )
        w.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        sketches = opt.get_sketches()
        torch.testing.assert_close(sketches.noise, torch.zeros(32), atol=0, rtol=0)

    def test_localization_zero_without_prior(self):
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        opt = _make_optimizer(
            SGMCMC.sgld, [w], lr=1.0, nbeta=2.0, noise_level=0.0, sketch_dim=32
        )
        w.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        sketches = opt.get_sketches()
        torch.testing.assert_close(
            sketches.localization, torch.zeros(32), atol=0, rtol=0
        )
        torch.testing.assert_close(
            sketches.weight_decay, torch.zeros(32), atol=0, rtol=0
        )

    def test_grad_zero_when_grad_is_zero(self):
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        opt = _make_optimizer(
            SGMCMC.sgld,
            [w],
            lr=1.0,
            nbeta=2.0,
            localization=1.0,
            noise_level=0.0,
            sketch_dim=32,
        )
        w.data = torch.tensor([3.0, 4.0], dtype=torch.float64)
        w.grad = torch.zeros(2, dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        sketches = opt.get_sketches()
        torch.testing.assert_close(
            sketches.scaled_grad, torch.zeros(32), atol=0, rtol=0
        )
        torch.testing.assert_close(
            sketches.unscaled_grad, torch.zeros(32), atol=0, rtol=0
        )


class TestSketchNormApproximatesMetricNorm:
    """The core cross-check: ||sketch(v)|| ≈ ||v|| (the exact metric norm).

    E[||Sv||²] = ||v||² with std ∝ ||v||²/√sketch_dim, so the relative
    standard deviation of ||Sv|| around ||v|| is O(1/√sketch_dim). At
    sketch_dim=64 that's ~12.5%, so we use 20-25% tolerances (roughly 2σ).

    Assertions skip quantities whose exact norm is near zero to avoid
    dividing by zero in the relative comparison — when the true norm is
    zero (e.g. no prior active), the sketch is exactly zero too, which
    is covered by TestSketchZeroComponents.
    """

    @pytest.mark.parametrize(
        "factory",
        [SGMCMC.sgld, SGMCMC.rmsprop_sgld, SGLD],
        ids=["sgmcmc_sgld", "rmsprop", "legacy_sgld"],
    )
    def test_scaled_grad_norm(self, factory):
        """Sketch norm of scaled_grad ≈ exact scaled_grad norm from Metrics."""
        w = nn.Parameter(torch.randn(200, dtype=torch.float64))
        opt = _make_optimizer(
            factory,
            [w],
            lr=0.5,
            nbeta=2.0,
            localization=1.0,
            noise_level=0.0,
            save_metrics=True,
            sketch_dim=64,
        )
        w.grad = torch.randn_like(w)
        opt.step(noise_generator=torch.Generator().manual_seed(42))

        metrics = opt.get_metrics()
        sketches = opt.get_sketches()

        exact_norm = metrics.scaled_grad.item()
        sketch_norm = sketches.scaled_grad.norm().item()

        assert exact_norm > 0
        assert sketch_norm == pytest.approx(exact_norm, rel=0.2)

    def test_all_quantities_over_multiple_steps(self):
        """All sketch norms track metric norms over several steps."""
        torch.manual_seed(42)
        w = nn.Parameter(torch.randn(200, dtype=torch.float64))
        opt = _make_optimizer(
            SGMCMC.sgld,
            [w],
            lr=0.5,
            nbeta=2.0,
            localization=1.0,
            weight_decay=0.5,
            save_metrics=True,
            sketch_dim=64,
        )

        gen = torch.Generator().manual_seed(99)
        for _ in range(5):
            w.grad = torch.randn_like(w)
            opt.step(noise_generator=gen)

            metrics = opt.get_metrics()
            sketches = opt.get_sketches()

            for q in ("scaled_grad", "unscaled_grad", "localization", "weight_decay"):
                exact = getattr(metrics, q).item()
                sketch_norm = getattr(sketches, q).norm().item()
                if exact > 1e-8:
                    assert sketch_norm == pytest.approx(exact, rel=0.25), q


class TestSketchDotProductApproximatesMetricDotProduct:
    """The dot product cross-check: ⟨S(v), S(w)⟩ ≈ ⟨v, w⟩.

    Since all sketches share the same hash, inner products in sketch space
    approximate inner products in the original space. This is a stronger
    check than norms alone — it verifies that the sketches are mutually
    coherent, not just individually well-scaled.

    Dot product estimation has higher variance than norm estimation. The
    absolute std dev is O(||v||·||w||/√k), so the error bound scales with
    the product of norms, not with the dot product itself. We use an
    absolute tolerance of ``margin * ||v|| * ||w|| / sqrt(k)`` which
    corresponds to a ~3σ bound.
    """

    PRIOR = "prior"

    DOT_PRODUCT_FIELDS = {
        "dot_grad_prior": ("scaled_grad", PRIOR),
        "dot_grad_noise": ("scaled_grad", "noise"),
        "dot_prior_noise": (PRIOR, "noise"),
    }

    @staticmethod
    def _get_sketch_vector(sketches: SketchBuffer, field: str) -> torch.Tensor:
        if field == "prior":
            return sketches.localization + sketches.weight_decay
        return getattr(sketches, field)

    @classmethod
    def _sketch_dot_and_norms(
        cls, sketches: SketchBuffer, field_a: str, field_b: str
    ) -> tuple[float, float, float]:
        a = cls._get_sketch_vector(sketches, field_a)
        b = cls._get_sketch_vector(sketches, field_b)
        return torch.dot(a, b).item(), a.norm().item(), b.norm().item()

    @pytest.mark.parametrize(
        "factory",
        [SGMCMC.sgld, SGMCMC.rmsprop_sgld, SGLD],
        ids=["sgmcmc_sgld", "rmsprop", "legacy_sgld"],
    )
    def test_dot_products_over_multiple_steps(self, factory):
        sketch_dim = 2048
        margin = 4.0

        torch.manual_seed(42)
        w = nn.Parameter(torch.randn(200, dtype=torch.float64))
        opt = _make_optimizer(
            factory,
            [w],
            lr=0.5,
            nbeta=2.0,
            localization=1.0,
            weight_decay=0.5,
            save_metrics=True,
            sketch_dim=sketch_dim,
        )

        gen = torch.Generator().manual_seed(99)
        for step_i in range(5):
            w.grad = torch.randn_like(w)
            opt.step(noise_generator=gen)

            metrics = opt.get_metrics()
            sketches = opt.get_sketches()

            for dot_field, (a, b) in self.DOT_PRODUCT_FIELDS.items():
                exact = getattr(metrics, dot_field).item()
                approx, norm_a, norm_b = self._sketch_dot_and_norms(sketches, a, b)
                atol = margin * norm_a * norm_b / (sketch_dim**0.5)
                assert abs(approx - exact) < atol, (
                    f"{dot_field} step={step_i}: "
                    f"|{approx:.4f} - {exact:.4f}| = {abs(approx - exact):.4f} > {atol:.4f}"
                )


class TestSketchWithMultipleParams:
    def test_accumulates_across_params(self):
        """Sketches accumulate correctly across multiple parameters."""
        w1 = nn.Parameter(torch.randn(100, dtype=torch.float64))
        w2 = nn.Parameter(torch.randn(100, dtype=torch.float64))
        opt = _make_optimizer(
            SGMCMC.sgld,
            [w1, w2],
            lr=0.5,
            nbeta=2.0,
            noise_level=0.0,
            save_metrics=True,
            sketch_dim=64,
        )

        w1.grad = torch.randn_like(w1)
        w2.grad = torch.randn_like(w2)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        metrics = opt.get_metrics()
        sketches = opt.get_sketches()

        exact_norm = metrics.scaled_grad.item()
        sketch_norm = sketches.scaled_grad.norm().item()
        assert exact_norm > 0
        assert sketch_norm == pytest.approx(exact_norm, rel=0.2)

    def test_multiple_param_groups(self):
        """Sketch aggregation across param groups produces correct norms."""
        w1 = nn.Parameter(torch.randn(100, dtype=torch.float64))
        w2 = nn.Parameter(torch.randn(100, dtype=torch.float64))
        opt = _make_optimizer(
            SGMCMC.sgld,
            [{"params": [w1]}, {"params": [w2]}],
            lr=0.5,
            nbeta=2.0,
            noise_level=0.0,
            save_metrics=True,
            sketch_dim=64,
        )

        w1.grad = torch.randn_like(w1)
        w2.grad = torch.randn_like(w2)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        metrics = opt.get_metrics()
        sketches = opt.get_sketches()

        exact_norm = metrics.scaled_grad.item()
        sketch_norm = sketches.scaled_grad.norm().item()
        assert exact_norm > 0
        assert sketch_norm == pytest.approx(exact_norm, rel=0.2)


class TestSketchWithGradNoneGap:
    """Sketch offsets stay correct when a middle param has grad=None.

    When the optimizer skips a param (no gradient), param_offset must still
    advance past it so subsequent params scatter into the right hash/sign
    region. Without that advance, the sketch vector would use wrong indices
    and the exact comparison below would fail.
    """

    @pytest.mark.parametrize("factory", [SGMCMC.sgld, SGLD], ids=["sgmcmc", "sgld"])
    def test_norm_crosscheck_with_grad_gap(self, factory):
        torch.manual_seed(42)
        w1 = nn.Parameter(torch.randn(80, dtype=torch.float64))
        w2 = nn.Parameter(torch.randn(40, dtype=torch.float64))
        w3 = nn.Parameter(torch.randn(80, dtype=torch.float64))

        opt = _make_optimizer(
            factory,
            [w1, w2, w3],
            lr=0.5,
            nbeta=2.0,
            noise_level=0.0,
            save_metrics=True,
            sketch_dim=64,
        )

        w1.grad = torch.randn_like(w1)
        # w2.grad is intentionally left as None (the default).
        w3.grad = torch.randn_like(w3)

        opt.step(noise_generator=torch.Generator().manual_seed(0))

        metrics = opt.get_metrics()
        sketches = opt.get_sketches()

        for q in ("scaled_grad", "unscaled_grad"):
            exact = getattr(metrics, q).item()
            sketch_norm = getattr(sketches, q).norm().item()
            assert exact > 0, q
            assert sketch_norm == pytest.approx(exact, rel=0.25), q

    def test_sketch_matches_manual_scatter(self):
        """Exact comparison: optimizer sketch == manual scatter at correct offsets."""
        torch.manual_seed(42)
        w1 = nn.Parameter(torch.randn(80, dtype=torch.float64))
        w2 = nn.Parameter(torch.randn(40, dtype=torch.float64))
        w3 = nn.Parameter(torch.randn(80, dtype=torch.float64))

        opt = _make_optimizer(
            SGMCMC.sgld,
            [w1, w2, w3],
            lr=0.5,
            nbeta=2.0,
            noise_level=0.0,
            sketch_dim=64,
        )

        w1.grad = torch.randn_like(w1)
        # w2.grad is intentionally left as None (the default).
        w3.grad = torch.randn_like(w3)

        # SGMCMC.sgld preconditioner: overall_coef=1, grad_coef=1
        # scaled_grad = (lr/2) * grad * nbeta = 0.25 * grad * 2.0 = 0.5 * grad
        sg1 = (0.5 * w1.grad).float()
        sg3 = (0.5 * w3.grad).float()

        expected = torch.zeros(64, dtype=torch.float32)
        assert opt._sketch is not None
        opt._sketch.scatter_into_(expected, sg1, offset=0)
        opt._sketch.scatter_into_(expected, sg3, offset=80 + 40)

        opt.step(noise_generator=torch.Generator().manual_seed(0))
        actual = opt.get_sketches().scaled_grad

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class TestSketchWithoutMetrics:
    """Sketches work independently of save_metrics."""

    def test_sketches_without_save_metrics(self):
        w = nn.Parameter(torch.zeros(50, dtype=torch.float64))
        opt = _make_optimizer(
            SGMCMC.sgld,
            [w],
            lr=0.5,
            nbeta=2.0,
            localization=1.0,
            noise_level=0.0,
            save_metrics=False,
            sketch_dim=64,
        )

        w.data.fill_(1.0)
        w.grad = torch.randn_like(w)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        sketches = opt.get_sketches()
        assert sketches.scaled_grad.norm().item() > 0
        assert sketches.localization.norm().item() > 0

    def test_decomposition_works_without_save_metrics(self):
        """Prior decomposition into loc/wd happens even without save_metrics."""
        w = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
        opt = _make_optimizer(
            SGMCMC.sgld,
            [w],
            lr=2.0,
            nbeta=2.0,
            localization=1.0,
            weight_decay=1.0,
            noise_level=0.0,
            save_metrics=False,
            sketch_dim=64,
        )

        w.data = torch.tensor([3.0, 4.0], dtype=torch.float64)
        w.grad = torch.zeros(2, dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        sketches = opt.get_sketches()
        assert sketches.localization.norm().item() > 0
        assert sketches.weight_decay.norm().item() > 0


# ---------------------------------------------------------------------------
# Legacy SGLD sketch tests
# ---------------------------------------------------------------------------


class TestSGLDSketchInit:
    def test_creates_sketch_buffer(self):
        w = nn.Parameter(torch.randn(10, dtype=torch.float64))
        opt = _make_optimizer(SGLD, [w], lr=0.5, nbeta=2.0, sketch_dim=32)
        assert opt.save_sketches
        assert isinstance(opt._sketch_buf, SketchBuffer)

    def test_correct_output_dim(self):
        w = nn.Parameter(torch.randn(10, dtype=torch.float64))
        opt = _make_optimizer(SGLD, [w], lr=0.5, nbeta=2.0, sketch_dim=64)
        assert opt._sketch_buf.scaled_grad.shape == (64,)

    def test_not_enabled_by_default(self):
        w = nn.Parameter(torch.randn(10, dtype=torch.float64))
        opt = _make_optimizer(SGLD, [w], lr=0.5, nbeta=2.0)
        assert not opt.save_sketches

    def test_get_sketches_raises_when_not_enabled(self):
        w = nn.Parameter(torch.randn(5))
        opt = _make_optimizer(SGLD, [w], lr=0.5, nbeta=2.0)
        with pytest.raises(RuntimeError, match="Sketches not enabled"):
            opt.get_sketches()

    def test_returns_all_quantities_on_cpu(self):
        w = nn.Parameter(torch.randn(10, dtype=torch.float64))
        opt = _make_optimizer(
            SGLD, [w], lr=0.5, nbeta=2.0, noise_level=0.0, sketch_dim=32
        )
        w.grad = torch.randn_like(w)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        sketches = opt.get_sketches()
        for q in SKETCH_QUANTITIES:
            t = getattr(sketches, q)
            assert t.shape == (32,), q
            assert t.device == torch.device("cpu"), q


class TestSGLDSketchZeroComponents:
    """When a component is zero in SGLD, its sketch should also be zero."""

    def test_noise_zero_when_disabled(self):
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        opt = _make_optimizer(
            SGLD, [w], lr=1.0, nbeta=2.0, noise_level=0.0, sketch_dim=32
        )
        w.grad = torch.randn_like(w)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        sketches = opt.get_sketches()
        torch.testing.assert_close(sketches.noise, torch.zeros(32), atol=0, rtol=0)

    def test_prior_zero_without_localization_or_weight_decay(self):
        w = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        opt = _make_optimizer(
            SGLD, [w], lr=1.0, nbeta=2.0, noise_level=0.0, sketch_dim=32
        )
        w.grad = torch.randn_like(w)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        sketches = opt.get_sketches()
        torch.testing.assert_close(
            sketches.localization, torch.zeros(32), atol=0, rtol=0
        )
        torch.testing.assert_close(
            sketches.weight_decay, torch.zeros(32), atol=0, rtol=0
        )


class TestSGLDSketchNormCrossCheck:
    """Sketch norms from SGLD should approximate the exact metric norms."""

    def test_all_quantities_over_multiple_steps(self):
        torch.manual_seed(42)
        w = nn.Parameter(torch.randn(200, dtype=torch.float64))
        opt = _make_optimizer(
            SGLD,
            [w],
            lr=0.5,
            nbeta=2.0,
            localization=1.0,
            weight_decay=0.5,
            save_metrics=True,
            sketch_dim=64,
        )

        gen = torch.Generator().manual_seed(99)
        for _ in range(5):
            w.grad = torch.randn_like(w)
            opt.step(noise_generator=gen)

            metrics = opt.get_metrics()
            sketches = opt.get_sketches()

            for q in ("scaled_grad", "unscaled_grad", "localization", "weight_decay"):
                exact = getattr(metrics, q).item()
                sketch_norm = getattr(sketches, q).norm().item()
                if exact > 1e-8:
                    assert sketch_norm == pytest.approx(exact, rel=0.25), q

    def test_with_mask(self):
        """SGLD applies masks manually; sketches must respect them."""
        w = nn.Parameter(torch.randn(100, dtype=torch.float64))
        mask = torch.zeros(100, dtype=torch.float64)
        mask[:50] = 1.0

        opt = _make_optimizer(
            SGLD,
            [{"params": [w], "mask": mask}],
            lr=0.5,
            nbeta=2.0,
            localization=1.0,
            save_metrics=True,
            noise_level=0.0,
            sketch_dim=64,
        )

        w.grad = torch.randn_like(w)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        metrics = opt.get_metrics()
        sketches = opt.get_sketches()

        exact_norm = metrics.scaled_grad.item()
        sketch_norm = sketches.scaled_grad.norm().item()
        assert exact_norm > 0
        assert sketch_norm == pytest.approx(exact_norm, rel=0.25)


class TestSGLDSketchWithoutMetrics:
    """SGLD sketches work independently of save_metrics."""

    def test_sketches_without_save_metrics(self):
        w = nn.Parameter(torch.zeros(50, dtype=torch.float64))
        opt = _make_optimizer(
            SGLD,
            [w],
            lr=0.5,
            nbeta=2.0,
            localization=1.0,
            noise_level=0.0,
            save_metrics=False,
            sketch_dim=64,
        )

        w.data.fill_(1.0)
        w.grad = torch.randn_like(w)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        sketches = opt.get_sketches()
        assert sketches.scaled_grad.norm().item() > 0
        assert sketches.localization.norm().item() > 0

    def test_loc_and_wd_decomposed_without_save_metrics(self):
        """Both localization and weight_decay sketches are non-zero
        even when save_metrics=False."""
        w = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
        opt = _make_optimizer(
            SGLD,
            [w],
            lr=2.0,
            nbeta=2.0,
            localization=1.0,
            weight_decay=1.0,
            noise_level=0.0,
            save_metrics=False,
            sketch_dim=64,
        )

        w.data = torch.tensor([3.0, 4.0], dtype=torch.float64)
        w.grad = torch.zeros(2, dtype=torch.float64)
        opt.step(noise_generator=torch.Generator().manual_seed(0))

        sketches = opt.get_sketches()
        assert sketches.localization.norm().item() > 0
        assert sketches.weight_decay.norm().item() > 0
