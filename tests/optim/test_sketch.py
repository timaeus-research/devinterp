"""Tests for CountSketch and SketchBuffer."""

import pytest
import torch
from devinterp.optim.sketch import SKETCH_QUANTITIES, CountSketch, SketchBuffer


class TestCountSketchCreation:
    def test_deterministic(self):
        """Same seed produces identical hash arrays."""
        a = CountSketch.create(100, 32, seed=42)
        b = CountSketch.create(100, 32, seed=42)
        torch.testing.assert_close(a.hash_indices, b.hash_indices, atol=0, rtol=0)
        torch.testing.assert_close(a.hash_signs, b.hash_signs, atol=0, rtol=0)

    def test_different_seeds_differ(self):
        a = CountSketch.create(100, 32, seed=0)
        b = CountSketch.create(100, 32, seed=1)
        assert not torch.equal(a.hash_indices, b.hash_indices)

    def test_dimensions(self):
        cs = CountSketch.create(input_dim=500, output_dim=64, seed=0)
        assert cs.input_dim == 500
        assert cs.output_dim == 64
        assert cs.hash_indices.shape == (500,)
        assert cs.hash_signs.shape == (500,)

    def test_hash_indices_in_range(self):
        cs = CountSketch.create(10000, 128, seed=0)
        assert cs.hash_indices.min() >= 0
        assert cs.hash_indices.max() < 128

    def test_hash_signs_are_pm1(self):
        cs = CountSketch.create(1000, 64, seed=0)
        mask = (cs.hash_signs == 1.0) | (cs.hash_signs == -1.0)
        assert torch.all(mask), (
            f"Non-±1 signs at indices {torch.where(~mask)[0].tolist()}"
        )


class TestCountSketchMath:
    def test_sketch_exact_small_example(self):
        """Verify sketch output by enumerating bucket assignments."""
        cs = CountSketch.create(4, 3, seed=0)
        h = cs.hash_indices
        s = cs.hash_signs
        v = torch.tensor([3.0, 4.0, 0.0, 5.0])

        expected = torch.zeros(3)
        for i in range(4):
            expected[h[i]] += s[i] * v[i]

        result = cs.sketch(v)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)

    def test_linearity(self):
        """S(v + w) = Sv + Sw."""
        cs = CountSketch.create(100, 32, seed=7)
        torch.manual_seed(0)
        v = torch.randn(100)
        w = torch.randn(100)

        s_sum = cs.sketch(v + w)
        sv_plus_sw = cs.sketch(v) + cs.sketch(w)

        torch.testing.assert_close(s_sum, sv_plus_sw, atol=1e-5, rtol=0)

    def test_scalar_scaling(self):
        """S(alpha * v) = alpha * Sv."""
        cs = CountSketch.create(100, 32, seed=7)
        torch.manual_seed(0)
        v = torch.randn(100)
        alpha = 3.5

        torch.testing.assert_close(
            cs.sketch(alpha * v), alpha * cs.sketch(v), atol=1e-5, rtol=0
        )

    def test_scatter_into_matches_full_sketch(self):
        """Accumulating via scatter_into_ with offset reproduces sketch(cat(v1, v2))."""
        cs = CountSketch.create(8, 4, seed=3)
        v1 = torch.tensor([1.0, 2.0, 3.0])
        v2 = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0])

        full = cs.sketch(torch.cat([v1, v2]))

        accum = torch.zeros(4)
        cs.scatter_into_(accum, v1, offset=0)
        cs.scatter_into_(accum, v2, offset=3)

        torch.testing.assert_close(accum, full, atol=0, rtol=0)

    def test_scatter_into_with_multidim_param(self):
        """scatter_into_ flattens multi-dimensional tensors."""
        cs = CountSketch.create(12, 8, seed=5)
        param = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        full = cs.sketch(param.reshape(-1))

        accum = torch.zeros(8)
        cs.scatter_into_(accum, param, offset=0)

        torch.testing.assert_close(accum, full, atol=0, rtol=0)

    def test_sketch_of_zeros_is_zero(self):
        cs = CountSketch.create(100, 32, seed=0)
        result = cs.sketch(torch.zeros(100))
        torch.testing.assert_close(result, torch.zeros(32), atol=0, rtol=0)


class TestCountSketchStatistical:
    """Statistical tests of the unbiasedness properties.

    These average over many independent sketches (different seeds) to verify
    that E[<Sv, Sw>] = <v, w>.
    """

    def test_inner_product_unbiased(self):
        """Mean sketch inner product converges to true inner product."""
        d, k, num_trials = 200, 64, 500
        v = torch.zeros(d)
        w = torch.zeros(d)
        v[:5] = torch.tensor([3.0, 4.0, 1.0, 2.0, 5.0])
        w[:5] = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
        true_dot = torch.dot(v, w)  # 3*4 + 4*3 + 1*2 + 2*1 = 28

        estimates = torch.empty(num_trials)
        for seed in range(num_trials):
            cs = CountSketch.create(d, k, seed=seed)
            estimates[seed] = torch.dot(cs.sketch(v), cs.sketch(w))

        mean_estimate = estimates.mean()
        torch.testing.assert_close(mean_estimate, true_dot, atol=1.0, rtol=0)

    def test_norm_preserved_in_expectation(self):
        """Mean sketch squared norm converges to true squared norm."""
        d, k, num_trials = 200, 64, 500
        v = torch.zeros(d)
        v[0], v[1] = 3.0, 4.0
        true_norm_sq = torch.tensor(25.0)

        estimates = torch.empty(num_trials)
        for seed in range(num_trials):
            cs = CountSketch.create(d, k, seed=seed)
            estimates[seed] = cs.sketch(v).norm().square()

        mean_estimate = estimates.mean()
        torch.testing.assert_close(mean_estimate, true_norm_sq, atol=1.0, rtol=0)

    def test_orthogonal_vectors_zero_dot(self):
        """Sketch inner product of orthogonal vectors averages to zero."""
        d, k, num_trials = 200, 64, 500
        v = torch.zeros(d)
        w = torch.zeros(d)
        v[0] = 5.0
        w[1] = 5.0

        estimates = torch.empty(num_trials)
        for seed in range(num_trials):
            cs = CountSketch.create(d, k, seed=seed)
            estimates[seed] = torch.dot(cs.sketch(v), cs.sketch(w))

        mean_estimate = estimates.mean()
        torch.testing.assert_close(mean_estimate, torch.tensor(0.0), atol=1.0, rtol=0)


class TestSketchBuffer:
    def test_create_shapes(self):
        buf = SketchBuffer.create(64)
        for q in SKETCH_QUANTITIES:
            t = getattr(buf, q)
            assert t.shape == (64,)
            assert t.dtype == torch.float32

    def test_zero(self):
        buf = SketchBuffer.create(32)
        buf.scaled_grad.fill_(1.0)
        buf.noise.fill_(2.0)
        buf.zero_()
        for q in SKETCH_QUANTITIES:
            torch.testing.assert_close(getattr(buf, q), torch.zeros(32), atol=0, rtol=0)

    def test_to_same_device_returns_self(self):
        buf = SketchBuffer.create(16)
        assert buf.to("cpu") is buf

    def test_device_inconsistent_raises(self):
        buf = SketchBuffer.create(16)
        buf.scaled_grad = buf.scaled_grad.to("meta")
        with pytest.raises(RuntimeError, match="Inconsistent devices"):
            buf.device


class TestCountSketchDevice:
    def test_to_same_device_returns_self(self):
        cs = CountSketch.create(32, 8, seed=0)
        assert cs.to("cpu") is cs

    def test_device_inconsistent_raises(self):
        cs = CountSketch.create(32, 8, seed=0)
        cs.hash_signs = cs.hash_signs.to("meta")
        with pytest.raises(RuntimeError, match="Inconsistent devices"):
            cs.device
