"""Расширенные тесты трансформаций: SoftplusTransform, AffineTransform, PowerTransform,
StickBreaking round-trip, ComposeTransform.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from hyperion_dsl.transforms import (
    SoftplusTransform, AffineTransform, PowerTransform,
    StickBreakingTransform, ComposeTransform, ExpTransform,
    SigmoidTransform, IdentityTransform,
)


class TestSoftplusTransform:
    def test_round_trip(self):
        t = SoftplusTransform()
        x = jnp.array([-2.0, 0.0, 1.0, 5.0])
        y = t.forward(x)
        x_back = t.inverse(y)
        assert jnp.allclose(x, x_back, atol=1e-4)

    def test_positive_output(self):
        t = SoftplusTransform()
        x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        y = t.forward(x)
        assert jnp.all(y > 0)


class TestAffineTransform:
    def test_round_trip(self):
        t = AffineTransform(loc=2.0, scale=3.0)
        x = jnp.array([-1.0, 0.0, 1.0, 5.0])
        y = t.forward(x)
        x_back = t.inverse(y)
        assert jnp.allclose(x, x_back, atol=1e-5)

    def test_log_det(self):
        t = AffineTransform(loc=2.0, scale=3.0)
        x = jnp.array(0.0)
        ld = t.log_abs_det_jacobian(x, t.forward(x))
        assert jnp.allclose(ld, jnp.log(jnp.array(3.0)), atol=1e-5)


class TestPowerTransform:
    def test_round_trip(self):
        t = PowerTransform(power=2.0)
        x = jnp.array([0.5, 1.0, 2.0, 3.0])
        y = t.forward(x)
        x_back = t.inverse(y)
        assert jnp.allclose(x, x_back, atol=1e-5)

    def test_power_zero_raises(self):
        with pytest.raises(ValueError):
            PowerTransform(power=0.0)


class TestStickBreakingRoundTrip:
    def test_round_trip(self):
        t = StickBreakingTransform()
        x = jnp.array([0.0, 0.0, 0.0])
        y = t.forward(x)
        assert jnp.allclose(jnp.sum(y), 1.0, atol=1e-5)
        x_back = t.inverse(y)
        assert jnp.allclose(x, x_back, atol=1e-4)

    def test_log_det_finite(self):
        t = StickBreakingTransform()
        x = jnp.array([0.5, -0.5, 1.0])
        y = t.forward(x)
        ld = t.log_abs_det_jacobian(x, y)
        assert jnp.isfinite(ld)


class TestComposeTransform:
    def test_round_trip(self):
        t = ComposeTransform([ExpTransform(), AffineTransform(loc=1.0, scale=2.0)])
        x = jnp.array([-1.0, 0.0, 1.0])
        y = t.forward(x)
        x_back = t.inverse(y)
        assert jnp.allclose(x, x_back, atol=1e-4)

    def test_log_det_additive(self):
        t1 = ExpTransform()
        t2 = AffineTransform(loc=0.0, scale=3.0)
        composed = ComposeTransform([t1, t2])
        x = jnp.array(1.0)
        y1 = t1.forward(x)
        y2 = t2.forward(y1)
        ld1 = t1.log_abs_det_jacobian(x, y1)
        ld2 = t2.log_abs_det_jacobian(y1, y2)
        ld_composed = composed.log_abs_det_jacobian(x, y2)
        assert jnp.allclose(ld_composed, ld1 + ld2, atol=1e-5)
