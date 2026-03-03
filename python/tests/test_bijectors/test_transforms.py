"""Тесты для биективных трансформ.

Round-trip: forward потом inverse — должен вернуть то же самое. Иначе зачем мы тут.
log_abs_det_jacobian — это то, ради чего мы страдаем при reparameterization.
"""

import jax
import jax.numpy as jnp
import pytest

from hyperion_dsl.transforms import (
    IdentityTransform,
    ExpTransform,
    SigmoidTransform,
    BoundedTransform,
    StickBreakingTransform,
    biject_to,
)
from hyperion_dsl.constraints import positive, real, unit_interval, Bounded


class TestIdentityTransform:
    """Identity — когда ничего не надо делать. Самое простое, но и его надо проверить."""

    def test_round_trip(self):
        # Round-trip: forward потом inverse — должен вернуть то же самое
        t = IdentityTransform()
        x = jnp.array([1.0, 2.0, 3.0])
        y = t.forward(x)
        x_rec = t.inverse(y)
        assert jnp.allclose(x, x_rec)

    def test_log_det_jacobian(self):
        # LDJ для identity = 0. Иначе якобиан сломан
        t = IdentityTransform()
        x = jnp.array([1.0, 2.0])
        y = t.forward(x)
        ldj = t.log_abs_det_jacobian(x, y)
        assert jnp.allclose(ldj, 0.0)


class TestExpTransform:
    """Exp — unconstrained -> positive. Классика для scale-параметров."""

    def test_round_trip(self):
        t = ExpTransform()
        x = jnp.array([-1.0, 0.0, 1.0, 2.0])
        y = t.forward(x)
        x_rec = t.inverse(y)
        assert jnp.allclose(x, x_rec, atol=1e-5)

    def test_positive_output(self):
        # Exp всегда даёт положительное — иначе это не exp
        t = ExpTransform()
        x = jnp.array([-10.0, -1.0, 0.0, 1.0])
        y = t.forward(x)
        assert jnp.all(y > 0)

    def test_log_det(self):
        # log|d(exp(x))/dx| = x. Формула из учебника
        t = ExpTransform()
        x = jnp.array([1.0, 2.0])
        y = t.forward(x)
        ldj = t.log_abs_det_jacobian(x, y)
        assert jnp.allclose(ldj, x)


class TestSigmoidTransform:
    """Sigmoid — real -> (0,1). Для вероятностей и unit_interval."""

    def test_round_trip(self):
        t = SigmoidTransform()
        x = jnp.array([-2.0, 0.0, 2.0])
        y = t.forward(x)
        x_rec = t.inverse(y)
        assert jnp.allclose(x, x_rec, atol=1e-5)

    def test_unit_interval(self):
        # Sigmoid всегда в (0, 1) — иначе это не sigmoid
        t = SigmoidTransform()
        x = jnp.array([-5.0, 0.0, 5.0])
        y = t.forward(x)
        assert jnp.all((y > 0) & (y < 1))


class TestBoundedTransform:
    """Bounded — общий случай [low, high]. Stick-breaking внутри для simplex."""

    def test_round_trip(self):
        t = BoundedTransform(-1.0, 3.0)
        x = jnp.array([-2.0, 0.0, 2.0])
        y = t.forward(x)
        x_rec = t.inverse(y)
        assert jnp.allclose(x, x_rec, atol=1e-4)

    def test_output_in_bounds(self):
        # Выход должен быть строго в границах
        t = BoundedTransform(0.0, 10.0)
        x = jnp.array([-10.0, 0.0, 10.0])
        y = t.forward(x)
        assert jnp.all((y > 0.0) & (y < 10.0))


class TestStickBreakingTransform:
    """Stick-breaking — для симплекса. Сумма = 1, иначе не Dirichlet."""

    def test_output_on_simplex(self):
        t = StickBreakingTransform()
        x = jnp.array([0.0, 0.0, 0.0])
        y = t.forward(x)
        assert y.shape[-1] == 4
        assert jnp.allclose(jnp.sum(y), 1.0, atol=1e-5)
        assert jnp.all(y >= 0)


class TestBijectTo:
    """biject_to — маппинг constraint -> transform. Должен возвращать правильный тип."""

    def test_positive(self):
        t = biject_to(positive)
        assert isinstance(t, ExpTransform)

    def test_real(self):
        t = biject_to(real)
        assert isinstance(t, IdentityTransform)

    def test_unit_interval(self):
        t = biject_to(unit_interval)
        assert isinstance(t, SigmoidTransform)

    def test_bounded(self):
        t = biject_to(Bounded(-1, 5))
        assert isinstance(t, BoundedTransform)

    def test_corr_cholesky_with_dim(self):
        from hyperion_dsl.constraints import CorrCholesky
        from hyperion_dsl.transforms import CorrCholeskyTransform
        t = biject_to(CorrCholesky(dim=3))
        assert isinstance(t, CorrCholeskyTransform)
        assert t.dim == 3

    def test_corr_cholesky_without_dim_raises(self):
        from hyperion_dsl.constraints import CorrCholesky
        with pytest.raises(ValueError, match="dimension"):
            biject_to(CorrCholesky())


class TestCholeskyTransformDiagIndices:
    """CholeskyTransform: packed diagonal indices must be correct."""

    def test_diag_indices_dim2(self):
        from hyperion_dsl.transforms import CholeskyTransform
        t = CholeskyTransform(dim=2)
        assert list(t._diag_indices) == [0, 2]

    def test_diag_indices_dim3(self):
        from hyperion_dsl.transforms import CholeskyTransform
        t = CholeskyTransform(dim=3)
        assert list(t._diag_indices) == [0, 2, 5]

    def test_diag_indices_dim4(self):
        from hyperion_dsl.transforms import CholeskyTransform
        t = CholeskyTransform(dim=4)
        assert list(t._diag_indices) == [0, 2, 5, 9]

    def test_forward_inverse_roundtrip(self):
        from hyperion_dsl.transforms import CholeskyTransform
        t = CholeskyTransform(dim=3)
        x = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        L = t.forward(x)
        x_rec = t.inverse(L)
        assert jnp.allclose(x, x_rec, atol=1e-5)

    def test_log_det_jacobian_uses_correct_diag(self):
        from hyperion_dsl.transforms import CholeskyTransform
        t = CholeskyTransform(dim=3)
        x = jnp.array([1.0, 0.0, 2.0, 0.0, 0.0, 3.0])
        y = t.forward(x)
        ldj = t.log_abs_det_jacobian(x, y)
        assert jnp.isclose(ldj, 1.0 + 2.0 + 3.0, atol=1e-5)
