"""Comprehensive tests covering previously untested components.

- LKJCholesky distribution
- Horseshoe distribution
- GaussianProcess distribution
- CorrCholeskyTransform round-trip + log_det
- IR Optimizer (CSE, dead nodes, topo reorder)
- GraphBuilder (build, edges, topo order)
- Primitives (plate, deterministic, factor)
- Negative tests (invalid params, edge cases)
- Serialization round-trip with samples_by_chain
"""

import io
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperion_dsl.primitives import sample, plate, deterministic, factor
from hyperion_dsl.distributions import (
    Normal, HalfNormal, LKJCholesky, Horseshoe, GaussianProcess,
    MultivariateNormal, Gamma, Beta, Uniform, Exponential,
    Binomial, Multinomial, Poisson, Delta,
)
from hyperion_dsl.model import model
from hyperion_dsl.transforms import CorrCholeskyTransform, SoftplusTransform
from hyperion_trace.trace import trace_model


# ============================================================
# LKJCholesky Tests
# ============================================================

class TestLKJCholesky:

    def test_sample_shape_dim2(self):
        d = LKJCholesky(dimension=2, concentration=1.0)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(10,))
        assert samples.shape == (10, 2, 2)

    def test_sample_shape_dim3(self):
        d = LKJCholesky(dimension=3, concentration=2.0)
        key = jax.random.PRNGKey(1)
        samples = d.sample(key, sample_shape=(5,))
        assert samples.shape == (5, 3, 3)

    def test_sample_is_lower_triangular(self):
        d = LKJCholesky(dimension=3, concentration=1.0)
        key = jax.random.PRNGKey(2)
        L = d.sample(key)
        for i in range(3):
            for j in range(i + 1, 3):
                assert float(jnp.abs(L[i, j])) < 1e-6, f"L[{i},{j}] = {L[i,j]}"

    def test_sample_unit_diagonal_row_norms(self):
        d = LKJCholesky(dimension=3, concentration=1.0)
        key = jax.random.PRNGKey(3)
        L = d.sample(key)
        R = L @ L.T
        diag = jnp.diag(R)
        np.testing.assert_allclose(diag, jnp.ones(3), atol=1e-5)

    def test_log_prob_finite(self):
        d = LKJCholesky(dimension=3, concentration=1.0)
        key = jax.random.PRNGKey(4)
        L = d.sample(key)
        lp = d.log_prob(L)
        assert jnp.isfinite(lp), f"log_prob should be finite, got {lp}"

    def test_log_prob_higher_for_identity(self):
        d = LKJCholesky(dimension=3, concentration=5.0)
        L_id = jnp.eye(3)
        key = jax.random.PRNGKey(5)
        L_rand = d.sample(key)
        lp_id = d.log_prob(L_id)
        lp_rand = d.log_prob(L_rand)
        assert lp_id >= lp_rand - 1.0, (
            f"For high concentration, identity should have high log_prob: "
            f"id={float(lp_id):.3f}, rand={float(lp_rand):.3f}"
        )

    def test_dimension_1_raises(self):
        with pytest.raises(ValueError, match=">="):
            LKJCholesky(dimension=1)

    def test_support_corr_cholesky(self):
        d = LKJCholesky(dimension=3, concentration=1.0)
        from hyperion_dsl.constraints import CorrCholesky
        assert isinstance(d.support, CorrCholesky)


# ============================================================
# Horseshoe Tests
# ============================================================

class TestHorseshoe:

    def test_sample_shape(self):
        d = Horseshoe(scale=1.0)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(100,))
        assert samples.shape == (100,)

    def test_sample_has_heavy_tails(self):
        d = Horseshoe(scale=1.0)
        key = jax.random.PRNGKey(42)
        samples = d.sample(key, sample_shape=(10000,))
        assert float(jnp.max(jnp.abs(samples))) > 5.0

    def test_log_prob_raises(self):
        d = Horseshoe(scale=1.0)
        with pytest.raises(NotImplementedError):
            d.log_prob(jnp.array(1.0))

    def test_batched_scale(self):
        d = Horseshoe(scale=jnp.array([1.0, 2.0]))
        key = jax.random.PRNGKey(1)
        samples = d.sample(key, sample_shape=(50,))
        assert samples.shape == (50, 2)


# ============================================================
# GaussianProcess Tests
# ============================================================

class TestGaussianProcess:

    def test_sample_shape(self):
        K = jnp.eye(3)
        d = GaussianProcess(loc=jnp.zeros(3), covariance_matrix=K)
        key = jax.random.PRNGKey(0)
        s = d.sample(key, sample_shape=(10,))
        assert s.shape == (10, 3)

    def test_log_prob_finite(self):
        K = jnp.eye(3)
        d = GaussianProcess(loc=jnp.zeros(3), covariance_matrix=K)
        lp = d.log_prob(jnp.zeros(3))
        assert jnp.isfinite(lp)

    def test_matches_mvn(self):
        loc = jnp.array([1.0, 2.0])
        K = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        gp = GaussianProcess(loc=loc, covariance_matrix=K)
        mvn = MultivariateNormal(loc=loc, covariance_matrix=K)
        x = jnp.array([1.5, 2.5])
        np.testing.assert_allclose(
            float(gp.log_prob(x)), float(mvn.log_prob(x)), atol=1e-5
        )


# ============================================================
# CorrCholeskyTransform Tests
# ============================================================

class TestCorrCholeskyTransformFull:

    def test_round_trip_dim3(self):
        t = CorrCholeskyTransform(dim=3)
        x = jnp.array([0.5, -0.3, 0.1])
        L = t.forward(x)
        x_back = t.inverse(L)
        np.testing.assert_allclose(x_back, x, atol=1e-4)

    def test_round_trip_dim4(self):
        t = CorrCholeskyTransform(dim=4)
        x = jax.random.normal(jax.random.PRNGKey(0), shape=(6,))
        L = t.forward(x)
        x_back = t.inverse(L)
        np.testing.assert_allclose(x_back, x, atol=1e-4)

    def test_output_is_valid_corr_cholesky(self):
        t = CorrCholeskyTransform(dim=3)
        x = jnp.array([0.3, -0.2, 0.4])
        L = t.forward(x)
        R = L @ L.T
        np.testing.assert_allclose(jnp.diag(R), jnp.ones(3), atol=1e-5)
        assert jnp.all(jnp.diag(L) > 0), "Diagonal must be positive"

    def test_log_det_jacobian_finite(self):
        t = CorrCholeskyTransform(dim=3)
        x = jnp.array([0.5, -0.3, 0.1])
        L = t.forward(x)
        ldj = t.log_abs_det_jacobian(x, L)
        assert jnp.isfinite(ldj), f"log_det_jacobian should be finite, got {ldj}"

    def test_log_det_jacobian_at_zero(self):
        t = CorrCholeskyTransform(dim=3)
        x = jnp.zeros(3)
        L = t.forward(x)
        ldj = t.log_abs_det_jacobian(x, L)
        assert jnp.isfinite(ldj)


# ============================================================
# SoftplusTransform Inverse Stability Tests
# ============================================================

class TestSoftplusTransformStability:

    def test_inverse_small_y(self):
        t = SoftplusTransform()
        y = jnp.array([0.01, 0.001, 0.1, 1.0, 10.0, 50.0])
        x = t.inverse(y)
        y_back = t.forward(x)
        np.testing.assert_allclose(y_back, y, atol=1e-4)

    def test_inverse_large_y(self):
        t = SoftplusTransform()
        y = jnp.array([100.0, 1000.0])
        x = t.inverse(y)
        assert jnp.all(jnp.isfinite(x))
        y_back = t.forward(x)
        np.testing.assert_allclose(y_back, y, atol=1e-3)


# ============================================================
# IR Optimizer Tests
# ============================================================

class TestIROptimizer:

    def _make_simple_ir(self):
        from hyperion_ir.ir import IRGraph, IRNode, IRNodeType
        g = IRGraph()
        g.add_node(IRNode(name="mu", node_type=IRNodeType.SAMPLE, shape=()))
        g.add_node(IRNode(name="obs", node_type=IRNodeType.OBSERVED, shape=(10,)))
        g.nodes["obs"].parents.append("mu")
        g.nodes["mu"].children.append("obs")
        return g

    def test_dead_node_elimination(self):
        from hyperion_ir.ir import IRGraph, IRNode, IRNodeType
        from hyperion_ir.optimizer import IROptimizer
        g = IRGraph()
        g.add_node(IRNode(name="mu", node_type=IRNodeType.SAMPLE, shape=()))
        g.add_node(IRNode(name="dead", node_type=IRNodeType.DETERMINISTIC, shape=()))
        g.add_node(IRNode(name="obs", node_type=IRNodeType.OBSERVED, shape=()))
        g.nodes["obs"].parents.append("mu")
        g.nodes["mu"].children.append("obs")

        opt = IROptimizer(passes=["dead_node_elimination"])
        g2 = opt.optimize(g)
        assert "dead" not in g2.nodes
        assert "mu" in g2.nodes
        assert "obs" in g2.nodes

    def test_cse_merges_same_hash(self):
        from hyperion_ir.ir import IRGraph, IRNode, IRNodeType
        from hyperion_ir.optimizer import IROptimizer
        g = IRGraph()
        g.add_node(IRNode(
            name="d1", node_type=IRNodeType.DETERMINISTIC,
            distribution_type="Normal", shape=(1,), plates=[],
            distribution_params={"loc": 0.0, "scale": 1.0},
        ))
        g.add_node(IRNode(
            name="d2", node_type=IRNodeType.DETERMINISTIC,
            distribution_type="Normal", shape=(1,), plates=[],
            distribution_params={"loc": 0.0, "scale": 1.0},
        ))
        g.add_node(IRNode(name="child", node_type=IRNodeType.OBSERVED, shape=()))
        g.nodes["child"].parents = ["d2"]
        g.nodes["d2"].children = ["child"]

        opt = IROptimizer(passes=["common_subexpression_elimination"])
        g2 = opt.optimize(g)
        assert "d2" not in g2.nodes or "d1" not in g2.nodes

    def test_cse_does_not_merge_different_params(self):
        from hyperion_ir.ir import IRGraph, IRNode, IRNodeType
        from hyperion_ir.optimizer import IROptimizer
        g = IRGraph()
        g.add_node(IRNode(
            name="d1", node_type=IRNodeType.DETERMINISTIC,
            distribution_type="Normal", shape=(1,), plates=[],
            distribution_params={"loc": 0.0, "scale": 1.0},
        ))
        g.add_node(IRNode(
            name="d2", node_type=IRNodeType.DETERMINISTIC,
            distribution_type="Normal", shape=(1,), plates=[],
            distribution_params={"loc": 0.0, "scale": 2.0},
        ))
        opt = IROptimizer(passes=["common_subexpression_elimination"])
        g2 = opt.optimize(g)
        assert "d1" in g2.nodes
        assert "d2" in g2.nodes

    def test_topological_reorder(self):
        from hyperion_ir.ir import IRGraph, IRNode, IRNodeType
        from hyperion_ir.optimizer import IROptimizer
        g = IRGraph()
        g.add_node(IRNode(name="obs", node_type=IRNodeType.OBSERVED, shape=()))
        g.add_node(IRNode(name="mu", node_type=IRNodeType.SAMPLE, shape=()))
        g.nodes["obs"].parents.append("mu")
        g.nodes["mu"].children.append("obs")

        opt = IROptimizer(passes=["topological_reorder"])
        g2 = opt.optimize(g)
        assert g2.execution_order.index("mu") < g2.execution_order.index("obs")

    def test_optimizer_idempotent(self):
        from hyperion_ir.ir import IRGraph, IRNode, IRNodeType
        from hyperion_ir.optimizer import IROptimizer
        g = IRGraph()
        g.add_node(IRNode(name="mu", node_type=IRNodeType.SAMPLE, shape=()))
        g.add_node(IRNode(name="obs", node_type=IRNodeType.OBSERVED, shape=()))
        g.nodes["obs"].parents.append("mu")
        g.nodes["mu"].children.append("obs")

        opt = IROptimizer()
        g1 = opt.optimize(g)
        n1 = len(g1.nodes)
        g2 = opt.optimize(g1)
        n2 = len(g2.nodes)
        assert n1 == n2


# ============================================================
# GraphBuilder Tests
# ============================================================

class TestGraphBuilder:

    def test_build_creates_nodes(self):
        from hyperion_graph.graph_builder import GraphBuilder

        @model
        def simple():
            mu = sample("mu", Normal(0.0, 1.0))
            sample("obs", Normal(mu, 1.0), obs=jnp.array(1.0))

        key = jax.random.PRNGKey(0)
        trace = trace_model(simple, rng_key=key)
        gb = GraphBuilder()
        mg = gb.build(trace)

        assert "mu" in mg.nodes
        assert "obs" in mg.nodes

    def test_latent_and_observed_filtering(self):
        from hyperion_graph.graph_builder import GraphBuilder

        @model
        def m():
            mu = sample("mu", Normal(0.0, 1.0))
            sample("obs", Normal(mu, 1.0), obs=jnp.array(1.0))

        trace = trace_model(m, rng_key=jax.random.PRNGKey(0))
        gb = GraphBuilder()
        mg = gb.build(trace)

        assert "mu" in mg.latent_nodes
        assert "obs" in mg.observed_nodes

    def test_topological_order(self):
        from hyperion_graph.graph_builder import GraphBuilder

        @model
        def m():
            mu = sample("mu", Normal(0.0, 1.0))
            sample("obs", Normal(mu, 1.0), obs=jnp.array(1.0))

        trace = trace_model(m, rng_key=jax.random.PRNGKey(0))
        gb = GraphBuilder()
        mg = gb.build(trace)

        order = mg.topological_order()
        assert len(order) >= 2

    def test_roots_and_leaves(self):
        from hyperion_graph.graph_builder import GraphBuilder

        @model
        def m():
            mu = sample("mu", Normal(0.0, 1.0))
            sample("obs", Normal(mu, 1.0), obs=jnp.array(1.0))

        trace = trace_model(m, rng_key=jax.random.PRNGKey(0))
        gb = GraphBuilder()
        mg = gb.build(trace)

        assert len(mg.roots()) >= 1
        assert len(mg.leaves()) >= 1


# ============================================================
# Primitives Tests (plate, deterministic, factor)
# ============================================================

class TestPrimitivePlate:

    def test_plate_in_trace(self):
        @model
        def m():
            mu = sample("mu", Normal(0.0, 1.0))
            with plate("data", 5):
                sample("obs", Normal(mu, 1.0), obs=jnp.ones(5))

        trace = trace_model(m, rng_key=jax.random.PRNGKey(0))
        assert "mu" in trace.latent_names
        assert "obs" in trace.observed_names
        assert jnp.isfinite(trace.log_joint())


class TestPrimitiveDeterministic:

    def test_deterministic_in_trace(self):
        @model
        def m():
            mu = sample("mu", Normal(0.0, 1.0))
            y = deterministic("y", mu * 2)
            sample("obs", Normal(y, 1.0), obs=jnp.array(1.0))

        trace = trace_model(m, rng_key=jax.random.PRNGKey(0))
        assert "y" in trace.entries
        assert trace.entries["y"].value is not None


class TestPrimitiveFactor:

    def test_factor_in_trace(self):
        @model
        def m():
            mu = sample("mu", Normal(0.0, 1.0))
            factor("penalty", -0.5 * mu**2)

        trace = trace_model(m, rng_key=jax.random.PRNGKey(0))
        assert "penalty" in trace.entries
        lj = trace.log_joint()
        assert jnp.isfinite(lj)


# ============================================================
# Negative Tests
# ============================================================

class TestNegativeDistributions:

    def test_normal_negative_scale_warns(self):
        with pytest.raises(ValueError, match="positive"):
            Normal(0.0, -1.0)

    def test_gamma_negative_concentration_warns(self):
        with pytest.raises(ValueError, match="positive"):
            Gamma(-1.0, 1.0)

    def test_gamma_negative_rate_warns(self):
        with pytest.raises(ValueError, match="positive"):
            Gamma(1.0, -1.0)

    def test_beta_zero_concentration_warns(self):
        with pytest.raises(ValueError, match="positive"):
            Beta(0.0, 1.0)

    def test_uniform_low_ge_high(self):
        with pytest.raises(ValueError, match="low < high"):
            Uniform(5.0, 3.0)

    def test_exponential_negative_rate(self):
        with pytest.raises(ValueError, match="positive"):
            Exponential(-1.0)

    def test_binomial_needs_probs_or_logits(self):
        with pytest.raises(ValueError):
            Binomial(total_count=10)

    def test_multinomial_needs_probs_or_logits(self):
        with pytest.raises(ValueError):
            Multinomial(total_count=10)

    def test_sample_outside_trace_raises(self):
        with pytest.raises(RuntimeError, match="trace_model"):
            sample("x", Normal(0.0, 1.0))


class TestNegativeInference:

    def test_mcmc_get_samples_before_run(self):
        from hyperion_inference.mcmc import MCMC
        from hyperion_inference.hmc import HMCKernel
        mcmc = MCMC(HMCKernel(), num_warmup=10, num_samples=10)
        with pytest.raises(RuntimeError, match="run"):
            mcmc.get_samples()

    def test_predictive_without_samples_or_num(self):
        from hyperion_inference.predictive import Predictive
        with pytest.raises(ValueError):
            Predictive(lambda: None)

    def test_lkj_dim1_raises(self):
        with pytest.raises(ValueError):
            LKJCholesky(dimension=1)


# ============================================================
# Serialization Round-trip with samples_by_chain
# ============================================================

class TestSerializationRoundTrip:

    def test_save_load_with_samples_by_chain(self):
        from hyperion_inference.base import InferenceResult
        from hyperion_exp.serialization import save_result, load_result

        samples = {"mu": jnp.array([1.0, 2.0, 3.0, 4.0])}
        by_chain = {"mu": jnp.array([[1.0, 2.0], [3.0, 4.0]])}
        result = InferenceResult(
            samples=samples,
            diagnostics={"mean_accept_prob": 0.8},
            num_chains=2,
            samples_by_chain=by_chain,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_result(result, tmpdir)
            loaded = load_result(tmpdir)

        assert loaded.samples_by_chain is not None
        np.testing.assert_allclose(
            np.asarray(loaded.samples_by_chain["mu"]),
            np.asarray(by_chain["mu"]),
            atol=1e-5,
        )
        assert loaded.num_chains == 2

    def test_save_load_without_by_chain(self):
        from hyperion_inference.base import InferenceResult
        from hyperion_exp.serialization import save_result, load_result

        samples = {"mu": jnp.array([1.0, 2.0, 3.0])}
        result = InferenceResult(samples=samples, diagnostics={})

        with tempfile.TemporaryDirectory() as tmpdir:
            save_result(result, tmpdir)
            loaded = load_result(tmpdir)

        assert loaded.samples_by_chain is None
        np.testing.assert_allclose(
            np.asarray(loaded.samples["mu"]),
            np.asarray(samples["mu"]),
            atol=1e-5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
