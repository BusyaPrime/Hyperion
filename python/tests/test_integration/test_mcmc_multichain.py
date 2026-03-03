"""End-to-end tests for MCMC multi-chain, print_summary, Predictive.

Verifies that HMC/NUTS с num_chains>1 не падают, summary_table
корректно использует per-chain ESS, print_summary не крашится,
Predictive работает с posterior samples от multi-chain.
"""

import io
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperion_dsl.primitives import sample
from hyperion_dsl.distributions import Normal, HalfNormal
from hyperion_dsl.model import model
from hyperion_inference.mcmc import MCMC
from hyperion_inference.hmc import HMCKernel
from hyperion_inference.nuts import NUTSKernel
from hyperion_inference.predictive import Predictive


_Y_OBS = jnp.array(np.random.RandomState(42).normal(3.0, 1.0, 30).astype(np.float32))


@model
def simple_normal_model():
    mu = sample("mu", Normal(0.0, 10.0))
    sample("obs", Normal(mu, 1.0), obs=_Y_OBS)


@pytest.fixture(scope="module")
def observed_data():
    return {"obs": _Y_OBS}


class TestMCMCMultiChainHMC:
    """HMC multi-chain end-to-end."""

    def test_hmc_2chains_runs(self, observed_data):
        kernel = HMCKernel()
        mcmc = MCMC(
            kernel,
            num_warmup=100,
            num_samples=200,
            num_chains=2,
            step_size=0.05,
            num_leapfrog=10,
        )
        key = jax.random.PRNGKey(0)
        mcmc.run(key, simple_normal_model, observed_data)

        samples = mcmc.get_samples()
        assert "mu" in samples
        assert samples["mu"].shape[0] == 400  # 2 chains * 200

        by_chain = mcmc.get_samples_by_chain()
        assert by_chain is not None
        assert by_chain["mu"].shape == (2, 200)

    def test_hmc_2chains_print_summary_no_crash(self, observed_data):
        kernel = HMCKernel()
        mcmc = MCMC(
            kernel, num_warmup=100, num_samples=200,
            num_chains=2, step_size=0.05, num_leapfrog=10,
        )
        mcmc.run(jax.random.PRNGKey(1), simple_normal_model, observed_data)

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            mcmc.print_summary()
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "mu" in output
        assert "r_hat" in output
        assert "ess" in output
        assert "Mean accept prob" in output

    def test_hmc_2chains_diagnostics_scalar(self, observed_data):
        kernel = HMCKernel()
        mcmc = MCMC(
            kernel, num_warmup=100, num_samples=200,
            num_chains=2, step_size=0.05, num_leapfrog=10,
        )
        mcmc.run(jax.random.PRNGKey(2), simple_normal_model, observed_data)

        d = mcmc.diagnostics
        if "mean_accept_prob" in d:
            val = d["mean_accept_prob"]
            assert np.ndim(val) == 0, f"Expected scalar, got ndim={np.ndim(val)}"
            assert 0 <= float(val) <= 1


class TestMCMCMultiChainNUTS:
    """NUTS multi-chain end-to-end."""

    def test_nuts_2chains_runs(self, observed_data):
        kernel = NUTSKernel()
        mcmc = MCMC(
            kernel,
            num_warmup=100,
            num_samples=200,
            num_chains=2,
            step_size=0.05,
            max_tree_depth=6,
        )
        key = jax.random.PRNGKey(10)
        mcmc.run(key, simple_normal_model, observed_data)

        samples = mcmc.get_samples()
        assert samples["mu"].shape[0] == 400

        by_chain = mcmc.get_samples_by_chain()
        assert by_chain is not None
        assert by_chain["mu"].shape == (2, 200)

    def test_nuts_2chains_print_summary_no_crash(self, observed_data):
        kernel = NUTSKernel()
        mcmc = MCMC(
            kernel, num_warmup=100, num_samples=200,
            num_chains=2, step_size=0.05, max_tree_depth=6,
        )
        mcmc.run(jax.random.PRNGKey(11), simple_normal_model, observed_data)

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            mcmc.print_summary()
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "mu" in output
        assert "r_hat" in output

    def test_nuts_2chains_diagnostics_scalar(self, observed_data):
        kernel = NUTSKernel()
        mcmc = MCMC(
            kernel, num_warmup=100, num_samples=200,
            num_chains=2, step_size=0.05, max_tree_depth=6,
        )
        mcmc.run(jax.random.PRNGKey(12), simple_normal_model, observed_data)

        d = mcmc.diagnostics
        for key in ("mean_accept_prob", "mean_tree_depth"):
            if key in d:
                assert np.ndim(d[key]) == 0, f"{key}: expected scalar"
        for key in ("num_divergences", "num_max_treedepth"):
            if key in d:
                assert np.ndim(d[key]) == 0, f"{key}: expected scalar"


class TestMCMCMultiChainPosteriorRecovery:
    """Posterior recovery: mu posterior mean should be close to true value."""

    def test_hmc_2chains_recovery(self, observed_data):
        kernel = HMCKernel()
        mcmc = MCMC(
            kernel, num_warmup=200, num_samples=500,
            num_chains=2, step_size=0.05, num_leapfrog=15,
        )
        mcmc.run(jax.random.PRNGKey(20), simple_normal_model, observed_data)

        samples = mcmc.get_samples()
        mu_mean = float(np.mean(samples["mu"]))
        assert abs(mu_mean - 3.0) < 0.5, f"mu posterior mean {mu_mean:.3f} far from true 3.0"

        by_chain = mcmc.get_samples_by_chain()
        from hyperion_diagnostics.metrics import r_hat
        rh = r_hat(by_chain["mu"])
        assert rh < 1.1, f"R-hat={rh:.3f} too high"


class TestMCMCMultiChainPredictive:
    """Predictive with multi-chain posterior."""

    def test_posterior_predictive_from_multichain(self, observed_data):
        kernel = HMCKernel()
        mcmc = MCMC(
            kernel, num_warmup=100, num_samples=200,
            num_chains=2, step_size=0.05, num_leapfrog=10,
        )
        mcmc.run(jax.random.PRNGKey(30), simple_normal_model, observed_data)

        post = mcmc.get_samples()
        pred = Predictive(
            simple_normal_model,
            posterior_samples=post,
            num_samples=50,
        )
        pp = pred(jax.random.PRNGKey(31))
        assert "obs" in pp or "mu" in pp
        first_key = next(iter(pp))
        assert pp[first_key].shape[0] == 50


class TestSummaryTableMultiChainESS:
    """Verify summary_table uses per-chain ESS when samples_by_chain provided."""

    def test_ess_multichain_vs_concatenated(self):
        from hyperion_diagnostics.metrics import summary_table

        rng = np.random.RandomState(42)
        chain1 = rng.normal(0, 1, 500)
        chain2 = rng.normal(0, 1, 500)
        concatenated = np.concatenate([chain1, chain2])

        by_chain_arr = np.stack([chain1, chain2])

        table_no_chain = summary_table({"x": concatenated})
        table_with_chain = summary_table(
            {"x": concatenated},
            samples_by_chain={"x": by_chain_arr},
        )

        ess_no = table_no_chain["x"]["ess"]
        ess_with = table_with_chain["x"]["ess"]
        assert ess_with >= ess_no * 0.8, (
            f"Multi-chain ESS ({ess_with:.0f}) should be at least ~80% of "
            f"concatenated ESS ({ess_no:.0f}) for IID chains"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
