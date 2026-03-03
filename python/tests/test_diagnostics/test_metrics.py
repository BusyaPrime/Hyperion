"""Тесты для метрик диагностики MCMC.

R-hat должен быть < 1.1. Если нет — твои цепи ещё гуляют.
Проверяем что ESS не ноль — потому что zero-ESS это диагноз, а не метрика. Лабы НИУ ИТМО.
Автокорреляция — бич MCMC. IID сэмплы должны иметь ACF ~ 0 на больших лагах.
"""

import numpy as np
import pytest

from hyperion_diagnostics.metrics import (
    autocorrelation,
    effective_sample_size,
    r_hat,
    split_r_hat,
    summary_table,
)


class TestAutocorrelation:
    """ACF: как быстро забывается прошлое. IID = быстро к нулю."""

    def test_iid_samples(self):
        # IID сэмплы — ACF[0]=1, на больших лагах ~0
        np.random.seed(42)
        samples = np.random.normal(size=1000)
        acf = autocorrelation(samples, max_lag=20)
        assert abs(acf[0] - 1.0) < 0.05
        assert all(abs(a) < 0.1 for a in acf[5:])

    def test_correlated_samples(self):
        # Random walk — сильная автокорреляция, ACF[1] близок к 1
        np.random.seed(42)
        samples = np.cumsum(np.random.normal(size=1000))
        acf = autocorrelation(samples, max_lag=20)
        assert acf[1] > 0.9


class TestESS:
    """ESS — сколько РЕАЛЬНО независимых сэмплов. Спойлер: меньше чем кажется."""

    def test_iid_high_ess(self):
        # IID сэмплы — высокий ESS (оценка может быть консервативной)
        np.random.seed(42)
        samples = np.random.normal(size=1000)
        ess = effective_sample_size(samples)
        assert ess > 200

    def test_correlated_low_ess(self):
        # Проверяем что ESS не ноль — потому что zero-ESS это диагноз, а не метрика
        np.random.seed(42)
        samples = np.cumsum(np.random.normal(size=1000))
        ess = effective_sample_size(samples)
        assert ess < 100


class TestRHat:
    """R-hat: цепи сошлись или нет. R≈1 — ок, R>1.1 — продолжаем сэмплить."""

    def test_converged_chains(self):
        # R-hat должен быть < 1.1. Сходившиеся цепи — R ≈ 1
        np.random.seed(42)
        chains = np.random.normal(size=(4, 500))
        rh = r_hat(chains)
        assert abs(rh - 1.0) < 0.05

    def test_diverged_chains(self):
        # Разошедшиеся цепи — R >> 1
        np.random.seed(42)
        chains = np.array([
            np.random.normal(0, 1, 500),
            np.random.normal(0, 1, 500),
            np.random.normal(5, 1, 500),
            np.random.normal(5, 1, 500),
        ])
        rh = r_hat(chains)
        assert rh > 1.5

    def test_single_chain_returns_nan(self):
        # Одна цепь — R-hat не определён, возвращаем NaN
        chains = np.random.normal(size=(1, 500))
        rh = r_hat(chains)
        assert np.isnan(rh)


class TestSplitRHat:
    """Split R-hat — консервативнее. Ловит нестационарность внутри цепи."""

    def test_converged(self):
        np.random.seed(42)
        chains = np.random.normal(size=(2, 1000))
        srh = split_r_hat(chains)
        assert abs(srh - 1.0) < 0.1


class TestSummaryTable:
    """Сводка: mean, std, HDI, ESS. Всё что нужно для отчёта."""

    def test_basic_summary(self):
        np.random.seed(42)
        samples = {"mu": np.random.normal(2.0, 0.5, 1000)}
        table = summary_table(samples)
        assert "mu" in table
        assert abs(table["mu"]["mean"] - 2.0) < 0.1
        assert "ess" in table["mu"]


class TestPrintSummary:
    """print_summary печатает таблицу в stdout."""

    def test_print_runs(self, capsys):
        from hyperion_diagnostics.metrics import print_summary
        np.random.seed(42)
        samples = {"alpha": np.random.normal(1.0, 0.3, 500)}
        print_summary(samples)
        captured = capsys.readouterr()
        assert "alpha" in captured.out
        assert "mean" in captured.out

    def test_print_multivar(self, capsys):
        from hyperion_diagnostics.metrics import print_summary
        np.random.seed(0)
        samples = {"x": np.random.normal(size=(200, 3))}
        print_summary(samples)
        captured = capsys.readouterr()
        assert "x[0]" in captured.out
        assert "x[2]" in captured.out


class TestComputeAllDiagnosticsMultichain:
    """compute_all_diagnostics с multi-chain R-hat."""

    def test_rhat_included_for_multichain(self):
        from hyperion_diagnostics.metrics import compute_all_diagnostics
        from hyperion_inference.base import InferenceResult

        np.random.seed(42)
        chains = np.random.normal(size=(4, 200))
        flat = chains.reshape(-1)
        result = InferenceResult(
            samples={"mu": flat},
            diagnostics={},
            num_chains=4,
            samples_by_chain={"mu": chains},
        )
        diag = compute_all_diagnostics(result)
        assert "mu/r_hat" in diag
        assert "mu/split_r_hat" in diag
        assert "mu/ess_multi" in diag
        assert abs(diag["mu/r_hat"] - 1.0) < 0.1

    def test_no_rhat_for_single_chain(self):
        from hyperion_diagnostics.metrics import compute_all_diagnostics
        from hyperion_inference.base import InferenceResult

        np.random.seed(0)
        result = InferenceResult(
            samples={"mu": np.random.normal(size=500)},
            diagnostics={},
        )
        diag = compute_all_diagnostics(result)
        assert "mu/r_hat" not in diag


class TestBFMI:
    """BFMI — Bayesian Fraction of Missing Information."""

    def test_bfmi_iid_energies(self):
        from hyperion_diagnostics.metrics import energy_diagnostic
        np.random.seed(42)
        energies = np.random.normal(0.0, 1.0, size=1000)
        result = energy_diagnostic(energies)
        assert "bfmi" in result
        assert result["bfmi"] > 0
        assert "mean_energy" in result
        assert "std_energy" in result

    def test_bfmi_constant_energies(self):
        from hyperion_diagnostics.metrics import energy_diagnostic
        energies = np.ones(100)
        result = energy_diagnostic(energies)
        assert np.isnan(result["bfmi"])

    def test_bfmi_short_chain(self):
        from hyperion_diagnostics.metrics import energy_diagnostic
        result = energy_diagnostic(np.array([1.0]))
        assert np.isnan(result["bfmi"])

    def test_bfmi_random_walk_low(self):
        from hyperion_diagnostics.metrics import energy_diagnostic
        np.random.seed(0)
        energies = np.cumsum(np.random.normal(0, 0.01, size=1000))
        result = energy_diagnostic(energies)
        assert result["bfmi"] < 0.5


class TestPPC:
    """Posterior Predictive Checks."""

    def test_ppc_basic(self):
        from hyperion_diagnostics.ppc import posterior_predictive_check, ppc_summary
        from hyperion_dsl.primitives import sample
        from hyperion_dsl.distributions import Normal
        from hyperion_dsl.model import model
        import jax

        @model
        def simple():
            mu = sample("mu", Normal(0.0, 1.0))
            sample("y", Normal(mu, 0.5), obs=np.array(1.0))

        posterior = {"mu": np.random.normal(1.0, 0.2, size=50)}
        data = {"y": np.array(1.0)}
        ppc = posterior_predictive_check(
            simple, posterior, data, jax.random.PRNGKey(0),
        )
        assert "y" in ppc
        assert ppc["y"].shape[0] == 50

    def test_ppc_summary_metrics(self):
        from hyperion_diagnostics.ppc import ppc_summary
        np.random.seed(42)
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.random.normal(
            loc=np.array([1.0, 2.0, 3.0]), scale=0.5, size=(100, 3)
        )
        result = ppc_summary(observed, predicted)
        assert "mae" in result
        assert "rmse" in result
        assert "bayesian_p_value" in result
        assert "coverage_90" in result
        assert result["mae"] < 1.0
        assert 0.2 < result["bayesian_p_value"] < 0.8

    def test_ppc_vectorized_mode(self):
        from hyperion_diagnostics.ppc import posterior_predictive_check
        from hyperion_dsl.primitives import sample
        from hyperion_dsl.distributions import Normal
        from hyperion_dsl.model import model
        import jax

        @model
        def simple():
            mu = sample("mu", Normal(0.0, 1.0))
            sample("y", Normal(mu, 0.5), obs=np.array(1.0))

        posterior = {"mu": np.random.normal(1.0, 0.2, size=20)}
        data = {"y": np.array(1.0)}
        ppc = posterior_predictive_check(
            simple, posterior, data, jax.random.PRNGKey(1),
            vectorized=True,
        )
        assert "y" in ppc
        assert ppc["y"].shape[0] == 20
