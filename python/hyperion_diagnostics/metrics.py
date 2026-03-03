"""Статистическая диагностика для MCMC-сэмплов.

R-hat < 1.01 — цепи сошлись. > 1.1 — ты гонишь, запускай дольше.
ESS показывает сколько РЕАЛЬНО независимых сэмплов ты получил. Спойлер: меньше чем кажется.
Автокорреляция — бич MCMC. Чем быстрее падает к нулю, тем лучше.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def autocorrelation(samples: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Автокорреляционная функция для 1D цепи.

    ACF(lag) = E[(x_t - mu)(x_{t+lag} - mu)] / var. Показывает, как быстро забывается
    прошлое — чем быстрее падает к нулю, тем лучше миксер.

    Args:
        samples: 1D массив сэмплов.
        max_lag: Максимальный лаг. По умолчанию n // 2.

    Returns:
        Массив ACF для лагов 0..max_lag-1.
    """
    samples = np.asarray(samples)
    n = len(samples)
    if max_lag is None:
        max_lag = n // 2

    mean = np.mean(samples)
    var = np.var(samples)
    if var == 0:
        return np.zeros(max_lag)

    centered = samples - mean
    acf = np.correlate(centered, centered, mode="full")[n - 1:]
    acf = acf[:max_lag] / (var * n)
    return acf


def effective_sample_size(samples: np.ndarray) -> float:
    """ESS по initial positive sequence (Geyer 1992, Stan).

    ESS = n / tau, где tau — суммарная автокорреляция. Показывает, сколько реально
    независимых сэмплов у тебя есть. Спойлер: почти всегда меньше, чем кажется.

    Args:
        samples: 1D массив сэмплов.

    Returns:
        Эффективный размер выборки (1..n).
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n < 4:
        return float(n)

    acf = autocorrelation(samples)
    # Geyer (1992): τ = -1 + 2·Σ Γ_m, where Γ_m = ρ_{2m} + ρ_{2m+1}
    tau = -1.0
    for i in range(0, len(acf) - 1, 2):
        gamma = acf[i] + acf[i + 1] if i + 1 < len(acf) else acf[i]
        if gamma < 0:
            break
        tau += 2.0 * gamma

    ess = n / tau
    return max(1.0, min(ess, float(n)))


def effective_sample_size_multichain(
    chains: np.ndarray,
) -> float:
    """ESS для нескольких цепей. Shape: (num_chains, num_samples). Суммируем по цепям."""
    chains = np.asarray(chains, dtype=np.float64)
    num_chains, num_samples = chains.shape

    chain_ess = [effective_sample_size(chains[c]) for c in range(num_chains)]
    return sum(chain_ess)


def r_hat(chains: np.ndarray) -> float:
    """R-hat (potential scale reduction factor).

    Сравниваем variance между цепями (B) и внутри (W). R = sqrt(var_hat/W).
    R ≈ 1 — цепи сошлись. R > 1.1 — гонишь, запускай дольше или чини модель.

    Args:
        chains: 2D массив (num_chains, num_samples).

    Returns:
        R-hat. nan если цепей < 2.
    """
    chains = np.asarray(chains, dtype=np.float64)
    num_chains, num_samples = chains.shape

    if num_chains < 2:
        return float("nan")

    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)

    grand_mean = np.mean(chain_means)
    B = num_samples * np.var(chain_means, ddof=1)
    W = np.mean(chain_vars)

    if W == 0:
        return float("nan")

    var_hat = (1.0 - 1.0 / num_samples) * W + B / num_samples
    return float(np.sqrt(var_hat / W))


def split_r_hat(chains: np.ndarray) -> float:
    """Split R-hat: режем каждую цепь пополам, считаем R-hat на 2x цепях.

    Консервативнее обычного R-hat — ловит нестационарность внутри цепи.
    Рекомендуется для серьёзной диагностики.

    Args:
        chains: 2D массив (num_chains, num_samples).

    Returns:
        Split R-hat.
    """
    chains = np.asarray(chains, dtype=np.float64)
    num_chains, num_samples = chains.shape
    half = num_samples // 2

    split_chains = np.concatenate([
        chains[:, :half],
        chains[:, half:2 * half],
    ], axis=0)

    return r_hat(split_chains)


def acceptance_rate(accept_probs: np.ndarray) -> float:
    """Средний acceptance rate. 0.65+ — ок, 0.2 — плохо, настраивай step size."""
    return float(np.mean(np.asarray(accept_probs)))


def divergence_count(divergences: Any) -> int:
    """Считаем divergent transitions. Если > 0 — HMC/NUTS поехал не туда."""
    if isinstance(divergences, (int, float)):
        return int(divergences)
    return int(np.sum(np.asarray(divergences)))


def energy_diagnostic(energies: np.ndarray) -> dict[str, float]:
    """BFMI (Bayesian Fraction of Missing Information) и энергетическая диагностика.

    BFMI = var(energy_diff) / var(energy). < 0.3 — проблемы с sampling, geometry кривая,
    возможно нужно адаптировать step size или упростить модель.

    Args:
        energies: Массив энергий (negative log joint) по траектории.

    Returns:
        dict с ключами bfmi, mean_energy, std_energy.
    """
    energies = np.asarray(energies, dtype=np.float64)
    if len(energies) < 2:
        return {"bfmi": float("nan")}

    energy_diff = np.diff(energies)
    var_energies = np.var(energies)
    bfmi = np.var(energy_diff) / var_energies if var_energies > 0 else float("nan")
    return {
        "bfmi": float(bfmi),
        "mean_energy": float(np.mean(energies)),
        "std_energy": float(np.std(energies)),
    }


def _flatten_to_1d_samples(
    chain_samples_np: np.ndarray,
    sample_axis: int = 0,
) -> list[tuple[str, np.ndarray]]:
    """Normalize array to list of (suffix, 1D samples) pairs.

    sample_axis: which axis indexes draws (0 = rows are samples, default).
    For 2D with sample_axis=0: shape (n_samples, n_params) -> per-param ESS.
    For 2D with sample_axis=1: shape (n_chains, n_samples) -> flatten chains first.
    For 3D: shape (n_chains, n_samples, n_params) -> flatten chains, per-param.
    """
    if chain_samples_np.ndim == 1:
        return [("", chain_samples_np)]

    if chain_samples_np.ndim == 2:
        if sample_axis == 0:
            return [
                (f"[{d}]", chain_samples_np[:, d])
                for d in range(chain_samples_np.shape[1])
            ]
        else:
            flat = chain_samples_np.reshape(-1)
            return [("", flat)]

    if chain_samples_np.ndim == 3:
        n_chains, n_samples, n_params = chain_samples_np.shape
        flat = chain_samples_np.reshape(n_chains * n_samples, n_params)
        return [
            (f"[{d}]", flat[:, d])
            for d in range(n_params)
        ]

    return []


def compute_all_diagnostics(
    inference_result: Any,
    sample_axis: int = 0,
) -> dict[str, Any]:
    """Собираем всю диагностику из InferenceResult.

    sample_axis: axis of individual draws.
      0 (default) — shape (n_samples,) or (n_samples, n_params)
      1 — shape (n_chains, n_samples) or (n_chains, n_samples, n_params)

    Multi-chain: если result.samples_by_chain доступен, считаем R-hat и split R-hat.
    """
    diagnostics = {}

    samples = inference_result.samples
    for name, chain_samples in samples.items():
        chain_samples_np = np.asarray(chain_samples)
        for suffix, arr in _flatten_to_1d_samples(chain_samples_np, sample_axis):
            diagnostics[f"{name}{suffix}/ess"] = effective_sample_size(arr)

    by_chain = getattr(inference_result, "samples_by_chain", None)
    n_chains = getattr(inference_result, "num_chains", 1)
    if by_chain is not None and n_chains > 1:
        for name, chain_arr in by_chain.items():
            chain_arr_np = np.asarray(chain_arr)
            if chain_arr_np.ndim == 2:
                diagnostics[f"{name}/r_hat"] = r_hat(chain_arr_np)
                diagnostics[f"{name}/split_r_hat"] = split_r_hat(chain_arr_np)
                diagnostics[f"{name}/ess_multi"] = effective_sample_size_multichain(chain_arr_np)
            elif chain_arr_np.ndim == 3:
                nc, ns, np_ = chain_arr_np.shape
                for d in range(np_):
                    diagnostics[f"{name}[{d}]/r_hat"] = r_hat(chain_arr_np[:, :, d])
                    diagnostics[f"{name}[{d}]/split_r_hat"] = split_r_hat(chain_arr_np[:, :, d])
                    diagnostics[f"{name}[{d}]/ess_multi"] = effective_sample_size_multichain(chain_arr_np[:, :, d])

    raw = inference_result.diagnostics
    if "accept_probs" in raw:
        diagnostics["accept_rate"] = acceptance_rate(np.array(raw["accept_probs"]))
    if "accept_rate" in raw:
        diagnostics["accept_rate"] = raw["accept_rate"]
    if "num_divergences" in raw:
        diagnostics["num_divergences"] = divergence_count(raw["num_divergences"])
    if "energy" in raw:
        diagnostics.update(energy_diagnostic(np.array(raw["energy"])))

    return diagnostics


def summary_table(
    samples: dict[str, np.ndarray],
    prob: float = 0.9,
    sample_axis: int = 0,
    samples_by_chain: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Сводка по параметрам: mean, std, median, CI, ESS.

    Аналог ArviZ/numpyro summary. Для multi-chain передай samples_by_chain —
    тогда ESS считается корректно (per-chain и суммируется).

    Args:
        samples: {param_name: array сэмплов}.
        prob: Уровень доверительного интервала (по умолчанию 0.9).
        sample_axis: Ось сэмплов (0 = строки, 1 = для multi-chain).
        samples_by_chain: Если есть — ESS считается по цепям и суммируется.

    Returns:
        {param: {mean, std, median, ci_*, ess}}.
    """
    alpha = (1.0 - prob) / 2.0
    table = {}

    for name, samps in samples.items():
        samps = np.asarray(samps, dtype=np.float64)
        for suffix, arr in _flatten_to_1d_samples(samps, sample_axis):
            key = f"{name}{suffix}" if suffix else name
            ess_val = effective_sample_size(arr)

            if samples_by_chain is not None and name in samples_by_chain:
                chain_arr = np.asarray(samples_by_chain[name], dtype=np.float64)
                if chain_arr.ndim == 2:
                    ess_val = effective_sample_size_multichain(chain_arr)
                elif chain_arr.ndim == 3 and suffix:
                    d = int(suffix.strip("[]"))
                    ess_val = effective_sample_size_multichain(chain_arr[:, :, d])

            table[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                f"ci_{alpha:.1%}": float(np.quantile(arr, alpha)),
                f"ci_{1 - alpha:.1%}": float(np.quantile(arr, 1 - alpha)),
                "ess": ess_val,
            }

    return table


def print_summary(
    samples: dict[str, np.ndarray],
    prob: float = 0.9,
    sample_axis: int = 0,
) -> None:
    """Печатает таблицу summary в stdout. Аналог ArviZ/numpyro print_summary."""
    table = summary_table(samples, prob=prob, sample_axis=sample_axis)
    alpha = (1.0 - prob) / 2.0
    ci_lo = f"ci_{alpha:.1%}"
    ci_hi = f"ci_{1 - alpha:.1%}"

    header = f"{'param':>20s} {'mean':>10s} {'std':>10s} {'median':>10s}"
    header += f" {ci_lo:>10s} {ci_hi:>10s} {'ess':>8s}"
    print(header)
    print("-" * len(header))

    for name, stats in table.items():
        row = f"{name:>20s}"
        row += f" {stats['mean']:10.3f}"
        row += f" {stats['std']:10.3f}"
        row += f" {stats['median']:10.3f}"
        row += f" {stats.get(ci_lo, float('nan')):10.3f}"
        row += f" {stats.get(ci_hi, float('nan')):10.3f}"
        row += f" {stats['ess']:8.0f}"
        print(row)
