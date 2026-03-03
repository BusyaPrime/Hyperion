"""HYPERION Inference — inference-движки для вероятностных моделей.

HMC, NUTS, SMC, VI, Flows, Laplace — всё для того, чтобы вытащить постериор из модели.

Три API:
  1. Функциональный (hmc_sample, nuts_sample, ...) — чистые JIT-compiled функции
  2. Класс-обёртки (HMCKernel, NUTSKernel, ...) — для ExperimentRunner
  3. High-level (MCMC, Predictive) — для удобного пользовательского API

Используй функциональный API если нужна скорость. Класс-обёртки — для удобства. Разработано в НИУ ИТМО.
"""

from hyperion_inference.base import InferenceEngine, InferenceState, InferenceResult
from hyperion_inference.warmup import (
    WarmupState, make_warmup_state,
    dual_averaging_update, welford_update,
    find_reasonable_step_size, finalize_mass,
    build_adaptation_schedule,
    apply_inv_mass, kinetic_energy, sample_momentum,
)
from hyperion_inference.hmc import (
    HMCKernel, HMCState,
    hmc_sample, hmc_sample_chains,
    _leapfrog, _hmc_step,
)
from hyperion_inference.nuts import (
    NUTSKernel, NUTSState,
    nuts_sample, nuts_sample_chains,
    _nuts_step,
)
from hyperion_inference.smc import SMCEngine, SMCState
from hyperion_inference.vi import VIEngine, VIState
from hyperion_inference.flows import FlowsEngine
from hyperion_inference.laplace import LaplaceApproximation
from hyperion_inference.mcmc import MCMC
from hyperion_inference.predictive import Predictive

__all__ = [
    "InferenceEngine", "InferenceState", "InferenceResult",
    # Warmup
    "WarmupState", "make_warmup_state",
    "dual_averaging_update", "welford_update",
    "find_reasonable_step_size", "finalize_mass",
    "build_adaptation_schedule",
    "apply_inv_mass", "kinetic_energy", "sample_momentum",
    # HMC
    "HMCKernel", "HMCState",
    "hmc_sample", "hmc_sample_chains",
    # NUTS
    "NUTSKernel", "NUTSState",
    "nuts_sample", "nuts_sample_chains",
    # Engines
    "SMCEngine", "SMCState",
    "VIEngine", "VIState",
    "FlowsEngine",
    "LaplaceApproximation",
    # High-level API
    "MCMC",
    "Predictive",
]
