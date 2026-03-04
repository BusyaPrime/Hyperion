# HYPERION

[![CI](https://github.com/BusyaPrime/Hyperion/actions/workflows/ci.yml/badge.svg)](https://github.com/BusyaPrime/Hyperion/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Probabilistic Programming Language на чистом JAX. Полный стек: от DSL до продакшн-ready инференса.

Проект собран так, чтобы один человек мог прогнать байесовский инференс за минуту,
а команда — развернуть на GPU-кластере без переписывания. Без магии, без скрытых зависимостей,
без «у меня на машине работает».

---

## Архитектура

Шесть слоёв, каждый делает одно дело и делает его нормально:

```
                    ┌──────────────────────────────────────────┐
                    │            @model + DSL                  │
                    │   sample, plate, factor, deterministic   │
                    └────────────────────┬─────────────────────┘
                                         │
                              трассировка модели
                                         │
                    ┌────────────────────▼─────────────────────┐
                    │           Trace / Handlers                │
                    │  TraceMessenger, Substitute, Replay,      │
                    │  Block — composable effect system         │
                    └────────────────────┬─────────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         │                               │
              ┌──────────▼──────────┐         ┌──────────▼──────────┐
              │     IR + Compiler   │         │    JAX Backend       │
              │  IRGraph, Optimizer │         │  potential_fn (JIT)  │
              │  CSE, dead nodes,   │         │  flatten/unflatten   │
              │  topo reorder       │         │  transforms, grads   │
              └──────────┬──────────┘         └──────────┬──────────┘
                         │                               │
                         └───────────────┬───────────────┘
                                         │
                              potential + gradients
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │                  Inference Engines                   │
              │                                                     │
              │  ┌─────────┐ ┌──────┐ ┌─────┐ ┌────┐ ┌─────────┐  │
              │  │   HMC   │ │ NUTS │ │ SMC │ │ VI │ │  Flows  │  │
              │  │leapfrog │ │ tree │ │MALA │ │ELBO│ │ RealNVP │  │
              │  │lax.scan │ │while │ │vmap │ │Adam│ │coupling │  │
              │  └─────────┘ └──────┘ └─────┘ └────┘ └─────────┘  │
              │          ┌──────────┐                               │
              │          │ Laplace  │    Warmup: dual averaging     │
              │          │ MAP+Hess │    + Welford + dense mass     │
              │          └──────────┘                               │
              └────────────────────────┬───────────────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
              ┌──────────▼──────────┐     ┌──────────▼──────────┐
              │    Diagnostics      │     │   High-level API     │
              │  ESS, R-hat, BFMI   │     │  MCMC, Predictive    │
              │  autocorrelation    │     │  print_summary()     │
              │  summary_table      │     │  multi-chain vmap    │
              └─────────────────────┘     └──────────────────────┘
```

**DSL** описывает модель как обычную Python-функцию. **Trace** перехватывает `sample()` вызовы
через стек хендлеров. **Backend** строит `potential_fn` — JIT-compiled log-joint, который
видит только `jnp`-операции и компилирует в XLA. **Inference** берёт этот потенциал и гоняет
цепи. **Diagnostics** проверяет что цепи не врут.

---

## Быстрый старт

### Установка

```bash
pip install jax jaxlib optax
cd python/
```

### Минимальный пример: Normal-Normal

```python
import jax.numpy as jnp
import jax.random as jrandom

from hyperion_dsl import model, sample
from hyperion_dsl.distributions import Normal
from hyperion_inference.mcmc import MCMC
from hyperion_inference.hmc import HMCKernel

# Описываем модель — обычная Python-функция
@model
def coin_model():
    mu = sample("mu", Normal(0.0, 10.0))
    sample("obs", Normal(mu, 1.0))

# Данные
data = jnp.array([2.1, 1.9, 2.3, 2.0, 1.8])

# Инференс — три строки
mcmc = MCMC(HMCKernel(), num_samples=2000, num_warmup=500)
mcmc.run(coin_model, data={"obs": data}, rng_key=jrandom.PRNGKey(42))
mcmc.print_summary()
```

Вывод:
```
=== MCMC Summary ===
Kernel: HMCKernel | Samples: 2000 | Chains: 1
Mean accept prob: 0.912

           mean    std    2.5%   97.5%    ESS   R-hat
    mu    2.017  0.447   1.152   2.908  1847.0   1.000
```

---

## Пример: Multi-chain NUTS

```python
from hyperion_inference.nuts import NUTSKernel

mcmc = MCMC(NUTSKernel(), num_samples=2000, num_warmup=1000, num_chains=4)
mcmc.run(coin_model, data={"obs": data}, rng_key=jrandom.PRNGKey(0))

# Posterior predictive
from hyperion_inference.predictive import Predictive

pred = Predictive(coin_model, posterior_samples=mcmc.get_samples())
ppc = pred(jrandom.PRNGKey(1))
```

---

## Пример: Байесовская линейная регрессия

```python
@model
def linear_regression(x, y_obs):
    alpha = sample("alpha", Normal(0.0, 10.0))
    beta = sample("beta", Normal(0.0, 10.0))
    sigma = sample("sigma", HalfNormal(1.0))
    mu = alpha + beta * x
    sample("y", Normal(mu, sigma), obs=y_obs)
```

---

## Пример: Иерархическая модель

```python
from hyperion_dsl import plate

@model
def hierarchical(y_obs, group_ids, num_groups):
    mu_pop = sample("mu_pop", Normal(0.0, 10.0))
    sigma_pop = sample("sigma_pop", HalfNormal(5.0))
    sigma_obs = sample("sigma_obs", HalfNormal(1.0))

    with plate("groups", num_groups):
        mu_group = sample("mu_group", Normal(mu_pop, sigma_pop))

    mu_y = mu_group[group_ids]
    sample("y", Normal(mu_y, sigma_obs), obs=y_obs)
```

---

## Структура модулей

```
python/
├── hyperion_dsl/            # DSL: @model, sample(), plate(), transforms
│   ├── model.py             #   декоратор @model — оборачивает функцию в Model
│   ├── primitives.py        #   sample, plate, factor, deterministic, param
│   ├── distributions.py     #   23 распределения (Normal → GaussianProcess)
│   ├── transforms.py        #   12 биективных трансформов + log_det_jacobian
│   └── constraints.py       #   9 ограничений (Positive, Simplex, Bounded, ...)
│
├── hyperion_trace/          # Effect handler system
│   ├── handler.py           #   базовый TraceHandler со стеком
│   ├── handlers.py          #   Trace, Substitute, Replay, Block
│   └── trace.py             #   trace_model() — прогон модели через хендлеры
│
├── hyperion_ir/             # Intermediate Representation
│   ├── ir.py                #   IRGraph, IRNode — граф вычислений
│   ├── compiler.py          #   ModelCompiler — trace → IR
│   └── optimizer.py         #   CSE, dead node elimination, topo reorder
│
├── hyperion_backends/       # Backend
│   ├── base.py              #   AbstractBackend — контракт
│   └── jax_backend.py       #   JAXBackend — potential_fn, flatten/unflatten, JIT
│
├── hyperion_inference/      # Inference engines
│   ├── hmc.py               #   HMC: leapfrog через lax.scan, dense mass
│   ├── nuts.py              #   NUTS: bidirectional tree doubling, while_loop
│   ├── smc.py               #   SMC: темперинг + MALA rejuvenation
│   ├── vi.py                #   VI: diagonal / low-rank / full-rank Gaussian
│   ├── flows.py             #   Normalizing Flows: RealNVP, affine coupling
│   ├── laplace.py           #   Laplace: MAP + Hessian
│   ├── warmup.py            #   Dual averaging, Welford, dense mass, step_size
│   ├── mcmc.py              #   MCMC high-level API, print_summary()
│   └── predictive.py        #   Predictive class — posterior predictive
│
├── hyperion_diagnostics/    # Диагностика
│   ├── metrics.py           #   ESS, R-hat, split R-hat, BFMI, autocorrelation
│   ├── ppc.py               #   Posterior predictive checks
│   └── report.py            #   Markdown-отчёт
│
├── hyperion_graph/          # Визуализация графа модели
│   └── graph_builder.py     #   GraphBuilder — ноды, рёбра, topo order
│
├── hyperion_exp/            # Эксперименты
│   ├── runner.py            #   ExperimentRunner — прогон + сравнение
│   └── serialization.py     #   save/load InferenceResult (numpy + JSON)
│
├── hyperion_api/            # gRPC сервер
│   └── server.py            #   мост между Python и внешними сервисами
│
├── examples/                # Демо-модели
│   └── demo_models.py       #   6 моделей из спецификации
│
└── tests/                   # 292 теста
    ├── test_bijectors/      #   round-trip трансформов, log_det_jacobian
    ├── test_diagnostics/    #   ESS, R-hat, autocorrelation
    ├── test_distributions/  #   KS-тесты, edge cases, log_prob
    ├── test_inference/      #   posterior recovery (HMC, NUTS, VI, SMC, Laplace, Flows)
    ├── test_integration/    #   end-to-end, multi-chain, comprehensive
    └── test_trace/          #   трассировка, хендлеры, substitution
```

---

## Распределения (23)

| Категория | Распределения |
|-----------|--------------|
| **Непрерывные** | `Normal`, `HalfNormal`, `Cauchy`, `HalfCauchy`, `Laplace`, `StudentT`, `LogNormal` |
| **Положительные** | `Gamma`, `Exponential`, `InverseGamma` |
| **Ограниченные** | `Beta`, `Uniform` |
| **Многомерные** | `MultivariateNormal`, `Dirichlet`, `Wishart`, `LKJCholesky`, `GaussianProcess` |
| **Дискретные** | `Bernoulli`, `Categorical`, `Poisson`, `Binomial`, `Multinomial` |
| **Специальные** | `Delta`, `Horseshoe` |

Каждое распределение реализует `log_prob()`, `sample()`, `support`, автоматический `default_transform`
для unconstrained-пространства.

---

## Трансформы (12)

| Трансформ | Откуда → Куда | Зачем |
|-----------|---------------|-------|
| `ExpTransform` | R → R+ | HalfNormal, Gamma, Exponential |
| `SigmoidTransform` | R → (0,1) | Beta |
| `SoftplusTransform` | R → R+ | мягкая альтернатива exp |
| `StickBreakingTransform` | R^(K-1) → Simplex^K | Dirichlet |
| `CholeskyTransform` | R^(n(n+1)/2) → L | Wishart |
| `CorrCholeskyTransform` | R^(n(n-1)/2) → CorrCholesky | LKJCholesky |
| `BoundedTransform` | R → (a,b) | Uniform |
| `ComposeTransform` | цепочка | комбинируем |
| `PowerTransform` | x → x^p | степенные |
| `AffineTransform` | x → ax + b | сдвиг + масштаб |
| `IdentityTransform` | x → x | placeholder |
| `LogTransform` | R+ → R | обратный exp |

Все трансформы реализуют `forward()`, `inverse()`, `log_det_jacobian()`.
Round-trip корректность проверена тестами.

---

## Inference Engines (6)

### HMC
Hamiltonian Monte Carlo. Leapfrog через `jax.lax.scan`. Windowed warmup
(Stan-style schedule), dual averaging для step_size, Welford для mass matrix.
Dense mass через Cholesky. Multi-chain через `jax.vmap`.

### NUTS
No-U-Turn Sampler. Bidirectional tree doubling через `jax.lax.while_loop` и `fori_loop`.
Multinomial weighting (Betancourt 2017). U-turn criterion на обоих концах траектории.
`max_treedepth` warning. Multi-chain через `vmap`.

### SMC
Sequential Monte Carlo. Adaptive tempering (бисекция по ESS).
MALA rejuvenation (Metropolis-adjusted Langevin). Systematic / multinomial / stratified
resampling. `vmap` по партиклам — без Python-циклов.

### VI
Variational Inference. Mean-field (diagonal), low-rank, full-rank Gaussian.
ELBO через reparameterization trick. Cosine / exponential LR schedule.
Early stopping по patience. Optional natural gradient.

### Normalizing Flows
RealNVP: affine coupling layers + ActNorm. Xavier init.
ELBO optimization через optax. Обратимые трансформации для сложных постериоров.

### Laplace
MAP через optax (Adam + line search fallback). Hessian или Fisher.
Гауссовская аппроксимация постериора вокруг MAP.

---

## Диагностика

```python
from hyperion_diagnostics.metrics import (
    effective_sample_size,
    r_hat,
    split_r_hat,
    bfmi,
    autocorrelation,
    summary_table,
)

# summary_table — таблица с mean, std, quantiles, ESS, R-hat
table = summary_table(samples, prob=0.95)

# Multi-chain ESS
table = summary_table(samples, samples_by_chain=by_chain)
```

---

## Тестирование

```bash
# Полный прогон (292 теста, ~4 минуты)
cd python/
python -m pytest tests/ -v

# Только posterior recovery
python -m pytest tests/test_inference/test_posterior_recovery.py -v

# Только multi-chain
python -m pytest tests/test_integration/test_mcmc_multichain.py -v
```

**292 теста**, 0 failures. Покрытие:
- **Posterior recovery**: HMC, NUTS, VI, SMC, Laplace, Flows — 1D и 5D
- **Constrained models**: HalfNormal (positive), Dirichlet (simplex), Beta
- **Multi-chain**: 2-chain HMC/NUTS, print_summary, diagnostics, predictive
- **Transforms**: round-trip для всех 12, log_det_jacobian
- **Edge cases**: invalid params, NaN guards, boundary values
- **IR**: CSE, dead nodes, topological reorder
- **Serialization**: save/load с samples_by_chain

---

## Стек

- **JAX** — autodiff, JIT, vmap, lax control flow
- **optax** — оптимизаторы для VI/Flows/Laplace
- **pytest** — тесты

---

## Лицензия

MIT
