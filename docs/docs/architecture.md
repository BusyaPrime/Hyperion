# Архитектура

## Pipeline

```
@model → Trace Handlers → IR Graph → JAX Backend → Inference → Diagnostics
```

### 1. DSL (`hyperion_dsl`)

Модель описывается как обычная Python-функция. `sample()` объявляет
случайные величины, `plate()` задаёт повторения, `factor()` добавляет
произвольные лог-веса.

### 2. Trace (`hyperion_trace`)

Effect handler system по мотивам Pyro/NumPyro. Четыре мессенджера:

- **TraceMessenger** — записывает все `sample()` вызовы в trace
- **SubstituteMessenger** — подставляет значения вместо семплирования
- **ReplayMessenger** — воспроизводит trace из предыдущего прогона
- **BlockMessenger** — скрывает указанные сайты

### 3. IR (`hyperion_ir`)

Промежуточное представление модели в виде DAG:

- **IRGraph** — направленный ациклический граф нод
- **ModelCompiler** — trace → IR
- **IROptimizer** — CSE, dead node elimination, topological reorder

### 4. Backend (`hyperion_backends`)

`JAXBackend` строит `potential_fn` — JIT-compiled функцию `R^n → R`,
которая считает log-joint probability. Автоматически применяет
transforms для constrained параметров (Positive → Exp, Simplex → StickBreaking).

### 5. Inference (`hyperion_inference`)

Шесть движков, каждый со своим подходом:

| Engine | Метод | JAX primitives |
|--------|-------|----------------|
| HMC | Leapfrog + MH | `lax.scan` |
| NUTS | Bidirectional tree | `lax.while_loop`, `fori_loop` |
| SMC | Tempering + MALA | `vmap` по партиклам |
| VI | ELBO + reparam | `value_and_grad`, optax |
| Flows | RealNVP coupling | optax, affine layers |
| Laplace | MAP + Hessian | `jax.hessian` |

### 6. Diagnostics (`hyperion_diagnostics`)

- ESS (effective sample size) — single-chain и multi-chain
- R-hat, split R-hat — сходимость цепей
- BFMI — Bayesian fraction of missing information
- Autocorrelation — степень зависимости между сэмплами
- `summary_table` — таблица со всеми метриками
