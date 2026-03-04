# HYPERION

Probabilistic Programming Language на чистом JAX.

Полный стек байесовского инференса: от описания модели в три строки
до 4-chain NUTS с dense mass matrix и автоматической диагностикой.

## Возможности

- **DSL** — `@model`, `sample()`, `plate()`, `factor()`, 23 распределения
- **6 inference engines** — HMC, NUTS, SMC, VI, Normalizing Flows, Laplace
- **JAX-native** — JIT, autodiff, vmap, lax control flow
- **Multi-chain** — `jax.vmap` по цепям, R-hat, per-chain ESS
- **Диагностика** — ESS, R-hat, split R-hat, BFMI, autocorrelation, summary_table
- **IR** — промежуточное представление с CSE, dead node elimination

## Быстрый пример

```python
from hyperion_dsl import model, sample
from hyperion_dsl.distributions import Normal
from hyperion_inference.mcmc import MCMC
from hyperion_inference.hmc import HMCKernel

@model
def my_model():
    mu = sample("mu", Normal(0.0, 10.0))
    sample("obs", Normal(mu, 1.0))

mcmc = MCMC(HMCKernel(), num_samples=2000, num_warmup=500)
mcmc.run(my_model, data={"obs": data}, rng_key=jrandom.PRNGKey(42))
mcmc.print_summary()
```
