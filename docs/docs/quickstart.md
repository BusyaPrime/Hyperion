# Quickstart

## Установка

```bash
pip install jax jaxlib optax
git clone https://github.com/BusyaPrime/Hyperion.git
cd Hyperion/python
pip install -e ".[dev]"
```

## Первая модель

Модель — обычная Python-функция с декоратором `@model`.
Внутри используем `sample()` для объявления случайных величин.

```python
import jax.numpy as jnp
import jax.random as jrandom
from hyperion_dsl import model, sample
from hyperion_dsl.distributions import Normal, HalfNormal

@model
def linear_regression(x, y_obs):
    alpha = sample("alpha", Normal(0.0, 10.0))
    beta = sample("beta", Normal(0.0, 10.0))
    sigma = sample("sigma", HalfNormal(1.0))
    mu = alpha + beta * x
    sample("y", Normal(mu, sigma), obs=y_obs)
```

## Запуск инференса

```python
from hyperion_inference.mcmc import MCMC
from hyperion_inference.nuts import NUTSKernel

mcmc = MCMC(NUTSKernel(), num_samples=2000, num_warmup=1000)
mcmc.run(
    linear_regression,
    data={"y": y_data},
    rng_key=jrandom.PRNGKey(0),
    model_args=(x_data,),
)
mcmc.print_summary()
```

## Multi-chain

```python
mcmc = MCMC(NUTSKernel(), num_samples=2000, num_warmup=1000, num_chains=4)
mcmc.run(my_model, data={"obs": data}, rng_key=jrandom.PRNGKey(0))
```

Step size вычисляется один раз вне `vmap` и передаётся всем цепям —
никаких RuntimeWarning, правильная адаптация из коробки.

## Posterior Predictive

```python
from hyperion_inference.predictive import Predictive

pred = Predictive(my_model, posterior_samples=mcmc.get_samples())
ppc = pred(jrandom.PRNGKey(1))
```

## Тесты

```bash
python -m pytest tests/ -v
```
