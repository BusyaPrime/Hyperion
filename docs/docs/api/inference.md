# Inference API

## High-level: MCMC

```python
from hyperion_inference.mcmc import MCMC
from hyperion_inference.hmc import HMCKernel
from hyperion_inference.nuts import NUTSKernel

mcmc = MCMC(
    kernel=NUTSKernel(),
    num_samples=2000,
    num_warmup=1000,
    num_chains=4,
)

mcmc.run(model_fn, data={"obs": y}, rng_key=key)
mcmc.print_summary(prob=0.9)

samples = mcmc.get_samples()            # dict[str, jnp.ndarray]
by_chain = mcmc.get_samples_by_chain()   # dict[str, jnp.ndarray] (chains x samples x ...)
```

## High-level: Predictive

```python
from hyperion_inference.predictive import Predictive

# Posterior predictive
pred = Predictive(model_fn, posterior_samples=samples)
ppc = pred(rng_key)

# Prior predictive
prior_pred = Predictive(model_fn, num_samples=500)
prior_ppc = prior_pred(rng_key)
```

## Functional API

Для тех, кому нужен полный контроль:

```python
from hyperion_inference.hmc import hmc_sample, hmc_sample_chains
from hyperion_inference.nuts import nuts_sample, nuts_sample_chains

# Single-chain
samples, log_probs, accept_probs, info = hmc_sample(
    potential_fn, rng_key, init_position,
    num_samples=2000, num_warmup=500,
    step_size=0.01, num_leapfrog=10,
    dense_mass=False,
)

# Multi-chain (vmap)
samples, log_probs, accept_probs, info = hmc_sample_chains(
    potential_fn, rng_key, init_positions,
    num_chains=4,
    num_samples=2000, num_warmup=500,
)
```

## Engines

### HMC

```python
hmc_sample(
    potential_fn,          # R^n -> R (log-joint)
    rng_key,
    init_position,         # jnp.ndarray shape (dim,)
    num_samples=1000,
    num_warmup=500,
    step_size=0.01,
    num_leapfrog=10,
    target_accept=0.8,
    dense_mass=False,
)
```

### NUTS

```python
nuts_sample(
    potential_fn,
    rng_key,
    init_position,
    num_samples=1000,
    num_warmup=500,
    step_size=0.01,
    max_tree_depth=10,
    target_accept=0.8,
    dense_mass=False,
)
```

### VI

```python
from hyperion_inference.vi import VIEngine

vi = VIEngine()
result = vi.run(backend, rng_key, {
    "num_steps": 5000,
    "learning_rate": 0.01,
    "covariance_type": "diagonal",  # или "low_rank", "full_rank"
    "num_posterior_samples": 2000,
})
```

### SMC

```python
from hyperion_inference.smc import SMCEngine

smc = SMCEngine()
result = smc.run(backend, rng_key, {
    "num_particles": 1000,
    "num_tempering_steps": 20,
    "rejuvenation_steps": 5,
    "adaptive_tempering": True,
})
```
