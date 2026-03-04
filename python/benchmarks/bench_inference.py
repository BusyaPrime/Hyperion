"""Бенчмарки inference engines — сколько сэмплов/сек, как масштабируется с dim.

Запуск:
    python benchmarks/bench_inference.py

Выводит таблицу: engine x dim → samples/sec, total time.
"""

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from hyperion_dsl.primitives import sample
from hyperion_dsl.distributions import Normal, HalfNormal
from hyperion_dsl.model import model
from hyperion_backends.jax_backend import JAXBackend
from hyperion_inference.hmc import hmc_sample
from hyperion_inference.nuts import nuts_sample


@dataclass
class BenchResult:
    engine: str
    dim: int
    num_samples: int
    wall_time: float
    samples_per_sec: float


def make_normal_model(dim: int):
    """Генерирует Normal-Normal модель заданной размерности."""
    @model
    def bench_model():
        for i in range(dim):
            mu = sample(f"mu_{i}", Normal(0.0, 5.0))
            sample(f"obs_{i}", Normal(mu, 1.0))
    return bench_model


def bench_hmc(dim: int, num_samples: int = 1000, num_warmup: int = 500) -> BenchResult:
    key = jrandom.PRNGKey(42)
    mdl = make_normal_model(dim)

    data_key, init_key, sample_key = jrandom.split(key, 3)
    obs = {f"obs_{i}": jrandom.normal(data_key, shape=(50,)) for i in range(dim)}

    backend = JAXBackend()
    backend.initialize(mdl, obs, init_key)
    init_pos = backend.flatten_latents(backend.sample_prior(init_key))

    # warmup JIT
    _ = hmc_sample(
        backend.potential_fn, sample_key, init_pos,
        num_samples=10, num_warmup=10, num_leapfrog=10,
    )

    start = time.perf_counter()
    _ = hmc_sample(
        backend.potential_fn, sample_key, init_pos,
        num_samples=num_samples, num_warmup=num_warmup, num_leapfrog=10,
    )
    jax.block_until_ready(_[0])
    elapsed = time.perf_counter() - start

    return BenchResult(
        engine="HMC",
        dim=dim,
        num_samples=num_samples,
        wall_time=elapsed,
        samples_per_sec=num_samples / elapsed,
    )


def bench_nuts(dim: int, num_samples: int = 1000, num_warmup: int = 500) -> BenchResult:
    key = jrandom.PRNGKey(42)
    mdl = make_normal_model(dim)

    data_key, init_key, sample_key = jrandom.split(key, 3)
    obs = {f"obs_{i}": jrandom.normal(data_key, shape=(50,)) for i in range(dim)}

    backend = JAXBackend()
    backend.initialize(mdl, obs, init_key)
    init_pos = backend.flatten_latents(backend.sample_prior(init_key))

    # warmup JIT
    _ = nuts_sample(
        backend.potential_fn, sample_key, init_pos,
        num_samples=10, num_warmup=10, max_tree_depth=5,
    )

    start = time.perf_counter()
    _ = nuts_sample(
        backend.potential_fn, sample_key, init_pos,
        num_samples=num_samples, num_warmup=num_warmup, max_tree_depth=10,
    )
    jax.block_until_ready(_[0])
    elapsed = time.perf_counter() - start

    return BenchResult(
        engine="NUTS",
        dim=dim,
        num_samples=num_samples,
        wall_time=elapsed,
        samples_per_sec=num_samples / elapsed,
    )


def main() -> None:
    dims = [1, 5, 10, 20]
    num_samples = 1000

    print("=" * 72)
    print(f"{'HYPERION Inference Benchmarks':^72}")
    print("=" * 72)
    print(f"{'Engine':<8} {'Dim':>4} {'Samples':>8} {'Time (s)':>10} {'Samples/sec':>12}")
    print("-" * 72)

    for dim in dims:
        for bench_fn in [bench_hmc, bench_nuts]:
            result = bench_fn(dim, num_samples=num_samples)
            print(
                f"{result.engine:<8} {result.dim:>4} {result.num_samples:>8} "
                f"{result.wall_time:>10.3f} {result.samples_per_sec:>12.0f}"
            )

    print("-" * 72)
    print(f"Platform: JAX {jax.__version__}, {jax.devices()[0].platform.upper()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
