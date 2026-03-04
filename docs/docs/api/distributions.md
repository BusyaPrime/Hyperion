# Распределения

Все распределения наследуют `Distribution` и реализуют:

- `log_prob(value)` — лог-плотность
- `sample(key, sample_shape)` — семплирование через JAX PRNG
- `support` — constraint (Positive, Simplex, ...)
- `default_transform` — биекция в unconstrained пространство

## Непрерывные

| Класс | Параметры | Support |
|-------|-----------|---------|
| `Normal(loc, scale)` | loc, scale > 0 | R |
| `HalfNormal(scale)` | scale > 0 | R+ |
| `Cauchy(loc, scale)` | loc, scale > 0 | R |
| `HalfCauchy(scale)` | scale > 0 | R+ |
| `Laplace(loc, scale)` | loc, scale > 0 | R |
| `StudentT(df, loc, scale)` | df > 0 | R |
| `LogNormal(loc, scale)` | loc, scale > 0 | R+ |

## Положительные

| Класс | Параметры | Support |
|-------|-----------|---------|
| `Gamma(concentration, rate)` | > 0 | R+ |
| `Exponential(rate)` | rate > 0 | R+ |
| `InverseGamma(concentration, scale)` | > 0 | R+ |

## Ограниченные

| Класс | Параметры | Support |
|-------|-----------|---------|
| `Beta(a, b)` | a, b > 0 | (0, 1) |
| `Uniform(low, high)` | low < high | (low, high) |

## Многомерные

| Класс | Параметры | Support |
|-------|-----------|---------|
| `MultivariateNormal(loc, cov)` | loc: R^d, cov: PD | R^d |
| `Dirichlet(concentration)` | > 0, K-dim | Simplex |
| `Wishart(df, scale)` | df > d-1 | PD |
| `LKJCholesky(dim, concentration)` | > 0 | CorrCholesky |
| `GaussianProcess(kernel, X, ...)` | kernel fn | R^n |

## Дискретные

| Класс | Параметры | Support |
|-------|-----------|---------|
| `Bernoulli(probs/logits)` | p in [0,1] | {0, 1} |
| `Categorical(probs/logits)` | simplex | {0, ..., K-1} |
| `Poisson(rate)` | rate > 0 | N |
| `Binomial(total_count, probs)` | n >= 0 | {0, ..., n} |
| `Multinomial(total_count, probs)` | n >= 0 | simplex-integer |

## Специальные

| Класс | Параметры | Support |
|-------|-----------|---------|
| `Delta(value)` | любое | точка |
| `Horseshoe(scale)` | scale > 0 | R |
