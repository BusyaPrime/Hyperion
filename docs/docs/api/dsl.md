# DSL API

## `@model`

Декоратор, оборачивающий функцию в `HyperionModel`.

```python
from hyperion_dsl import model

@model
def my_model():
    ...
```

## `sample(name, dist, obs=None)`

Объявляет случайную величину.

- `name` — уникальное имя сайта
- `dist` — распределение (Normal, Gamma, ...)
- `obs` — наблюдённые данные (если есть, фиксирует значение)

```python
mu = sample("mu", Normal(0.0, 1.0))          # латентная
sample("obs", Normal(mu, 1.0), obs=data)      # наблюдаемая
```

## `plate(name, size)`

Контекстный менеджер для батчирования.

```python
with plate("data", N):
    sample("x", Normal(0.0, 1.0))  # N независимых сэмплов
```

## `factor(name, log_weight)`

Добавляет произвольный лог-вес в joint probability.

```python
factor("custom_prior", -0.5 * jnp.sum(x ** 2))
```

## `deterministic(name, value)`

Записывает детерминированное значение в trace.

```python
predicted = deterministic("y_pred", alpha + beta * x)
```
