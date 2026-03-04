# Contributing to HYPERION

Рады контрибьюторам. Ниже — правила, без которых PR не примем.

## Как начать

```bash
git clone https://github.com/BusyaPrime/Hyperion.git
cd Hyperion/python
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Структура веток

- `main` — стабильная ветка, CI зелёный
- `feature/<name>` — фичи
- `fix/<name>` — баги
- `refactor/<name>` — рефакторинг без изменения поведения

## Code style

- **Python 3.10+**, type hints обязательны
- **mypy strict** — `disallow_untyped_defs`, `no_implicit_optional`
- **ruff** — линтинг, `line-length = 100`
- **black** — форматирование

Перед коммитом:

```bash
make lint      # ruff + mypy
make format    # black + ruff --fix
make test      # pytest
```

## Тесты

Каждый PR должен содержать тесты. Без тестов — без мержа.

- Unit-тесты: `tests/test_<module>/`
- Integration: `tests/test_integration/`
- Posterior recovery: `tests/test_inference/test_posterior_recovery.py`

```bash
make test           # все тесты
make test-fast      # без integration (для быстрой итерации)
```

## Commit messages

Формат: `<type>: <description>`

Типы:
- `feat:` — новая функциональность
- `fix:` — исправление бага
- `refactor:` — рефакторинг без изменения поведения
- `test:` — тесты
- `docs:` — документация
- `build:` — зависимости, CI, сборка
- `chore:` — прочее (gitignore, configs)
- `bench:` — бенчмарки
- `ci:` — CI/CD

## Pull Request

1. Fork → branch → code → tests → PR
2. Описание: что сделано и зачем
3. CI должен быть зелёный
4. Один PR = одна логическая единица

## Распределения

Добавляешь новое распределение — обязательно:

1. Наследуйся от `Distribution`
2. Реализуй `log_prob`, `sample`, `support`
3. Добавь в `hyperion_dsl/__init__.py` → `__all__`
4. Тесты: `test_distributions/`, включая edge cases (zero scale, inf values)
5. Трансформ в `transforms.py` если support != Real

## Inference engine

Новый движок — обязательно:

1. Наследуйся от `InferenceEngine`
2. Реализуй `initialize`, `step`, `get_samples`, `run`
3. Posterior recovery тест: Normal-Normal, проверь mean и variance
4. Logging через `logging.getLogger(__name__)`
