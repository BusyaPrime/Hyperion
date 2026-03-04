.PHONY: help install test test-fast test-cov lint typecheck format clean bench demo proto docker-build docker-up docker-down java-build java-test docs

help:  ## Показать эту справку — что вообще можно наваять
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Поставить Python зависимости
	cd python && pip install -e ".[dev]"

test:  ## Прогнать все Python тесты
	cd python && python -m pytest tests/ -v --tb=short

test-fast:  ## Только быстрые тесты (без интеграционных)
	cd python && python -m pytest tests/ -v --tb=short -k "not integration"

test-cov:  ## Тесты с coverage отчётом
	cd python && python -m pytest tests/ -v --tb=short --cov=hyperion_dsl --cov=hyperion_trace --cov=hyperion_ir --cov=hyperion_backends --cov=hyperion_inference --cov=hyperion_diagnostics --cov-report=term-missing --cov-report=html

lint:  ## Прогнать линтеры (ruff)
	cd python && ruff check .

typecheck:  ## Запустить mypy strict type checking
	cd python && mypy hyperion_dsl hyperion_trace hyperion_graph hyperion_ir hyperion_backends hyperion_inference hyperion_diagnostics --ignore-missing-imports

format:  ## Отформатировать Python код (black + ruff)
	cd python && black .
	cd python && ruff check --fix .

bench:  ## Запустить бенчмарки inference engines
	cd python && python benchmarks/bench_inference.py

clean:  ## Почистить артефакты сборки и кэши
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf python/*.egg-info
	rm -rf python/htmlcov python/.coverage

demo:  ## Запустить демо-модель
	python scripts/run_demo.py

docs:  ## Собрать MkDocs документацию
	cd docs && mkdocs build

docs-serve:  ## Поднять локальный сервер документации
	cd docs && mkdocs serve

proto:  ## Скомпилировать protobuf определения
	python -m grpc_tools.protoc \
		-I proto \
		--python_out=python/hyperion_api \
		--grpc_python_out=python/hyperion_api \
		--pyi_out=python/hyperion_api \
		proto/hyperion.proto

docker-build:  ## Собрать Docker образы
	docker-compose -f docker/docker-compose.yml build

docker-up:  ## Поднять сервисы через Docker
	docker-compose -f docker/docker-compose.yml up -d

docker-down:  ## Остановить Docker сервисы
	docker-compose -f docker/docker-compose.yml down

java-build:  ## Собрать Java сервисы
	cd java && ./gradlew build -x test

java-test:  ## Прогнать Java тесты
	cd java && ./gradlew test

all: lint typecheck test  ## lint + typecheck + test — полная проверка
