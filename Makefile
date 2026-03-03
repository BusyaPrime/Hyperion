.PHONY: help install test lint format clean docker-build docker-up demo proto

help:  ## Показать эту справку — что вообще можно наваять
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Поставить Python зависимости
	cd python && pip install -e ".[dev]"

test:  ## Прогнать все Python тесты
	cd python && python -m pytest tests/ -v --tb=short

test-fast:  ## Только быстрые тесты (без интеграционных)
	cd python && python -m pytest tests/ -v --tb=short -k "not integration"

lint:  ## Прогнать линтеры
	cd python && ruff check .
	cd python && mypy hyperion_dsl hyperion_trace hyperion_graph hyperion_ir --ignore-missing-imports

format:  ## Отформатировать Python код
	cd python && black .
	cd python && ruff check --fix .

clean:  ## Почистить артефакты сборки
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf python/*.egg-info

demo:  ## Запустить демо-модель
	python scripts/run_demo.py

proto:  ## Скомпилировать protobuf определения
	python -m grpc_tools.protoc \
		-I proto \
		--python_out=python/hyperion_api \
		--grpc_python_out=python/hyperion_api \
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
