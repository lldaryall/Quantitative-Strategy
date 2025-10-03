.PHONY: help install install-dev test test-cov lint format type-check security clean build docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=qbacktester --cov-report=html --cov-report=term-missing

lint: ## Run linting
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format: ## Format code
	black src/ tests/
	isort src/ tests/

type-check: ## Run type checking
	mypy src/qbacktester

security: ## Run security checks
	bandit -r src/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build package
	python -m build

docs: ## Generate documentation
	@echo "Documentation generation not yet implemented"

pre-commit: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

docker-build: ## Build Docker image
	docker build -t qbacktester .

docker-run: ## Run Docker container
	docker run --rm -v $(PWD)/reports:/app/reports qbacktester

ci: ## Run CI checks locally
	make lint
	make type-check
	make security
	make test-cov

all: clean install-dev pre-commit ci ## Run all checks and install dev dependencies

