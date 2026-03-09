.DEFAULT_GOAL := help
PYTHON := python
PIP := pip
PYTEST := pytest
PROJECT_NAME := pytorch-mastery-hub
SRC_DIR := src
TESTS_DIR := tests
DOCS_DIR := docs
NOTEBOOKS_DIR := notebooks
VENV_DIR := venv

.PHONY: help install install-dev install-all clean clean-pyc clean-build clean-test \
	test test-unit test-integration test-coverage lint format type-check security-check \
	docs docs-serve notebooks notebook-clean pre-commit docker docker-gpu tox \
	setup-dev check-all

## ─────────────────────────────────────────────────────────
##  Help
## ─────────────────────────────────────────────────────────

help:  ## Show this help message
	@echo ""
	@echo "PyTorch Mastery Hub - Development Commands"
	@echo "─────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""

## ─────────────────────────────────────────────────────────
##  Installation
## ─────────────────────────────────────────────────────────

venv:  ## Create a virtual environment in ./venv
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created. Activate with: source $(VENV_DIR)/bin/activate"

install:  ## Install core dependencies
	$(PIP) install -e .

install-dev:  ## Install development dependencies
	$(PIP) install -e ".[dev,notebooks]"
	pre-commit install
	@echo "Development environment ready!"

install-all:  ## Install all optional dependencies
	$(PIP) install -e ".[all]"
	pre-commit install

setup-dev: venv install-dev  ## Full dev setup from scratch (creates venv + installs deps)

## ─────────────────────────────────────────────────────────
##  Testing
## ─────────────────────────────────────────────────────────

test:  ## Run all tests
	$(PYTEST) $(TESTS_DIR) -v

test-unit:  ## Run only unit tests
	$(PYTEST) $(TESTS_DIR) -v -m "unit"

test-integration:  ## Run only integration tests
	$(PYTEST) $(TESTS_DIR) -v -m "integration"

test-fast:  ## Run all tests except slow ones
	$(PYTEST) $(TESTS_DIR) -v -m "not slow"

test-gpu:  ## Run GPU-specific tests (requires CUDA)
	$(PYTEST) $(TESTS_DIR) -v -m "gpu"

test-coverage:  ## Run tests with coverage report
	$(PYTEST) $(TESTS_DIR) -v \
		--cov=$(SRC_DIR) \
		--cov-report=html \
		--cov-report=xml \
		--cov-report=term-missing \
		--cov-fail-under=70
	@echo "HTML coverage report generated in htmlcov/"

test-parallel:  ## Run tests in parallel (requires pytest-xdist)
	$(PYTEST) $(TESTS_DIR) -v -n auto

## ─────────────────────────────────────────────────────────
##  Code Quality
## ─────────────────────────────────────────────────────────

lint:  ## Lint source code with flake8
	flake8 $(SRC_DIR) $(TESTS_DIR)

format:  ## Format code with black and isort
	black $(SRC_DIR) $(TESTS_DIR)
	isort $(SRC_DIR) $(TESTS_DIR)

format-check:  ## Check code formatting without modifying files
	black --check $(SRC_DIR) $(TESTS_DIR)
	isort --check-only $(SRC_DIR) $(TESTS_DIR)

type-check:  ## Run mypy type checking
	mypy $(SRC_DIR) --ignore-missing-imports

security-check:  ## Run bandit security analysis
	bandit -r $(SRC_DIR) -x $(TESTS_DIR)

pre-commit:  ## Run all pre-commit hooks
	pre-commit run --all-files

pre-commit-update:  ## Update pre-commit hooks
	pre-commit autoupdate

check-all: format-check lint type-check security-check test  ## Run all checks (format, lint, type, security, tests)

## ─────────────────────────────────────────────────────────
##  Documentation
## ─────────────────────────────────────────────────────────

docs:  ## Build Sphinx documentation
	cd $(DOCS_DIR) && make html
	@echo "Documentation built in $(DOCS_DIR)/_build/html/"

docs-serve:  ## Build and serve documentation locally
	cd $(DOCS_DIR) && make html
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8080

docs-clean:  ## Clean generated documentation
	cd $(DOCS_DIR) && make clean

docs-api:  ## Generate API documentation from docstrings
	sphinx-apidoc -o $(DOCS_DIR)/api $(SRC_DIR) --force --module-first

docs-all: docs-api docs  ## Generate API docs and build HTML docs

## ─────────────────────────────────────────────────────────
##  Notebooks
## ─────────────────────────────────────────────────────────

notebooks:  ## Start Jupyter Lab
	jupyter lab --notebook-dir=$(NOTEBOOKS_DIR)

notebook-clean:  ## Strip output from all notebooks
	nbstripout $(NOTEBOOKS_DIR)/**/*.ipynb

notebook-test:  ## Execute a sample notebook for testing
	jupyter nbconvert --to notebook --execute \
		$(NOTEBOOKS_DIR)/01_fundamentals/01_introduction_to_tensors.ipynb \
		--output executed_test.ipynb
	rm -f executed_test.ipynb
	@echo "Notebook execution test passed!"

notebook-html:  ## Convert all notebooks to HTML
	@mkdir -p docs/notebooks
	jupyter nbconvert --to html $(NOTEBOOKS_DIR)/**/*.ipynb \
		--output-dir docs/notebooks/

## ─────────────────────────────────────────────────────────
##  Docker
## ─────────────────────────────────────────────────────────

docker-build:  ## Build CPU Docker image
	docker build -t $(PROJECT_NAME):cpu .

docker-build-gpu:  ## Build GPU Docker image
	docker build -f Dockerfile.gpu -t $(PROJECT_NAME):gpu .

docker-run:  ## Run CPU container with Jupyter
	docker-compose --profile cpu up -d
	@echo "Jupyter Lab running at http://localhost:8888"

docker-run-gpu:  ## Run GPU container with Jupyter
	docker-compose --profile gpu up -d
	@echo "Jupyter Lab running at http://localhost:8888"

docker-stop:  ## Stop all running containers
	docker-compose down

docker-clean:  ## Remove containers and images
	docker-compose down --rmi all --volumes

## ─────────────────────────────────────────────────────────
##  Multi-environment Testing
## ─────────────────────────────────────────────────────────

tox:  ## Run tox for multi-environment testing
	tox

tox-lint:  ## Run tox lint environment
	tox -e lint

tox-docs:  ## Run tox docs build
	tox -e docs

## ─────────────────────────────────────────────────────────
##  Package Building & Distribution
## ─────────────────────────────────────────────────────────

build:  ## Build distributable package
	$(PYTHON) -m build
	twine check dist/*

clean-build:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -name '*.egg-info' -exec rm -rf {} +

publish-test:  ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI (use with caution!)
	twine upload dist/*

## ─────────────────────────────────────────────────────────
##  Cleaning
## ─────────────────────────────────────────────────────────

clean-pyc:  ## Remove Python compiled files
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pycache' -type d -exec rm -rf {} +

clean-test:  ## Remove test artifacts
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -f *.xml
	find . -name '.pytest_cache' -type d -exec rm -rf {} +

clean-mypy:  ## Remove mypy cache
	rm -rf .mypy_cache/

clean-tox:  ## Remove tox environments
	rm -rf .tox/

clean: clean-pyc clean-test clean-build clean-mypy  ## Clean all temporary files

clean-all: clean clean-tox  ## Clean everything including tox

## ─────────────────────────────────────────────────────────
##  Utilities
## ─────────────────────────────────────────────────────────

download-data:  ## Download sample datasets
	$(PYTHON) scripts/download_datasets.py

check-env:  ## Verify environment is set up correctly
	$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
	$(PYTHON) -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
	$(PYTHON) -c "import numpy; print(f'NumPy: {numpy.__version__}')"

gpu-info:  ## Print GPU information
	$(PYTHON) -c "import torch; print(torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'No GPU available')"

version:  ## Show current version
	$(PYTHON) -c "import src; print(getattr(src, '__version__', '1.0.0'))"

requirements-freeze:  ## Freeze current environment to requirements.txt
	$(PIP) freeze > requirements-frozen.txt
	@echo "Frozen requirements saved to requirements-frozen.txt"
