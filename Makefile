.PHONY: help setup install format lint test pipeline pipeline-status build-image run-container stop-container clean docker-clean

PYTHON_SYS = python3
VENV = .venv
BIN = $(VENV)/bin
PYTHON = $(BIN)/python
PIP = $(BIN)/pip

IMAGE_NAME = customer-churn-api
IMAGE_TAG = latest

help:
	@echo "======================================================================"
	@echo "                    Customer Churn Prediction CLI                     "
	@echo "======================================================================"
	@echo "Available targets:"
	@echo "  setup            - Create Python virtual environment"
	@echo "  install          - Install dependencies and local package in dev mode"
	@echo "  format           - Format codebase using black"
	@echo "  lint             - Audit codebase using flake8 and mypy"
	@echo "  test             - Run all unit and integration tests with pytest"
	@echo "  pipeline         - Execute DVC pipeline (dvc repro)"
	@echo "  pipeline-status  - View DVC pipeline status"
	@echo "  build-image      - Build multi-stage Docker serving image"
	@echo "  run-container    - Run Docker serving image locally"
	@echo "  stop-container   - Stop and remove local running container"
	@echo "  clean            - Remove Python build files and cache directories"
	@echo "  docker-clean     - Remove Docker containers and built images"
	@echo "======================================================================"

setup:
	$(PYTHON_SYS) -m venv $(VENV)
	@echo "Virtual environment created. Run 'source $(VENV)/bin/activate' to activate."

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .[train,dev]


format:
	$(BIN)/black src tests setup.py

lint:
	$(BIN)/flake8 src tests setup.py
	$(BIN)/mypy src tests setup.py

test:
	PYTHONPATH=. $(BIN)/pytest

pipeline:
	dvc repro

pipeline-status:
	dvc status

build-image:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

run-container: stop-container
	docker run -d --name $(IMAGE_NAME) -p 8000:8000 $(IMAGE_NAME):$(IMAGE_TAG)

stop-container:
	docker stop $(IMAGE_NAME) 2>/dev/null || true
	docker rm $(IMAGE_NAME) 2>/dev/null || true

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete

docker-clean: stop-container
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
