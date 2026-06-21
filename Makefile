.PHONY: setup install format lint test clean

PYTHON = python3
PIP = pip

setup:
	$(PYTHON) -m venv .venv
	@echo "Virtual environment created. Run 'source .venv/bin/activate' to activate."

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

format:
	black src tests setup.py

lint:
	flake8 src tests setup.py
	mypy src tests setup.py

test:
	pytest tests/

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
