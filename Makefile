.PHONY: setup test lint format all

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -e .
	@echo "Setup complete. Activate with: source .venv/bin/activate"

test:
	. .venv/bin/activate && pytest -v

lint:
	. .venv/bin/activate && ruff check src tests

format:
	. .venv/bin/activate && black src tests
	. .venv/bin/activate && ruff check --fix src tests

all: setup test