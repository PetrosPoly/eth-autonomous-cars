POETRY ?= poetry

.PHONY: help install format lint test check clean

help:
	@echo "Available targets:"
	@echo "  install  - install dependencies with Poetry"
	@echo "  format   - format source and tests with ruff"
	@echo "  lint     - run ruff lint checks"
	@echo "  test     - run pytest"
	@echo "  check    - run lint + format-check + tests"
	@echo "  clean    - remove local caches"

install:
	$(POETRY) install

format:
	$(POETRY) run ruff format src tests

lint:
	$(POETRY) run ruff check src tests

test:
	$(POETRY) run pytest

check:
	$(POETRY) run ruff check src tests
	$(POETRY) run ruff format --check src tests
	$(POETRY) run pytest

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov
