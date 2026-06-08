.PHONY: test lint format

TEST ?= .
PYTHON_FILES=.

test:
	uv run pytest $(TEST)

lint:
	uv run ruff check .
	uv run ruff format $(PYTHON_FILES) --diff
	uv run ruff check --select I $(PYTHON_FILES)
	uv run ty check $(PYTHON_FILES)

format:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --fix $(PYTHON_FILES)
