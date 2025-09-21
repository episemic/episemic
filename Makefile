.PHONY: install test lint format type-check clean dev docs

# Install dependencies
install:
	poetry install

# Install with dev dependencies
dev:
	poetry install --with dev

# Run tests
test:
	poetry run pytest tests/ -v

# Run linting
lint:
	poetry run ruff check episemic_core/ tests/

# Format code
format:
	poetry run ruff format episemic_core/ tests/

# Type checking
type-check:
	poetry run mypy episemic_core/

# Run all checks
check: lint type-check test

# Clean up
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -name "*.pyc" -delete

# Install pre-commit hooks
pre-commit:
	poetry run pre-commit install

# Run the CLI in development mode
run:
	poetry run episemic

# Build the package
build:
	poetry build

# Generate API documentation
docs:
	python generate_docs.py

# View documentation
docs-view:
	@echo "Opening documentation in browser..."
	@python -c "import webbrowser; webbrowser.open('file://$(PWD)/docs/index.html')"

# Publish to PyPI (requires authentication)
publish:
	poetry publish