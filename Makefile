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

# Package for distribution
package-build:
	./scripts/build.sh

# Upload to test PyPI
package-test:
	./scripts/package.sh test

# Upload to production PyPI
package-prod:
	./scripts/package.sh prod

# Version management with bump2version
bump-patch:
	bump2version patch

bump-minor:
	bump2version minor

bump-major:
	bump2version major

# Show current version
version:
	@echo "Current version: $$(grep 'current_version =' .bumpversion.cfg | cut -d'=' -f2 | xargs)"

# Combined release workflows
release-patch: bump-patch package-test
	@echo "Patch release completed and uploaded to test PyPI"

release-minor: bump-minor package-test
	@echo "Minor release completed and uploaded to test PyPI"

release-major: bump-major package-test
	@echo "Major release completed and uploaded to test PyPI"

# Publish to PyPI (requires authentication)
publish:
	poetry publish