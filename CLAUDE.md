# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Setup
```bash
# Initial development setup
make dev                    # Install with dev dependencies
make pre-commit            # Install pre-commit hooks

# Alternative with Poetry
poetry install --with dev
poetry run pre-commit install
```

### Testing
```bash
make test                  # Run all tests
pytest tests/test_simple_api.py -v  # Run specific test file
pytest tests/ -v --cov=episemic  # Run with coverage
pytest -k "test_memory_creation" -v   # Run specific test by name

# Test with specific storage backend
EPISEMIC_PREFER_QDRANT=false pytest tests/ -v  # DuckDB only
EPISEMIC_PREFER_QDRANT=true pytest tests/ -v   # Qdrant if available
```

### Code Quality
```bash
make check                 # Run all quality checks (lint + type + test)
make lint                  # Run ruff linting
make format                # Format code with ruff
make type-check            # Run mypy type checking
```

### Development Server
```bash
make run                   # Run CLI in development mode
poetry run episemic        # Alternative CLI execution
```

### Version Management
```bash
make version               # Show current version
make bump-patch            # Bump patch version (1.0.2 → 1.0.3)
make release-patch         # Bump version and upload to test PyPI
```

### Building and Packaging
```bash
make package-build         # Build package for distribution
make package-test          # Build and upload to test PyPI
make package-prod          # Build and upload to production PyPI
```

## Architecture Overview

Episemic is a brain-inspired memory system with modular architecture supporting dual storage backends.

### Core Components

**Two-Layer Memory System (Brain-Inspired)**:
- **Hippocampus** (`episemic/hippocampus/`): Fast, episodic memory for recent experiences
- **Cortex** (`episemic/cortex/`): Long-term semantic memory for consolidated knowledge

**Storage Backend Architecture**:
- **DuckDB Backend** (Default): Zero-config local storage with built-in vector search
- **Qdrant + PostgreSQL Backend**: Production-ready with separate vector and relational stores
- **Automatic Fallback**: DuckDB used when Qdrant/PostgreSQL unavailable

**Processing Engines**:
- **Consolidation Engine** (`episemic/consolidation/`): Transfers memories from hippocampus to cortex
- **Retrieval Engine** (`episemic/retrieval/`): Multi-strategy search across both memory stores

### API Layers

**Three-Tier API Design**:
1. **Simple API** (`episemic/simple.py`): User-friendly `Episemic` and `EpistemicSync` classes
2. **Core API** (`episemic/api.py`): `EpistemicAPI` for library integration
3. **CLI Interface** (`episemic/cli/`): Typer-based command-line tools

**Key API Pattern**:
- `simple.py` wraps `api.py` for ease of use
- `api.py` orchestrates core components (hippocampus, cortex, consolidation, retrieval)
- All APIs use the same underlying `models.py` data structures

### Configuration System

**Hierarchical Configuration** (`episemic/config.py`):
- Pydantic-based configuration with defaults
- Environment variable support
- Per-component configs: `QdrantConfig`, `DuckDBConfig`, `PostgreSQLConfig`, etc.
- Storage backend preference controlled by `prefer_qdrant` and `use_duckdb_fallback`

### Storage Backend Selection Logic

The system automatically chooses storage backend:
1. If `prefer_qdrant=True` and Qdrant/PostgreSQL available → Use Qdrant + PostgreSQL
2. If `prefer_qdrant=False` or services unavailable → Use DuckDB fallback
3. DuckDB supports both file-based (`db_path` set) and in-memory (`db_path=None`) modes

## Development Patterns

### Memory Model
All memory operations use consistent `Memory` objects with:
- `id`: Unique identifier
- `text`: Main content
- `title`: Optional title
- `tags`: List of string tags
- `metadata`: Arbitrary JSON metadata
- `created_at`: Timestamp

### Async/Sync Pattern
- Core implementations are async-first
- `EpistemicSync` wrapper provides synchronous interface
- Both APIs expose identical methods: `remember()`, `recall()`, `get()`, `find_related()`

### Testing Strategy
- **Storage Backend Testing**: Tests run against both DuckDB and Qdrant backends
- **API Layer Testing**: Comprehensive tests for simple, core, and CLI APIs
- **Error Path Testing**: Dedicated tests for fallback scenarios and error conditions
- **Integration Testing**: End-to-end workflows across all components

### Configuration Patterns
Environment variables follow naming convention:
- `QDRANT_HOST`, `QDRANT_PORT` for Qdrant config
- `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_PASSWORD` for PostgreSQL
- `DUCKDB_PATH`, `DUCKDB_MODEL` for DuckDB config
- `EPISEMIC_PREFER_QDRANT`, `EPISEMIC_DEBUG` for global settings

## Package Structure

**Core Library** (`episemic/`):
- `__init__.py`: Exports simple API classes
- `simple.py`: User-friendly async/sync interfaces
- `api.py`: Internal API for component orchestration
- `models.py`: Pydantic data models
- `config.py`: Configuration management

**Memory Components**:
- `hippocampus/hippocampus.py`: Qdrant + Redis implementation
- `hippocampus/duckdb_hippocampus.py`: DuckDB fallback implementation
- `cortex/cortex.py`: PostgreSQL long-term storage
- `consolidation/consolidation.py`: Memory transfer logic
- `retrieval/retrieval.py`: Multi-strategy search

**Interface Layer**:
- `cli/main.py`: Typer-based command-line interface

## Testing Notes

- Use `EPISEMIC_PREFER_QDRANT=false` to force DuckDB-only testing
- Use `EPISEMIC_DEBUG=true` for verbose logging during development
- DuckDB tests require no external services and run fastest
- Qdrant/PostgreSQL tests require Docker services but test production paths
- Tests automatically fall back to DuckDB if external services unavailable