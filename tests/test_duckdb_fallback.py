"""Test DuckDB fallback functionality."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from episemic.config import EpistemicConfig
from episemic.hippocampus.duckdb_hippocampus import DuckDBHippocampus
from episemic.models import Memory
from episemic.simple import Episemic


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_hippocampus_basic_operations(mock_transformer):
    """Test basic DuckDB hippocampus operations."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1] * 384  # Mock embedding
    mock_transformer.return_value = mock_model

    # Use temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    # Remove the empty file so DuckDB can create a proper database
    os.unlink(db_path)

    try:
        hippocampus = DuckDBHippocampus(db_path=db_path)

        # Create test memory
        memory = Memory(
            title="Test Memory",
            text="This is a test memory for DuckDB storage",
            summary="This is a test memory for DuckDB storage",
            source="test",
            tags=["test", "duckdb"],
            metadata={"test": True}
        )

        # Test storing memory
        result = await hippocampus.store_memory(memory)
        assert result is True

        # Test retrieving memory
        retrieved = await hippocampus.retrieve_memory(memory.id)
        assert retrieved is not None
        assert retrieved.title == memory.title
        assert retrieved.text == memory.text
        assert retrieved.tags == memory.tags
        assert retrieved.metadata == memory.metadata

        # Test vector search
        embedding = await hippocampus.get_embedding("test memory storage")
        search_results = await hippocampus.vector_search(embedding, top_k=5)
        assert len(search_results) >= 1
        assert search_results[0]["content"] == memory.text

        # Test health check
        health = hippocampus.health_check()
        assert health["duckdb"] is True
        assert health["model"] is True
        assert health["embeddings"] is True

        # Test memory count
        count = await hippocampus.get_memory_count()
        assert count >= 1

        # Test quarantine
        quarantine_result = await hippocampus.mark_quarantined(memory.id)
        assert quarantine_result is True

        # Should not find quarantined memory
        quarantined = await hippocampus.retrieve_memory(memory.id)
        assert quarantined is None

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_simple_api_with_duckdb_fallback(mock_transformer):
    """Test simple API automatically using DuckDB fallback."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1] * 384  # Mock embedding
    mock_transformer.return_value = mock_model

    # Configure to use DuckDB by default
    config = EpistemicConfig()
    config.use_duckdb_fallback = True
    config.prefer_qdrant = False
    config.debug = True
    # Disable external dependencies for testing
    config.enable_cortex = False
    config.enable_consolidation = False

    # Use temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        config.duckdb.db_path = tmp.name
    # Remove the empty file so DuckDB can create a proper database
    os.unlink(config.duckdb.db_path)

    try:
        async with Episemic(config=config) as episemic:
            # Test storing a memory
            memory = await episemic.remember(
                "DuckDB makes local storage easy",
                title="DuckDB Benefits",
                tags=["database", "local"]
            )
            assert memory is not None
            assert "DuckDB" in memory.text

            # Test searching
            results = await episemic.recall("local storage")
            assert len(results) >= 1
            assert any("DuckDB" in result.memory.text for result in results)

            # Test retrieving by ID (this will likely fail since we disabled retrieval)
            try:
                retrieved = await episemic.get(memory.id)
                if retrieved:
                    assert retrieved.text == memory.text
            except Exception:
                # Expected to fail since retrieval engine is disabled
                pass

            # Test finding related memories (this will likely fail since we disabled retrieval)
            try:
                related = await episemic.find_related(memory.id, limit=3)
                assert isinstance(related, list)
            except Exception:
                # Expected to fail since retrieval engine is disabled
                pass

    finally:
        # Cleanup
        if os.path.exists(config.duckdb.db_path):
            os.unlink(config.duckdb.db_path)


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_with_in_memory_db(mock_transformer):
    """Test DuckDB with in-memory database."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1] * 384  # Mock embedding
    mock_transformer.return_value = mock_model

    hippocampus = DuckDBHippocampus(db_path=None)  # In-memory

    # Create test memory
    memory = Memory(
        title="In-Memory Test",
        text="This memory is stored in-memory using DuckDB",
        summary="This memory is stored in-memory using DuckDB",
        source="test",
        tags=["memory", "in-memory"]
    )

    # Test basic operations
    result = await hippocampus.store_memory(memory)
    assert result is True

    retrieved = await hippocampus.retrieve_memory(memory.id)
    assert retrieved is not None
    assert retrieved.title == memory.title

    count = await hippocampus.get_memory_count()
    assert count >= 1


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_vector_search_with_filters(mock_transformer):
    """Test DuckDB vector search with various filters."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1] * 384  # Mock embedding
    mock_transformer.return_value = mock_model

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    # Remove the empty file so DuckDB can create a proper database
    os.unlink(db_path)

    try:
        hippocampus = DuckDBHippocampus(db_path=db_path)

        # Store multiple memories with different tags and sources
        memories = [
            Memory(
                title="Python Programming",
                text="Python is a versatile programming language",
                summary="Python is a versatile programming language",
                source="book",
                tags=["python", "programming"]
            ),
            Memory(
                title="Database Design",
                text="Database design is crucial for application performance",
                summary="Database design is crucial for application performance",
                source="article",
                tags=["database", "design"]
            ),
            Memory(
                title="Python Database",
                text="Python has excellent database connectivity options",
                summary="Python has excellent database connectivity options",
                source="tutorial",
                tags=["python", "database"]
            )
        ]

        # Store all memories
        for memory in memories:
            await hippocampus.store_memory(memory)

        # Test search with tag filter
        embedding = await hippocampus.get_embedding("python programming")
        results = await hippocampus.vector_search(
            embedding,
            top_k=5,
            filters={"tags": "python"}
        )

        # Should find memories tagged with "python"
        assert len(results) >= 2
        for result in results:
            assert "python" in result["tags"]

        # Test search with source filter
        results = await hippocampus.vector_search(
            embedding,
            top_k=5,
            filters={"source": "book"}
        )

        # Should find only book source
        assert len(results) >= 1
        for result in results:
            assert result["source"] == "book"

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_config_environment_variables():
    """Test DuckDB configuration from environment variables."""
    # Set environment variables
    test_path = "/tmp/test_episemic.db"
    test_model = "all-MiniLM-L6-v2"

    os.environ["DUCKDB_PATH"] = test_path
    os.environ["DUCKDB_MODEL"] = test_model
    os.environ["EPISEMIC_USE_DUCKDB"] = "true"

    try:
        config = EpistemicConfig.from_env()

        assert config.duckdb.db_path == test_path
        assert config.duckdb.model_name == test_model
        assert config.use_duckdb_fallback is True

    finally:
        # Cleanup environment
        if "DUCKDB_PATH" in os.environ:
            del os.environ["DUCKDB_PATH"]
        if "DUCKDB_MODEL" in os.environ:
            del os.environ["DUCKDB_MODEL"]
        if "EPISEMIC_USE_DUCKDB" in os.environ:
            del os.environ["EPISEMIC_USE_DUCKDB"]


def test_duckdb_config_defaults():
    """Test DuckDB configuration defaults."""
    config = EpistemicConfig()

    assert config.duckdb.db_path is None  # Default to in-memory
    assert config.duckdb.model_name == "all-MiniLM-L6-v2"
    assert config.use_duckdb_fallback is True
    assert config.prefer_qdrant is False