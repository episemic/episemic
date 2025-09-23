"""Comprehensive tests for DuckDB hippocampus implementation."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from episemic.config import EpistemicConfig, DuckDBConfig
from episemic.hippocampus.duckdb_hippocampus import DuckDBHippocampus
from episemic.models import Memory


@pytest.mark.asyncio
@patch("episemic.hippocampus.duckdb_hippocampus.SentenceTransformer")
async def test_duckdb_initialization_and_health(mock_transformer):
    """Test DuckDB initialization and health check."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1] * 384
    mock_transformer.return_value = mock_model

    # Test in-memory initialization
    hippocampus = DuckDBHippocampus(db_path=None)

    # Test health check before initialization
    health = hippocampus.health_check()
    assert health["duckdb"] is False
    assert health["model"] is False
    assert health["embeddings"] is False

    # Initialize and test health
    await hippocampus._ensure_initialized()
    health = hippocampus.health_check()
    assert health["duckdb"] is True
    assert health["model"] is True
    assert health["embeddings"] is True

    # Test model encoding
    embedding = await hippocampus.get_embedding("test text")
    assert len(embedding) == 384
    assert all(x == 0.1 for x in embedding)


@pytest.mark.asyncio
@patch("episemic.hippocampus.duckdb_hippocampus.SentenceTransformer")
async def test_duckdb_memory_operations(mock_transformer):
    """Test comprehensive memory operations."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.2] * 384
    mock_transformer.return_value = mock_model

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    os.unlink(db_path)

    try:
        hippocampus = DuckDBHippocampus(db_path=db_path)

        # Test memory count when empty
        count = await hippocampus.get_memory_count()
        assert count == 0

        # Create and store memory
        memory = Memory(
            title="Test Memory",
            text="This is a comprehensive test memory",
            summary="This is a comprehensive test memory",
            source="test",
            tags=["test", "comprehensive"],
            metadata={"importance": "high", "category": "testing"},
        )

        # Store memory
        result = await hippocampus.store_memory(memory)
        assert result is True

        # Test memory count after storing
        count = await hippocampus.get_memory_count()
        assert count == 1

        # Retrieve memory
        retrieved = await hippocampus.retrieve_memory(memory.id)
        assert retrieved is not None
        assert retrieved.title == memory.title
        assert retrieved.text == memory.text
        assert retrieved.tags == memory.tags
        assert retrieved.metadata == memory.metadata

        # Test non-existent memory
        non_existent = await hippocampus.retrieve_memory("non-existent-id")
        assert non_existent is None

        # Test integrity verification
        integrity = await hippocampus.verify_integrity(memory.id)
        assert integrity is True

        integrity_bad = await hippocampus.verify_integrity("non-existent-id")
        assert integrity_bad is False

        # Test quarantine
        quarantine_result = await hippocampus.mark_quarantined(memory.id)
        assert quarantine_result is True

        # Should not retrieve quarantined memory
        quarantined = await hippocampus.retrieve_memory(memory.id)
        assert quarantined is None

        # Memory count should be 0 after quarantine
        count = await hippocampus.get_memory_count()
        assert count == 0

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
@patch("episemic.hippocampus.duckdb_hippocampus.SentenceTransformer")
async def test_duckdb_vector_search_comprehensive(mock_transformer):
    """Test comprehensive vector search functionality."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.3] * 384
    mock_transformer.return_value = mock_model

    hippocampus = DuckDBHippocampus(db_path=None)  # In-memory

    # Store multiple memories
    memories = []
    for i in range(5):
        memory = Memory(
            title=f"Memory {i}",
            text=f"Content for memory number {i}",
            summary=f"Content for memory number {i}",
            source=f"source_{i % 2}",  # Alternate sources
            tags=[f"tag_{i}", "common"] if i % 2 == 0 else [f"tag_{i}"],
            metadata={"index": i, "even": i % 2 == 0},
        )
        memories.append(memory)
        await hippocampus.store_memory(memory)

    # Test basic vector search
    embedding = await hippocampus.get_embedding("test query")
    results = await hippocampus.vector_search(embedding, top_k=3)
    assert len(results) == 3

    # Test search with tag filter
    results = await hippocampus.vector_search(embedding, top_k=10, filters={"tags": "common"})
    # Should find memories with "common" tag (even indices: 0, 2, 4)
    assert len(results) == 3
    for result in results:
        assert "common" in result["tags"]

    # Test search with source filter
    results = await hippocampus.vector_search(embedding, top_k=10, filters={"source": "source_0"})
    # Should find memories with source_0 (even indices: 0, 2, 4)
    assert len(results) == 3
    for result in results:
        assert result["source"] == "source_0"

    # Test search with no filters
    results = await hippocampus.vector_search(embedding, top_k=10)
    assert len(results) == 5


@pytest.mark.asyncio
@patch("episemic.hippocampus.duckdb_hippocampus.SentenceTransformer")
async def test_duckdb_error_handling(mock_transformer):
    """Test error handling in DuckDB operations."""
    # Mock the sentence transformer to raise an error
    mock_transformer.side_effect = Exception("Model loading failed")

    hippocampus = DuckDBHippocampus(db_path=None)

    # Test store memory with model error
    memory = Memory(
        title="Error Test", text="This should fail", summary="This should fail", source="test"
    )

    # This should fall back gracefully and still work (without embeddings)
    result = await hippocampus.store_memory(memory)
    assert result is True  # Should succeed with fallback

    # Reset transformer mock for successful initialization
    mock_transformer.side_effect = None
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.4] * 384
    mock_transformer.return_value = mock_model

    # Create new instance for successful operations
    hippocampus2 = DuckDBHippocampus(db_path=None)

    # Test successful operations
    result = await hippocampus2.store_memory(memory)
    assert result is True

    # Test error in vector search
    with patch.object(hippocampus2, "conn") as mock_conn:
        mock_conn.execute.side_effect = Exception("Database error")

        embedding = await hippocampus2.get_embedding("test")
        results = await hippocampus2.vector_search(embedding)
        assert results == []


@pytest.mark.asyncio
@patch("episemic.hippocampus.duckdb_hippocampus.SentenceTransformer")
async def test_duckdb_close_operation(mock_transformer):
    """Test DuckDB close operation."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.5] * 384
    mock_transformer.return_value = mock_model

    hippocampus = DuckDBHippocampus(db_path=None)
    await hippocampus._ensure_initialized()

    # Test close operation
    hippocampus.close()

    # After close, should not have conn attribute
    assert not hasattr(hippocampus, "conn") or hippocampus.conn is None


def test_duckdb_config_integration():
    """Test DuckDB configuration integration."""
    # Test DuckDB config creation
    config = DuckDBConfig()
    assert config.db_path is None
    assert config.model_name == "all-MiniLM-L6-v2"

    # Test custom config
    config = DuckDBConfig(db_path="/tmp/test.db", model_name="custom-model")
    assert config.db_path == "/tmp/test.db"
    assert config.model_name == "custom-model"

    # Test in EpistemicConfig
    episemic_config = EpistemicConfig()
    assert episemic_config.duckdb.db_path is None
    assert episemic_config.use_duckdb_fallback is True
    assert episemic_config.prefer_qdrant is False


@pytest.mark.asyncio
@patch("episemic.hippocampus.duckdb_hippocampus.SentenceTransformer")
async def test_duckdb_file_persistence(mock_transformer):
    """Test file persistence across sessions."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.6] * 384
    mock_transformer.return_value = mock_model

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    os.unlink(db_path)

    try:
        # First session: store memory
        hippocampus1 = DuckDBHippocampus(db_path=db_path)

        memory = Memory(
            title="Persistent Memory",
            text="This memory should persist across sessions",
            summary="This memory should persist across sessions",
            source="persistence_test",
            tags=["persistent"],
        )

        result = await hippocampus1.store_memory(memory)
        assert result is True

        count1 = await hippocampus1.get_memory_count()
        assert count1 == 1

        hippocampus1.close()

        # Second session: retrieve memory
        hippocampus2 = DuckDBHippocampus(db_path=db_path)

        count2 = await hippocampus2.get_memory_count()
        assert count2 == 1

        retrieved = await hippocampus2.retrieve_memory(memory.id)
        assert retrieved is not None
        assert retrieved.title == memory.title
        assert retrieved.text == memory.text

        hippocampus2.close()

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
@patch("episemic.hippocampus.duckdb_hippocampus.SentenceTransformer")
async def test_duckdb_multiple_memories_search(mock_transformer):
    """Test search with multiple memories and various scenarios."""
    # Mock the sentence transformer with different embeddings
    mock_model = MagicMock()

    def mock_encode(text):
        # Different embeddings based on content
        if "python" in text.lower():
            return MagicMock(tolist=lambda: [0.8] * 384)
        elif "javascript" in text.lower():
            return MagicMock(tolist=lambda: [0.6] * 384)
        else:
            return MagicMock(tolist=lambda: [0.4] * 384)

    mock_model.encode.side_effect = mock_encode
    mock_transformer.return_value = mock_model

    hippocampus = DuckDBHippocampus(db_path=None)

    # Store diverse memories
    memories_data = [
        (
            "Python Programming",
            "Python is a versatile programming language",
            ["python", "programming"],
        ),
        ("JavaScript Basics", "JavaScript runs in browsers and servers", ["javascript", "web"]),
        ("Database Design", "Database design is crucial for applications", ["database", "design"]),
        ("Machine Learning", "ML algorithms learn from data", ["ml", "ai"]),
        ("Web Development", "Building web applications with modern tools", ["web", "development"]),
    ]

    for title, text, tags in memories_data:
        memory = Memory(title=title, text=text, summary=text, source="knowledge_base", tags=tags)
        await hippocampus.store_memory(memory)

    # Test search for Python-related content
    python_embedding = await hippocampus.get_embedding("python programming")
    results = await hippocampus.vector_search(python_embedding, top_k=3)
    assert len(results) <= 3

    # Test search with limit
    all_results = await hippocampus.vector_search(python_embedding, top_k=10)
    assert len(all_results) == 5

    # Test empty search
    empty_results = await hippocampus.vector_search(python_embedding, top_k=0)
    assert len(empty_results) == 0

    # Test with no matches for specific tag
    no_match_results = await hippocampus.vector_search(
        python_embedding, filters={"tags": "nonexistent"}
    )
    assert len(no_match_results) == 0
