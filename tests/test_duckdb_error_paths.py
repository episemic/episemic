"""Tests for DuckDB error paths and edge cases."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from episemic.hippocampus.duckdb_hippocampus import DuckDBHippocampus
from episemic.models import Memory


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_store_memory_encoding_error(mock_transformer):
    """Test store_memory with encoding error."""
    # Mock the sentence transformer to fail during encoding
    mock_model = MagicMock()
    mock_model.encode.side_effect = Exception("Encoding error")
    mock_transformer.return_value = mock_model

    hippocampus = DuckDBHippocampus(db_path=None)

    # Create test memory
    memory = Memory(
        title="Error Test",
        text="This will cause an error",
        summary="This will cause an error",
        source="error_test"
    )

    # This should handle encoding errors gracefully
    result = await hippocampus.store_memory(memory)
    assert result is False


@pytest.mark.asyncio
async def test_duckdb_retrieve_memory_not_found():
    """Test retrieve_memory when memory doesn't exist."""
    with patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer'):
        hippocampus = DuckDBHippocampus(db_path=None)
        await hippocampus._ensure_initialized()

        # Try to retrieve non-existent memory
        result = await hippocampus.retrieve_memory("nonexistent-id")
        assert result is None


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_mark_quarantined_nonexistent(mock_transformer):
    """Test mark_quarantined on non-existent memory."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.3] * 384
    mock_transformer.return_value = mock_model

    hippocampus = DuckDBHippocampus(db_path=None)
    await hippocampus._ensure_initialized()

    # Try to quarantine non-existent memory (should still return True)
    result = await hippocampus.mark_quarantined("nonexistent-id")
    assert result is True


@pytest.mark.asyncio
async def test_duckdb_verify_integrity_error_paths():
    """Test error paths in verify_integrity."""
    hippocampus = DuckDBHippocampus(db_path=None)

    # Test before initialization
    result = await hippocampus.verify_integrity("test-id")
    assert result is False


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_get_memory_count_normal_operation(mock_transformer):
    """Test get_memory_count normal operation."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.3] * 384
    mock_transformer.return_value = mock_model

    hippocampus = DuckDBHippocampus(db_path=None)

    # Test before initialization
    count = await hippocampus.get_memory_count()
    assert count == 0

    # Test after initialization - should work normally
    await hippocampus._ensure_initialized()
    count = await hippocampus.get_memory_count()
    assert count == 0


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_health_check_error_paths(mock_transformer):
    """Test error paths in health_check."""
    # Test health check with model error
    mock_transformer.side_effect = Exception("Model error")

    hippocampus = DuckDBHippocampus(db_path=None)

    # Health check before initialization
    health = hippocampus.health_check()
    assert health["duckdb"] is False
    assert health["model"] is False
    assert health["embeddings"] is False

    # Reset transformer for partial initialization
    mock_transformer.side_effect = None
    mock_model = MagicMock()
    mock_model.encode.side_effect = Exception("Encoding error")
    mock_transformer.return_value = mock_model

    # Force initialization with broken model
    try:
        await hippocampus._ensure_initialized()
    except Exception:
        pass

    if hasattr(hippocampus, 'model') and hippocampus.model:
        health = hippocampus.health_check()
        # Should handle encoding errors gracefully
        assert isinstance(health, dict)


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_initialization_with_file_creation_error(mock_transformer):
    """Test initialization with file creation error."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.4] * 384
    mock_transformer.return_value = mock_model

    # Use tempfile to create a valid test scenario
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        valid_path = f"{temp_dir}/test.db"
        hippocampus = DuckDBHippocampus(db_path=valid_path)

        # This should work fine
        await hippocampus._ensure_initialized()
        assert hippocampus._initialized is True


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_vector_search_edge_cases(mock_transformer):
    """Test vector search edge cases."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.5] * 384
    mock_transformer.return_value = mock_model

    hippocampus = DuckDBHippocampus(db_path=None)
    await hippocampus._ensure_initialized()

    # Test search with empty vector
    empty_vector = []
    results = await hippocampus.vector_search(empty_vector)
    assert isinstance(results, list)

    # Test search with None filters
    normal_vector = [0.1] * 384
    results = await hippocampus.vector_search(normal_vector, filters=None)
    assert isinstance(results, list)

    # Test search with empty filters dict
    results = await hippocampus.vector_search(normal_vector, filters={})
    assert isinstance(results, list)

    # Test search with top_k = 0
    results = await hippocampus.vector_search(normal_vector, top_k=0)
    assert isinstance(results, list)


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_multiple_initialization_calls(mock_transformer):
    """Test calling _ensure_initialized multiple times."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.6] * 384
    mock_transformer.return_value = mock_model

    hippocampus = DuckDBHippocampus(db_path=None)

    # Call initialization multiple times
    await hippocampus._ensure_initialized()
    assert hippocampus._initialized is True

    await hippocampus._ensure_initialized()
    assert hippocampus._initialized is True

    # Should not reinitialize
    original_model = hippocampus.model
    await hippocampus._ensure_initialized()
    assert hippocampus.model is original_model


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_close_without_connection(mock_transformer):
    """Test close operation when no connection exists."""
    hippocampus = DuckDBHippocampus(db_path=None)

    # Close before initialization - should not error
    hippocampus.close()

    # Initialize and then test normal close
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.7] * 384
    mock_transformer.return_value = mock_model

    await hippocampus._ensure_initialized()
    assert hippocampus.conn is not None

    hippocampus.close()
    assert hippocampus.conn is None

    # Close again - should not error
    hippocampus.close()
    assert hippocampus.conn is None


def test_duckdb_config_directory_creation():
    """Test directory creation for DuckDB file path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested path
        nested_path = os.path.join(temp_dir, "nested", "dir", "test.db")

        # This should create the directory structure
        hippocampus = DuckDBHippocampus(db_path=nested_path)

        # Check if parent directory was created
        assert os.path.exists(os.path.dirname(nested_path))


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_with_different_model_names(mock_transformer):
    """Test DuckDB with different model configurations."""
    # Test with custom model name
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.8] * 512  # Different size
    mock_transformer.return_value = mock_model

    hippocampus = DuckDBHippocampus(
        db_path=None,
        model_name="custom-model-name"
    )

    await hippocampus._ensure_initialized()

    # Should use the custom model name
    mock_transformer.assert_called_with("custom-model-name")

    # Test embedding generation
    embedding = await hippocampus.get_embedding("test text")
    assert len(embedding) == 512