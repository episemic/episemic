"""Tests to cover specific missing lines in various modules."""

import json
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from episemic.config import EpistemicConfig
from episemic.models import Memory


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_hippocampus_embedding_error_fallback(mock_transformer):
    """Test DuckDB hippocampus fallback when embedding generation fails."""
    from episemic.hippocampus.duckdb_hippocampus import DuckDBHippocampus

    # Mock transformer to fail during embedding generation
    mock_transformer.side_effect = Exception("Embedding generation failed")

    hippocampus = DuckDBHippocampus(db_path=None)

    memory = Memory(
        title="Test Memory",
        text="Test content",
        summary="Test summary",
        source="test"
    )

    # This should trigger the embedding error fallback (line 92)
    result = await hippocampus.store_memory(memory)
    assert result is True  # Should still work without embeddings


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_various_error_paths(mock_transformer):
    """Test various error paths in DuckDB hippocampus."""
    from episemic.hippocampus.duckdb_hippocampus import DuckDBHippocampus

    # Mock successful transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1] * 384
    mock_transformer.return_value = mock_model

    hippocampus = DuckDBHippocampus(db_path=None)
    await hippocampus._ensure_initialized()

    # Test vector search with empty results (lines around 249)
    vector_results = await hippocampus.vector_search([0.1] * 384, top_k=5)
    assert vector_results == []

    # Test retrieve_memory with non-existent ID (lines around 155)
    memory = await hippocampus.retrieve_memory("non-existent-id")
    assert memory is None

    # Test vector search with empty results (lines around 249)
    vector_results = await hippocampus.vector_search([0.1] * 384, top_k=5)
    assert vector_results == []


@pytest.mark.asyncio
async def test_consolidation_engine_error_paths():
    """Test consolidation engine error handling paths."""
    from episemic.consolidation.consolidation import ConsolidationEngine

    # Mock hippocampus and cortex with errors
    mock_hippocampus = MagicMock()
    mock_cortex = MagicMock()

    engine = ConsolidationEngine(mock_hippocampus, mock_cortex)

    # Test consolidation with exception
    # Let's test the actual available method
    result = await engine.consolidate_memory("test-id")
    # This should work with mocked dependencies
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_retrieval_engine_error_paths():
    """Test retrieval engine error handling paths."""
    from episemic.retrieval.retrieval import RetrievalEngine

    # Mock components
    mock_hippocampus = MagicMock()
    mock_cortex = None  # Simulate no cortex
    mock_consolidation = None

    engine = RetrievalEngine(mock_hippocampus, mock_cortex)

    # Test search with no results (lines around 64-66)
    mock_hippocampus.search.return_value = []
    mock_hippocampus.vector_search.return_value = []

    from episemic.models import SearchQuery
    query = SearchQuery(query="test", top_k=5)

    results = await engine.search(query)
    assert results == []


@pytest.mark.asyncio
async def test_simple_api_error_paths():
    """Test simple API error handling paths."""
    from episemic.simple import Episemic

    config = EpistemicConfig()
    episemic = Episemic(config=config)

    # Test operations before starting (line 149)
    with pytest.raises(RuntimeError, match="not started"):
        await episemic.recall("test")

    # Start the system
    await episemic.start()

    # Test remember with metadata that triggers JSON serialization
    memory = await episemic.remember(
        "Test content",
        title="Test",
        metadata={"complex": {"nested": "data"}}
    )
    assert memory is not None


# Skip CLI tests as they require complex setup


def test_hippocampus_import_error_path():
    """Test hippocampus import error paths."""
    from episemic.hippocampus.hippocampus import Hippocampus

    # Test initialization (lines 8-14)
    with pytest.raises(Exception):
        # This should fail since we don't have Qdrant/Redis in test environment
        hippocampus = Hippocampus()


@pytest.mark.asyncio
async def test_config_edge_cases():
    """Test configuration edge cases."""
    from episemic.config import EpistemicConfig

    # Test config creation with various parameters
    config = EpistemicConfig(
        debug=True,
        enable_cortex=True,
        enable_consolidation=True,
        enable_retrieval=True
    )

    assert config.debug is True
    assert config.enable_cortex is True


@pytest.mark.asyncio
@patch('episemic.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_duckdb_file_operations(mock_transformer):
    """Test DuckDB file-based operations."""
    from episemic.hippocampus.duckdb_hippocampus import DuckDBHippocampus

    # Mock transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1] * 384
    mock_transformer.return_value = mock_model

    # Test with file-based database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    os.unlink(db_path)

    try:
        hippocampus = DuckDBHippocampus(db_path=db_path)
        await hippocampus._ensure_initialized()

        memory = Memory(
            title="File Test",
            text="File content",
            summary="File summary",
            source="test"
        )

        result = await hippocampus.store_memory(memory)
        assert result is True

        # Test retrieval
        retrieved = await hippocampus.retrieve_memory(memory.id)
        assert retrieved is not None
        assert retrieved.title == "File Test"

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_models_edge_cases():
    """Test model edge cases and methods."""
    from episemic.models import Memory, MemoryLink, LinkType

    # Test memory with all fields
    memory = Memory(
        title="Complete Memory",
        text="Complete content",
        summary="Complete summary",
        source="test",
        tags=["tag1", "tag2"],
        metadata={"key": "value"}
    )

    # Test memory methods
    original_hash = memory.hash
    assert memory.verify_integrity() is True

    # Modify memory and check integrity
    memory.text = "Modified content"
    assert memory.verify_integrity() is False

    # Test access increment
    original_count = memory.access_count
    memory.increment_access()
    assert memory.access_count == original_count + 1

    # Test memory link
    link = MemoryLink(
        target_id="target-123",
        type=LinkType.CITES,
        weight=0.8
    )
    assert link.target_id == "target-123"
    assert link.type == LinkType.CITES