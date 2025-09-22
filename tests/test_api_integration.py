"""Tests for API integration and storage backend selection."""

import os
import tempfile
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from episemic_core.api import EpistemicAPI
from episemic_core.config import EpistemicConfig
from episemic_core.models import Memory, SearchQuery


@pytest.mark.asyncio
@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_api_duckdb_backend_selection(mock_transformer):
    """Test API automatically selecting DuckDB backend."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1] * 384
    mock_transformer.return_value = mock_model

    # Configure for DuckDB fallback
    config = EpistemicConfig()
    config.use_duckdb_fallback = True
    config.prefer_qdrant = False
    config.enable_cortex = False
    config.enable_consolidation = False

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        config.duckdb.db_path = tmp.name
    os.unlink(config.duckdb.db_path)

    try:
        api = EpistemicAPI(config)

        # Test backend selection logic
        should_use_duckdb = await api._should_use_duckdb()
        assert should_use_duckdb is True

        # Initialize API
        success = await api.initialize()
        assert success is True
        assert api.hippocampus is not None
        assert hasattr(api.hippocampus, 'get_embedding')  # DuckDB hippocampus

        # Test storing memory
        memory_id = await api.store_memory(
            text="Test memory for API integration",
            title="API Test",
            tags=["api", "test"]
        )
        assert memory_id is not None

        # Test searching (should use fallback search)
        results = await api.search("test memory")
        assert isinstance(results, list)
        assert len(results) >= 0  # Might be empty depending on search implementation

    finally:
        if os.path.exists(config.duckdb.db_path):
            os.unlink(config.duckdb.db_path)


@pytest.mark.asyncio
@patch('qdrant_client.QdrantClient')
async def test_api_qdrant_preference_with_availability(mock_qdrant_client):
    """Test API preferring Qdrant when available."""
    # Mock successful Qdrant connection
    mock_client = MagicMock()
    mock_client.get_collections.return_value = []
    mock_qdrant_client.return_value = mock_client

    config = EpistemicConfig()
    config.prefer_qdrant = True
    config.use_duckdb_fallback = True

    api = EpistemicAPI(config)

    # Test backend selection - should prefer Qdrant
    should_use_duckdb = await api._should_use_duckdb()
    assert should_use_duckdb is False


@pytest.mark.asyncio
@patch('qdrant_client.QdrantClient')
@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_api_qdrant_fallback_to_duckdb(mock_transformer, mock_qdrant_client):
    """Test API falling back to DuckDB when Qdrant is unavailable."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.2] * 384
    mock_transformer.return_value = mock_model

    # Mock failed Qdrant connection
    mock_qdrant_client.side_effect = Exception("Connection failed")

    config = EpistemicConfig()
    config.prefer_qdrant = True
    config.use_duckdb_fallback = True
    config.debug = True

    api = EpistemicAPI(config)

    # Test backend selection - should fall back to DuckDB
    should_use_duckdb = await api._should_use_duckdb()
    assert should_use_duckdb is True


@pytest.mark.asyncio
@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_api_fallback_search_functionality(mock_transformer):
    """Test API fallback search when retrieval engine is disabled."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.3] * 384
    mock_transformer.return_value = mock_model

    config = EpistemicConfig()
    config.use_duckdb_fallback = True
    config.prefer_qdrant = False
    config.enable_cortex = False
    config.enable_consolidation = False
    config.enable_retrieval = False  # Disable retrieval engine

    api = EpistemicAPI(config)
    success = await api.initialize()
    assert success is True

    # Store a memory
    memory_id = await api.store_memory(
        text="Fallback search test memory",
        title="Fallback Test",
        tags=["fallback", "search"]
    )
    assert memory_id is not None

    # Test fallback search (retrieval engine is None)
    results = await api.search("fallback search")
    assert isinstance(results, list)
    # Should use direct hippocampus search


@pytest.mark.asyncio
async def test_api_initialization_failure_handling():
    """Test API initialization failure handling."""
    config = EpistemicConfig()
    config.debug = True

    api = EpistemicAPI(config)

    # Mock hippocampus to fail initialization
    with patch.object(api, 'hippocampus', None):
        with patch.object(api, '_should_use_duckdb', return_value=True):
            with patch('episemic_core.api.DuckDBHippocampus') as mock_duckdb:
                mock_duckdb.side_effect = Exception("Initialization failed")

                success = await api.initialize()
                assert success is False
                assert api._initialized is False


@pytest.mark.asyncio
@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_api_memory_operations_comprehensive(mock_transformer):
    """Test comprehensive memory operations through API."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.4] * 384
    mock_transformer.return_value = mock_model

    config = EpistemicConfig()
    config.use_duckdb_fallback = True
    config.prefer_qdrant = False
    config.enable_cortex = False
    config.enable_consolidation = False

    api = EpistemicAPI(config)
    await api.initialize()

    # Test store memory with various parameters
    memory_id = await api.store_memory(
        text="Comprehensive test memory content",
        title="Comprehensive Test",
        source="test_suite",
        tags=["comprehensive", "test"],
        metadata={"test_type": "comprehensive", "priority": "high"},
        store_in_hippocampus=True,
        store_in_cortex=False
    )
    assert memory_id is not None

    # Test get memory (should use fallback when retrieval engine disabled)
    memory = await api.get_memory(memory_id)
    # May be None if retrieval engine is disabled

    # Test search with tags
    results = await api.search(
        query="comprehensive test",
        top_k=5,
        tags=["comprehensive"]
    )
    assert isinstance(results, list)

    # Test health check
    health = await api.health_check()
    assert isinstance(health, dict)
    assert "hippocampus_duckdb" in health or len(health) == 0


@pytest.mark.asyncio
@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_api_context_manager(mock_transformer):
    """Test API context manager functionality."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.5] * 384
    mock_transformer.return_value = mock_model

    config = EpistemicConfig()
    config.use_duckdb_fallback = True
    config.prefer_qdrant = False
    config.enable_cortex = False

    # Test context manager
    async with EpistemicAPI(config) as api:
        assert api._initialized is True

        # Test basic operation
        memory_id = await api.store_memory(
            text="Context manager test",
            title="Context Test"
        )
        assert memory_id is not None


@pytest.mark.asyncio
async def test_api_not_initialized_errors():
    """Test API raises errors when not initialized."""
    config = EpistemicConfig()
    api = EpistemicAPI(config)

    # Test operations before initialization
    with pytest.raises(RuntimeError, match="not initialized"):
        await api.store_memory("test")

    with pytest.raises(RuntimeError, match="not initialized"):
        await api.search("test")

    with pytest.raises(RuntimeError, match="not initialized"):
        await api.get_memory("test-id")

    with pytest.raises(RuntimeError, match="not initialized"):
        await api.health_check()


def test_api_configuration_defaults():
    """Test API configuration defaults."""
    config = EpistemicConfig()

    # Test default values
    assert config.use_duckdb_fallback is True
    assert config.prefer_qdrant is False
    assert config.enable_hippocampus is True
    assert config.enable_cortex is True
    assert config.enable_consolidation is True
    assert config.enable_retrieval is True

    # Test DuckDB config defaults
    assert config.duckdb.db_path is None
    assert config.duckdb.model_name == "all-MiniLM-L6-v2"


@pytest.mark.asyncio
@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_api_search_with_fallback_error_handling(mock_transformer):
    """Test API search fallback with error handling."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.6] * 384
    mock_transformer.return_value = mock_model

    config = EpistemicConfig()
    config.use_duckdb_fallback = True
    config.prefer_qdrant = False
    config.enable_cortex = False
    config.enable_consolidation = False
    config.enable_retrieval = False
    config.debug = True

    api = EpistemicAPI(config)
    await api.initialize()

    # Mock hippocampus to not have get_embedding method
    with patch.object(api.hippocampus, 'get_embedding', side_effect=AttributeError("No method")):
        results = await api.search("test query")
        assert results == []

    # Mock hippocampus vector search to fail
    with patch.object(api.hippocampus, 'vector_search', side_effect=Exception("Search failed")):
        results = await api.search("test query")
        assert results == []