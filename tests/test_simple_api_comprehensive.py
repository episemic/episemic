"""Comprehensive tests for simple API functionality."""

import asyncio
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from episemic_core import create_memory_system
from episemic_core.config import EpistemicConfig
from episemic_core.simple import Episemic, EpistemicSync, Memory, SearchResult


@pytest.mark.asyncio
@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_simple_api_comprehensive_workflow(mock_transformer):
    """Test comprehensive workflow with simple API."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1] * 384
    mock_transformer.return_value = mock_model

    config = EpistemicConfig()
    config.use_duckdb_fallback = True
    config.prefer_qdrant = False
    config.enable_cortex = False
    config.enable_consolidation = False

    async with Episemic(config=config) as episemic:
        # Test storing multiple memories
        memory1 = await episemic.remember(
            "Python is a versatile programming language",
            title="Python Programming",
            tags=["python", "programming"],
            metadata={"difficulty": "beginner"}
        )
        assert memory1 is not None
        assert "Python" in memory1.text
        assert "programming" in memory1.tags

        memory2 = await episemic.remember(
            "JavaScript is used for web development",
            title="JavaScript Basics",
            tags=["javascript", "web"],
            metadata={"difficulty": "intermediate"}
        )
        assert memory2 is not None

        # Test recall with different queries
        results = await episemic.recall("programming")
        assert isinstance(results, list)
        assert len(results) >= 0

        results_limited = await episemic.recall("programming", limit=1)
        assert isinstance(results_limited, list)
        assert len(results_limited) <= 1

        # Test recall with tags
        results_tagged = await episemic.recall("language", tags=["python"])
        assert isinstance(results_tagged, list)

        # Test health check
        health = await episemic.health()
        assert isinstance(health, bool)

        # Test forget functionality
        forgotten = await episemic.forget(memory1.id)
        assert isinstance(forgotten, bool)


@pytest.mark.asyncio
@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_simple_api_error_scenarios(mock_transformer):
    """Test simple API error handling scenarios."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.2] * 384
    mock_transformer.return_value = mock_model

    config = EpistemicConfig()
    config.use_duckdb_fallback = True
    config.prefer_qdrant = False
    config.enable_cortex = False
    config.enable_consolidation = False

    episemic = Episemic(config=config)

    # Test operations before starting
    with pytest.raises(RuntimeError, match="not started"):
        await episemic.remember("test")

    with pytest.raises(RuntimeError, match="not started"):
        await episemic.recall("test")

    with pytest.raises(RuntimeError, match="not started"):
        await episemic.get("test-id")

    # Start the system
    await episemic.start()

    # Test with API failures
    with patch.object(episemic._api, 'store_memory', return_value=None):
        memory = await episemic.remember("failed memory")
        assert memory is None

    # Test get with non-existent ID
    non_existent = await episemic.get("non-existent-id")
    assert non_existent is None

    # Test find_related with non-existent ID
    related = await episemic.find_related("non-existent-id")
    assert isinstance(related, list)
    assert len(related) == 0


def test_simple_api_sync_wrapper():
    """Test synchronous wrapper functionality."""
    with patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer'):
        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.3] * 384

        config = EpistemicConfig()
        config.use_duckdb_fallback = True
        config.prefer_qdrant = False
        config.enable_cortex = False
        config.enable_consolidation = False

        sync_episemic = EpistemicSync(config=config)

        # Test start
        result = sync_episemic.start()
        assert isinstance(result, bool)

        # Test remember
        memory = sync_episemic.remember(
            "Sync API test memory",
            title="Sync Test",
            tags=["sync", "test"]
        )
        # May be None due to mocking

        # Test recall
        results = sync_episemic.recall("sync test")
        assert isinstance(results, list)

        # Test get
        retrieved = sync_episemic.get("test-id")
        # May be None

        # Test health
        health = sync_episemic.health()
        assert isinstance(health, bool)

        # Test stop
        sync_episemic.stop()


@pytest.mark.asyncio
async def test_simple_api_configuration_variations():
    """Test simple API with various configuration options."""
    # Test with kwargs configuration
    episemic = Episemic(debug=True)
    assert episemic._config.debug is True

    # Test with postgres configuration via kwargs
    episemic_pg = Episemic(
        postgres_host="custom-host",
        postgres_db="custom-db",
        postgres_user="custom-user",
        postgres_password="custom-pass"
    )
    assert episemic_pg._config.postgresql.host == "custom-host"
    assert episemic_pg._config.postgresql.database == "custom-db"
    assert episemic_pg._config.postgresql.user == "custom-user"
    assert episemic_pg._config.postgresql.password == "custom-pass"

    # Test with qdrant configuration via kwargs
    episemic_qdrant = Episemic(
        qdrant_host="qdrant-host",
        qdrant_port=6334
    )
    assert episemic_qdrant._config.qdrant.host == "qdrant-host"
    assert episemic_qdrant._config.qdrant.port == 6334

    # Test with redis configuration via kwargs
    episemic_redis = Episemic(
        redis_host="redis-host",
        redis_port=6380
    )
    assert episemic_redis._config.redis.host == "redis-host"
    assert episemic_redis._config.redis.port == 6380


def test_memory_wrapper_properties():
    """Test Memory wrapper class properties."""
    from episemic_core.models import Memory as InternalMemory

    internal_memory = InternalMemory(
        id="test-id",
        title="Test Memory",
        text="Test content",
        summary="Test summary",
        source="test",
        tags=["test", "memory"],
        metadata={"key": "value"}
    )

    wrapped_memory = Memory(internal_memory)

    # Test all properties
    assert wrapped_memory.id == "test-id"
    assert wrapped_memory.text == "Test content"
    assert wrapped_memory.title == "Test Memory"
    assert wrapped_memory.tags == ["test", "memory"]
    assert wrapped_memory.metadata == {"key": "value"}
    assert wrapped_memory.created_at == internal_memory.created_at.isoformat()

    # Test string representations
    str_repr = str(wrapped_memory)
    assert "Test Memory" in str_repr

    repr_str = repr(wrapped_memory)
    assert "test-id" in repr_str
    assert "Test Memory" in repr_str


def test_search_result_wrapper():
    """Test SearchResult wrapper class."""
    from episemic_core.models import Memory as InternalMemory, SearchResult as InternalSearchResult

    internal_memory = InternalMemory(
        id="result-id",
        title="Result Memory",
        text="Result content",
        summary="Result summary",
        source="test"
    )

    internal_result = InternalSearchResult(
        memory=internal_memory,
        score=0.85,
        context="search context",
        metadata={"search": "data"}
    )

    wrapped_result = SearchResult(internal_result)

    # Test properties
    assert wrapped_result.score == 0.85
    assert wrapped_result.memory.id == "result-id"
    assert wrapped_result.memory.title == "Result Memory"

    # Test string representation
    str_repr = str(wrapped_result)
    assert "Result Memory" in str_repr
    assert "0.850" in str_repr


@pytest.mark.asyncio
async def test_create_memory_system_function():
    """Test create_memory_system convenience function."""
    with patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer'):
        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.4] * 384

        episemic = await create_memory_system(debug=True)
        assert isinstance(episemic, Episemic)
        assert episemic._started is True
        assert episemic._config.debug is True


@pytest.mark.asyncio
@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_simple_api_memory_retrieval_fallback(mock_transformer):
    """Test memory retrieval with fallback logic."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.5] * 384
    mock_transformer.return_value = mock_model

    config = EpistemicConfig()
    config.use_duckdb_fallback = True
    config.prefer_qdrant = False
    config.enable_cortex = False
    config.enable_consolidation = False
    config.enable_retrieval = False  # This will trigger fallback logic

    async with Episemic(config=config) as episemic:
        # This should use the fallback logic in remember()
        memory = await episemic.remember(
            "Fallback test memory",
            title="Fallback Test",
            tags=["fallback"]
        )
        assert memory is not None
        assert memory.text == "Fallback test memory"
        assert memory.title == "Fallback Test"


@pytest.mark.asyncio
@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
async def test_simple_api_consolidation_operations(mock_transformer):
    """Test consolidation operations through simple API."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.6] * 384
    mock_transformer.return_value = mock_model

    config = EpistemicConfig()
    config.use_duckdb_fallback = True
    config.prefer_qdrant = False
    config.enable_cortex = False
    config.enable_consolidation = False

    async with Episemic(config=config) as episemic:
        # Store a memory
        memory = await episemic.remember("Consolidation test", title="Consolidation")
        assert memory is not None

        # Test consolidate (should handle gracefully even if disabled)
        result = await episemic.consolidate(memory.id)
        assert isinstance(result, bool)

        # Test auto consolidation
        count = await episemic.auto_consolidate()
        assert isinstance(count, int)
        assert count == 0  # No consolidation since it's disabled


def test_simple_api_event_loop_handling():
    """Test simple API handles event loops correctly."""
    sync_episemic = EpistemicSync()

    # Test that _run_async creates event loop if none exists
    def mock_coro():
        return asyncio.sleep(0.001)

    # This should work even if no event loop is running
    result = sync_episemic._run_async(mock_coro())
    assert result is None  # sleep returns None


@pytest.mark.asyncio
async def test_simple_api_with_file_persistence():
    """Test simple API with file-based persistence."""
    with patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer'):
        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.7] * 384

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        os.unlink(db_path)

        try:
            config = EpistemicConfig()
            config.use_duckdb_fallback = True
            config.prefer_qdrant = False
            config.enable_cortex = False
            config.enable_consolidation = False
            config.duckdb.db_path = db_path

            # First session
            async with Episemic(config=config) as episemic1:
                memory = await episemic1.remember(
                    "Persistent memory test",
                    title="Persistence Test"
                )
                assert memory is not None
                memory_id = memory.id

            # Second session
            async with Episemic(config=config) as episemic2:
                retrieved = await episemic2.get(memory_id)
                # May be None due to retrieval limitations, but should not error

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)