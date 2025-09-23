"""Tests for the simple API."""

import pytest

from episemic import Episemic, EpistemicSync, Memory, SearchResult


def test_simple_imports():
    """Test that the simple API imports correctly."""
    from episemic import Episemic, EpistemicSync, Memory, SearchResult, create_memory_system

    # Should be able to create instances
    episemic = Episemic()
    assert episemic is not None

    episemic_sync = EpistemicSync()
    assert episemic_sync is not None


def test_episemic_initialization():
    """Test Episemic initialization with various configs."""
    # Default config
    episemic = Episemic()
    assert episemic._config is not None
    assert episemic._started is False

    # Custom config
    episemic = Episemic(debug=True, postgres_host="custom-host", qdrant_port=9999)
    assert episemic._config.debug is True
    assert episemic._config.postgresql.host == "custom-host"
    assert episemic._config.qdrant.port == 9999


def test_episemic_not_started_error():
    """Test that using Episemic before start() raises error."""
    episemic = Episemic()

    with pytest.raises(RuntimeError, match="not started"):
        episemic._check_started()


@pytest.mark.asyncio
async def test_episemic_start_with_disabled_services():
    """Test that Episemic can start with services disabled."""
    # This should work even without external services
    episemic = Episemic(debug=True)

    # Mock the underlying API to avoid actual service connections
    episemic._api.config.enable_hippocampus = False
    episemic._api.config.enable_cortex = False
    episemic._api.config.enable_consolidation = False
    episemic._api.config.enable_retrieval = False

    await episemic.start()
    assert episemic._started is True  # Should be True even if some services fail


@pytest.mark.asyncio
async def test_episemic_context_manager():
    """Test Episemic as async context manager."""
    # Mock services disabled
    async with Episemic() as episemic:
        # Should be started automatically
        assert episemic._started is True


def test_sync_episemic():
    """Test synchronous Episemic wrapper."""
    episemic = EpistemicSync(debug=True)

    # Should have the async version internally
    assert episemic._async_episemic is not None
    assert isinstance(episemic._async_episemic, Episemic)


def test_memory_wrapper():
    """Test Memory wrapper class."""
    from episemic.models import Memory as InternalMemory

    # Create a mock internal memory
    internal = InternalMemory(
        title="Test Memory",
        text="This is test content",
        summary="Test summary",
        source="test",
        tags=["test", "memory"],
    )

    # Wrap it
    memory = Memory(internal)

    # Test properties
    assert memory.title == "Test Memory"
    assert memory.text == "This is test content"
    assert memory.tags == ["test", "memory"]
    assert len(memory.id) > 0
    assert memory.created_at  # Should be ISO format string

    # Test string representation
    assert "Test Memory" in str(memory)
    assert "Test Memory" in repr(memory)


def test_search_result_wrapper():
    """Test SearchResult wrapper class."""
    from episemic.models import Memory as InternalMemory, SearchResult as InternalSearchResult

    # Create mock internal objects
    internal_memory = InternalMemory(
        title="Search Result",
        text="Found content",
        summary="Found summary",
        source="test",
        tags=["found"],
    )

    internal_result = InternalSearchResult(memory=internal_memory, score=0.85)

    # Wrap it
    result = SearchResult(internal_result)

    # Test properties
    assert isinstance(result.memory, Memory)
    assert result.memory.title == "Search Result"
    assert result.score == 0.85

    # Test string representation
    assert "Search Result" in str(result)
    assert "0.85" in str(result)


@pytest.mark.asyncio
async def test_create_memory_system_function():
    """Test the create_memory_system convenience function."""
    from episemic import create_memory_system

    episemic = await create_memory_system(debug=True)
    assert isinstance(episemic, Episemic)
    assert episemic._started is True
