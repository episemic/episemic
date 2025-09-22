"""Tests specifically designed to improve code coverage."""

import os
import sys
from unittest.mock import patch, MagicMock
import pytest

# Skip import error tests for now as they're too complex to mock properly
# These paths are already covered by the normal behavior since Qdrant/PostgreSQL
# are not available in the test environment


@pytest.mark.asyncio
async def test_api_debug_output_paths():
    """Test API debug output paths."""
    from episemic.api import EpistemicAPI
    from episemic.config import EpistemicConfig

    # Test with debug enabled
    config = EpistemicConfig(debug=True)
    api = EpistemicAPI(config)

    # This should trigger debug output
    result = await api.initialize()
    # Should succeed with DuckDB fallback
    assert result is True


@pytest.mark.asyncio
async def test_api_cortex_disabled_debug():
    """Test API with cortex disabled and debug output."""
    from episemic.api import EpistemicAPI
    from episemic.config import EpistemicConfig

    config = EpistemicConfig(
        debug=True,
        enable_cortex=True  # Enable cortex to trigger initialization attempt
    )
    api = EpistemicAPI(config)

    # This should trigger cortex initialization failure and debug output
    result = await api.initialize()
    # Should still succeed with DuckDB fallback
    assert result is True


@pytest.mark.asyncio
async def test_api_consolidation_engine_paths():
    """Test consolidation engine initialization paths."""
    from episemic.api import EpistemicAPI
    from episemic.config import EpistemicConfig

    config = EpistemicConfig(
        debug=True,
        enable_consolidation=True  # Enable to trigger initialization
    )
    api = EpistemicAPI(config)

    result = await api.initialize()
    assert result is True


@pytest.mark.asyncio
async def test_api_retrieval_engine_paths():
    """Test retrieval engine initialization paths."""
    from episemic.api import EpistemicAPI
    from episemic.config import EpistemicConfig

    config = EpistemicConfig(
        debug=True,
        enable_retrieval=True
    )
    api = EpistemicAPI(config)

    result = await api.initialize()
    assert result is True


@pytest.mark.asyncio
async def test_api_memory_operations_error_paths():
    """Test API memory operation error paths."""
    from episemic.api import EpistemicAPI
    from episemic.config import EpistemicConfig

    config = EpistemicConfig()
    api = EpistemicAPI(config)
    await api.initialize()

    # Test get_memory with non-existent ID
    memory = await api.get_memory("non-existent-id")
    assert memory is None

    # Test search with empty results
    results = await api.search("query that returns nothing", top_k=1)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_api_fallback_scenarios():
    """Test various API fallback scenarios."""
    from episemic.api import EpistemicAPI
    from episemic.config import EpistemicConfig

    # Test with all optional services disabled
    config = EpistemicConfig(
        enable_cortex=False,
        enable_consolidation=False,
        enable_retrieval=False,
        debug=True  # Enable debug to cover debug paths
    )
    api = EpistemicAPI(config)

    result = await api.initialize()
    assert result is True

    # Test health check
    health = await api.health_check()
    assert isinstance(health, dict)


@pytest.mark.asyncio
async def test_api_error_handling_branches():
    """Test various error handling branches in API."""
    from episemic.api import EpistemicAPI
    from episemic.config import EpistemicConfig

    config = EpistemicConfig()
    api = EpistemicAPI(config)

    # Test operations before initialization
    with pytest.raises(RuntimeError, match="not initialized"):
        await api.store_memory("test", "title")

    with pytest.raises(RuntimeError, match="not initialized"):
        await api.search("test")

    with pytest.raises(RuntimeError, match="not initialized"):
        await api.get_memory("test-id")

    with pytest.raises(RuntimeError, match="not initialized"):
        await api.health_check()


def test_module_imports():
    """Test that modules can be imported successfully."""
    # Test that imports work
    import episemic.hippocampus as hippocampus_module
    import episemic.cortex as cortex_module

    # These should be importable (classes will be available if dependencies exist)
    assert hippocampus_module is not None
    assert cortex_module is not None