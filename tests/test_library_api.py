"""Tests for the high-level library API."""

import pytest

# Import the internal API for testing (not exposed to users)
from episemic.api import EpistemicAPI, create_config
from episemic.config import EpistemicConfig


def test_config_creation():
    """Test configuration object creation."""
    # Default config
    config = EpistemicConfig()
    assert config.qdrant.host == "localhost"
    assert config.qdrant.port == 6333
    assert config.postgresql.host == "localhost"
    assert config.enable_hippocampus is True

    # Custom config
    config = EpistemicConfig(qdrant={"host": "custom-host", "port": 9999}, debug=True)
    assert config.qdrant.host == "custom-host"
    assert config.qdrant.port == 9999
    assert config.debug is True


def test_config_from_dict():
    """Test creating config from dictionary."""
    config_dict = {"qdrant": {"host": "test-host"}, "debug": True, "enable_cortex": False}

    config = EpistemicConfig.from_dict(config_dict)
    assert config.qdrant.host == "test-host"
    assert config.debug is True
    assert config.enable_cortex is False


def test_config_to_dict():
    """Test exporting config to dictionary."""
    config = EpistemicConfig(debug=True)
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert config_dict["debug"] is True
    assert "qdrant" in config_dict
    assert "postgresql" in config_dict


def test_create_config_helper():
    """Test the create_config helper function."""
    config = create_config(enable_hippocampus=False, debug=True)

    assert config.enable_hippocampus is False
    assert config.debug is True


def test_api_initialization():
    """Test API initialization without external services."""
    # Default config
    api = EpistemicAPI()
    assert api.config is not None
    assert api._initialized is False

    # Custom config
    config = EpistemicConfig(enable_hippocampus=False)
    api = EpistemicAPI(config)
    assert api.config.enable_hippocampus is False


def test_api_not_initialized_error():
    """Test that using API before initialization raises error."""
    api = EpistemicAPI()

    # These should raise RuntimeError since not initialized
    with pytest.raises(RuntimeError, match="not initialized"):
        api._check_initialized()


@pytest.mark.asyncio
async def test_api_initialization_with_disabled_services():
    """Test API initialization with services disabled."""
    config = EpistemicConfig(
        enable_hippocampus=False,
        enable_cortex=False,
        enable_consolidation=False,
        enable_retrieval=False,
    )

    api = EpistemicAPI(config)
    # With all services disabled including hippocampus, initialization should fail
    result = await api.initialize()

    # With all services disabled, initialization should fail
    # because at least hippocampus is required
    assert result is False
    assert api.hippocampus is None
    assert api.cortex is None
    assert api.consolidation_engine is None
    assert api.retrieval_engine is None
    assert api._initialized is False


@pytest.mark.asyncio
async def test_api_context_manager():
    """Test API as async context manager with minimal config."""
    config = EpistemicConfig(
        enable_hippocampus=True,  # Keep hippocampus enabled for minimum functionality
        enable_cortex=False,
        enable_consolidation=False,
        enable_retrieval=False,
    )

    async with EpistemicAPI(config) as api:
        assert api._initialized is True


def test_config_environment_structure():
    """Test that environment config has proper structure."""
    # This tests the structure without actually reading env vars
    config = EpistemicConfig.from_env()

    # Should have all expected sections
    assert hasattr(config, "qdrant")
    assert hasattr(config, "postgresql")
    assert hasattr(config, "redis")
    assert hasattr(config, "consolidation")

    # Should have proper types
    assert isinstance(config.qdrant.host, str)
    assert isinstance(config.qdrant.port, int)
    assert isinstance(config.debug, bool)
