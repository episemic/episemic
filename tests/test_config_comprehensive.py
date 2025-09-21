"""Comprehensive tests for configuration system."""

import os
from unittest.mock import patch

import pytest

from episemic_core.config import (
    EpistemicConfig,
    QdrantConfig,
    DuckDBConfig,
    PostgreSQLConfig,
    RedisConfig,
    ConsolidationConfig
)


def test_qdrant_config():
    """Test QdrantConfig creation and defaults."""
    config = QdrantConfig()
    assert config.host == "localhost"
    assert config.port == 6333
    assert config.collection_name == "episodic_memories"
    assert config.vector_size == 768

    # Test custom values
    custom_config = QdrantConfig(
        host="custom-host",
        port=6334,
        collection_name="custom_collection",
        vector_size=512
    )
    assert custom_config.host == "custom-host"
    assert custom_config.port == 6334
    assert custom_config.collection_name == "custom_collection"
    assert custom_config.vector_size == 512


def test_duckdb_config():
    """Test DuckDBConfig creation and defaults."""
    config = DuckDBConfig()
    assert config.db_path is None
    assert config.model_name == "all-MiniLM-L6-v2"

    # Test custom values
    custom_config = DuckDBConfig(
        db_path="/tmp/test.db",
        model_name="custom-model"
    )
    assert custom_config.db_path == "/tmp/test.db"
    assert custom_config.model_name == "custom-model"


def test_postgresql_config():
    """Test PostgreSQLConfig creation and defaults."""
    config = PostgreSQLConfig()
    assert config.host == "localhost"
    assert config.port == 5432
    assert config.database == "episemic"
    assert config.user == "postgres"
    assert config.password == "postgres"

    # Test custom values
    custom_config = PostgreSQLConfig(
        host="pg-host",
        port=5433,
        database="custom_db",
        user="custom_user",
        password="custom_pass"
    )
    assert custom_config.host == "pg-host"
    assert custom_config.port == 5433
    assert custom_config.database == "custom_db"
    assert custom_config.user == "custom_user"
    assert custom_config.password == "custom_pass"


def test_redis_config():
    """Test RedisConfig creation and defaults."""
    config = RedisConfig()
    assert config.host == "localhost"
    assert config.port == 6379
    assert config.db == 0
    assert config.ttl == 3600

    # Test custom values
    custom_config = RedisConfig(
        host="redis-host",
        port=6380,
        db=1,
        ttl=7200
    )
    assert custom_config.host == "redis-host"
    assert custom_config.port == 6380
    assert custom_config.db == 1
    assert custom_config.ttl == 7200


def test_consolidation_config():
    """Test ConsolidationConfig creation and defaults."""
    config = ConsolidationConfig()
    assert config.threshold_hours == 2
    assert config.access_threshold == 3
    assert config.auto_consolidation_enabled is True
    assert config.consolidation_interval_minutes == 60

    # Test custom values
    custom_config = ConsolidationConfig(
        threshold_hours=4,
        access_threshold=5,
        auto_consolidation_enabled=False,
        consolidation_interval_minutes=120
    )
    assert custom_config.threshold_hours == 4
    assert custom_config.access_threshold == 5
    assert custom_config.auto_consolidation_enabled is False
    assert custom_config.consolidation_interval_minutes == 120


def test_episemic_config_defaults():
    """Test EpistemicConfig creation and defaults."""
    config = EpistemicConfig()

    # Test nested config defaults
    assert isinstance(config.qdrant, QdrantConfig)
    assert isinstance(config.duckdb, DuckDBConfig)
    assert isinstance(config.postgresql, PostgreSQLConfig)
    assert isinstance(config.redis, RedisConfig)
    assert isinstance(config.consolidation, ConsolidationConfig)

    # Test storage preferences
    assert config.use_duckdb_fallback is True
    assert config.prefer_qdrant is False

    # Test global settings
    assert config.enable_hippocampus is True
    assert config.enable_cortex is True
    assert config.enable_consolidation is True
    assert config.enable_retrieval is True

    # Test development settings
    assert config.debug is False
    assert config.log_level == "INFO"


def test_episemic_config_from_dict():
    """Test EpistemicConfig.from_dict functionality."""
    config_dict = {
        "qdrant": {
            "host": "custom-qdrant",
            "port": 6334
        },
        "duckdb": {
            "db_path": "/tmp/custom.db",
            "model_name": "custom-model"
        },
        "postgresql": {
            "host": "custom-pg",
            "database": "custom_db"
        },
        "use_duckdb_fallback": False,
        "prefer_qdrant": True,
        "debug": True,
        "log_level": "DEBUG"
    }

    config = EpistemicConfig.from_dict(config_dict)

    assert config.qdrant.host == "custom-qdrant"
    assert config.qdrant.port == 6334
    assert config.duckdb.db_path == "/tmp/custom.db"
    assert config.duckdb.model_name == "custom-model"
    assert config.postgresql.host == "custom-pg"
    assert config.postgresql.database == "custom_db"
    assert config.use_duckdb_fallback is False
    assert config.prefer_qdrant is True
    assert config.debug is True
    assert config.log_level == "DEBUG"


def test_episemic_config_to_dict():
    """Test EpistemicConfig.to_dict functionality."""
    config = EpistemicConfig()
    config.debug = True
    config.log_level = "DEBUG"

    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert config_dict["debug"] is True
    assert config_dict["log_level"] == "DEBUG"
    assert "qdrant" in config_dict
    assert "duckdb" in config_dict
    assert "postgresql" in config_dict
    assert "redis" in config_dict
    assert "consolidation" in config_dict


def test_episemic_config_from_env_complete():
    """Test EpistemicConfig.from_env with comprehensive environment variables."""
    env_vars = {
        # Qdrant settings
        "QDRANT_HOST": "env-qdrant-host",
        "QDRANT_PORT": "6334",
        "QDRANT_COLLECTION": "env_collection",

        # DuckDB settings
        "DUCKDB_PATH": "/env/path/test.db",
        "DUCKDB_MODEL": "env-model",

        # Storage backend preferences
        "EPISEMIC_USE_DUCKDB": "false",
        "EPISEMIC_PREFER_QDRANT": "true",

        # PostgreSQL settings
        "POSTGRES_HOST": "env-pg-host",
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "env_db",
        "POSTGRES_USER": "env_user",
        "POSTGRES_PASSWORD": "env_pass",

        # Redis settings
        "REDIS_HOST": "env-redis-host",
        "REDIS_PORT": "6380",
        "REDIS_DB": "2",

        # Debug settings
        "EPISEMIC_DEBUG": "true",
        "EPISEMIC_LOG_LEVEL": "DEBUG"
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = EpistemicConfig.from_env()

        # Test Qdrant settings
        assert config.qdrant.host == "env-qdrant-host"
        assert config.qdrant.port == 6334
        assert config.qdrant.collection_name == "env_collection"

        # Test DuckDB settings
        assert config.duckdb.db_path == "/env/path/test.db"
        assert config.duckdb.model_name == "env-model"

        # Test storage preferences
        assert config.use_duckdb_fallback is False
        assert config.prefer_qdrant is True

        # Test PostgreSQL settings
        assert config.postgresql.host == "env-pg-host"
        assert config.postgresql.port == 5433
        assert config.postgresql.database == "env_db"
        assert config.postgresql.user == "env_user"
        assert config.postgresql.password == "env_pass"

        # Test Redis settings
        assert config.redis.host == "env-redis-host"
        assert config.redis.port == 6380
        assert config.redis.db == 2

        # Test debug settings
        assert config.debug is True
        assert config.log_level == "DEBUG"


def test_episemic_config_from_env_partial():
    """Test EpistemicConfig.from_env with partial environment variables."""
    env_vars = {
        "QDRANT_HOST": "partial-qdrant",
        "POSTGRES_DB": "partial_db",
        "EPISEMIC_DEBUG": "yes",  # Test different truthy value
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = EpistemicConfig.from_env()

        # Test set values
        assert config.qdrant.host == "partial-qdrant"
        assert config.postgresql.database == "partial_db"
        assert config.debug is True

        # Test default values remain
        assert config.qdrant.port == 6333  # Default
        assert config.postgresql.host == "localhost"  # Default
        assert config.log_level == "INFO"  # Default


def test_episemic_config_from_env_boolean_variations():
    """Test EpistemicConfig.from_env with various boolean value formats."""
    test_cases = [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("invalid", False),  # Invalid values should default to False
    ]

    for env_value, expected in test_cases:
        env_vars = {"EPISEMIC_DEBUG": env_value}
        with patch.dict(os.environ, env_vars, clear=False):
            config = EpistemicConfig.from_env()
            assert config.debug is expected, f"Failed for env_value: {env_value}"


def test_episemic_config_from_env_no_env_vars():
    """Test EpistemicConfig.from_env when no environment variables are set."""
    # Clear relevant environment variables
    env_vars_to_clear = [
        "QDRANT_HOST", "QDRANT_PORT", "QDRANT_COLLECTION",
        "DUCKDB_PATH", "DUCKDB_MODEL",
        "EPISEMIC_USE_DUCKDB", "EPISEMIC_PREFER_QDRANT",
        "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD",
        "REDIS_HOST", "REDIS_PORT", "REDIS_DB",
        "EPISEMIC_DEBUG", "EPISEMIC_LOG_LEVEL"
    ]

    # Backup existing values
    backup_env = {}
    for var in env_vars_to_clear:
        if var in os.environ:
            backup_env[var] = os.environ[var]
            del os.environ[var]

    try:
        config = EpistemicConfig.from_env()

        # Should have all default values
        assert config.qdrant.host == "localhost"
        assert config.qdrant.port == 6333
        assert config.duckdb.db_path is None
        assert config.duckdb.model_name == "all-MiniLM-L6-v2"
        assert config.use_duckdb_fallback is True
        assert config.prefer_qdrant is False
        assert config.debug is False
        assert config.log_level == "INFO"

    finally:
        # Restore environment variables
        for var, value in backup_env.items():
            os.environ[var] = value


def test_episemic_config_from_env_integer_conversion():
    """Test EpistemicConfig.from_env with integer conversion."""
    env_vars = {
        "QDRANT_PORT": "6334",
        "POSTGRES_PORT": "5433",
        "REDIS_PORT": "6380",
        "REDIS_DB": "3"
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = EpistemicConfig.from_env()

        assert config.qdrant.port == 6334
        assert config.postgresql.port == 5433
        assert config.redis.port == 6380
        assert config.redis.db == 3


def test_episemic_config_custom_initialization():
    """Test EpistemicConfig with custom initialization parameters."""
    config = EpistemicConfig(
        use_duckdb_fallback=False,
        prefer_qdrant=True,
        enable_cortex=False,
        enable_consolidation=False,
        debug=True,
        log_level="WARNING"
    )

    assert config.use_duckdb_fallback is False
    assert config.prefer_qdrant is True
    assert config.enable_cortex is False
    assert config.enable_consolidation is False
    assert config.debug is True
    assert config.log_level == "WARNING"

    # Nested configs should still have defaults
    assert config.qdrant.host == "localhost"
    assert config.duckdb.db_path is None


def test_config_model_validation():
    """Test Pydantic model validation in configs."""
    # Test invalid port (should be positive integer)
    with pytest.raises(Exception):  # Pydantic will raise validation error
        QdrantConfig(port=-1)

    # Test empty string for required field
    with pytest.raises(Exception):
        PostgreSQLConfig(database="")

    # Test invalid model data types
    with pytest.raises(Exception):
        DuckDBConfig(model_name=123)  # Should be string