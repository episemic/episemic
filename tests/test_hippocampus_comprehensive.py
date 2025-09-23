"""Comprehensive tests for the hippocampus module with mocked Qdrant and Redis."""

import pytest
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime
import json

from episemic.hippocampus.hippocampus import Hippocampus
from episemic.models import Memory, MemoryStatus, RetentionPolicy


@pytest.fixture
def mock_qdrant():
    """Mock QdrantClient and its methods."""
    with patch("episemic.hippocampus.hippocampus.QdrantClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock collections response
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        yield mock_client


@pytest.fixture
def mock_redis():
    """Mock Redis client and its methods."""
    with patch("episemic.hippocampus.hippocampus.redis") as mock_redis_module:
        mock_redis_client = MagicMock()
        mock_redis_module.Redis.return_value = mock_redis_client

        # Setup default return values
        mock_redis_client.get.return_value = None
        mock_redis_client.ping.return_value = True
        mock_redis_client.setex.return_value = True
        mock_redis_client.delete.return_value = True

        yield mock_redis_client


@pytest.fixture
def hippocampus(mock_qdrant, mock_redis):
    """Create a hippocampus instance with mocked dependencies."""
    return Hippocampus(
        qdrant_host="test_qdrant",
        qdrant_port=6333,
        redis_host="test_redis",
        redis_port=6379,
        collection_name="test_memories",
    )


@pytest.fixture
def sample_memory():
    """Create a sample memory with embedding for testing."""
    return Memory(
        id="test-memory-id",
        title="Test Memory",
        text="This is a test memory",
        summary="Test memory summary",
        source="test",
        tags=["test", "memory"],
        metadata={"category": "test"},
        embedding_v1=[0.1] * 768,  # Required for hippocampus
        status=MemoryStatus.ACTIVE,
        retention_policy=RetentionPolicy.DEFAULT,
    )


def test_hippocampus_initialization(mock_qdrant, mock_redis):
    """Test hippocampus initialization."""
    hippocampus = Hippocampus(
        qdrant_host="localhost",
        qdrant_port=6333,
        redis_host="localhost",
        redis_port=6379,
        collection_name="episodic_memories",
    )

    assert hippocampus.qdrant_client == mock_qdrant
    assert hippocampus.redis_client == mock_redis
    assert hippocampus.collection_name == "episodic_memories"

    # Verify setup_collection was called
    mock_qdrant.get_collections.assert_called()


def test_hippocampus_initialization_custom_params(mock_qdrant, mock_redis):
    """Test hippocampus initialization with custom parameters."""
    hippocampus = Hippocampus(
        qdrant_host="custom-qdrant",
        qdrant_port=9999,
        redis_host="custom-redis",
        redis_port=8888,
        collection_name="custom_memories",
    )

    assert hippocampus.collection_name == "custom_memories"


def test_setup_collection_creates_new(mock_qdrant, mock_redis):
    """Test collection creation when it doesn't exist."""
    # Mock empty collections
    mock_collections = MagicMock()
    mock_collections.collections = []
    mock_qdrant.get_collections.return_value = mock_collections

    hippocampus = Hippocampus(collection_name="new_collection")

    # Verify collection creation was called
    mock_qdrant.create_collection.assert_called_once()


def test_setup_collection_exists(mock_qdrant, mock_redis):
    """Test setup when collection already exists."""
    # Mock existing collection
    mock_collection = MagicMock()
    mock_collection.name = "existing_collection"
    mock_collections = MagicMock()
    mock_collections.collections = [mock_collection]
    mock_qdrant.get_collections.return_value = mock_collections

    hippocampus = Hippocampus(collection_name="existing_collection")

    # Verify collection creation was NOT called
    mock_qdrant.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_store_memory_success(hippocampus, mock_qdrant, mock_redis, sample_memory):
    """Test successful memory storage."""
    result = await hippocampus.store_memory(sample_memory)

    assert result is True

    # Verify Qdrant upsert was called
    mock_qdrant.upsert.assert_called_once()
    call_args = mock_qdrant.upsert.call_args
    assert call_args[1]["collection_name"] == "test_memories"
    assert len(call_args[1]["points"]) == 1

    # Verify Redis cache was set
    mock_redis.setex.assert_called_once()
    redis_call_args = mock_redis.setex.call_args
    assert redis_call_args[0][0] == "memory:test-memory-id"
    assert redis_call_args[0][1] == 3600  # TTL


@pytest.mark.asyncio
async def test_store_memory_no_embedding(hippocampus, mock_qdrant, mock_redis):
    """Test storing memory without embedding fails."""
    memory_no_embedding = Memory(
        id="no-embedding",
        title="No Embedding Memory",
        text="No embedding content",
        summary="No embedding summary",
        source="test",
        # No embedding_v1
    )

    with patch("builtins.print") as mock_print:
        result = await hippocampus.store_memory(memory_no_embedding)

    assert result is False
    mock_print.assert_called_with(
        "Error storing memory in Hippocampus: Memory must have embedding_v1 to store in Hippocampus"
    )

    # Verify Qdrant and Redis were not called
    mock_qdrant.upsert.assert_not_called()
    mock_redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_store_memory_qdrant_error(hippocampus, mock_qdrant, mock_redis, sample_memory):
    """Test memory storage with Qdrant error."""
    mock_qdrant.upsert.side_effect = Exception("Qdrant error")

    with patch("builtins.print") as mock_print:
        result = await hippocampus.store_memory(sample_memory)

    assert result is False
    mock_print.assert_called_with("Error storing memory in Hippocampus: Qdrant error")


@pytest.mark.asyncio
async def test_retrieve_memory_from_cache(hippocampus, mock_redis, sample_memory):
    """Test memory retrieval from Redis cache."""
    # Mock Redis to return cached memory
    mock_redis.get.return_value = sample_memory.model_dump_json()

    result = await hippocampus.retrieve_memory("test-memory-id")

    assert result is not None
    assert result.id == "test-memory-id"
    assert result.title == "Test Memory"

    # Verify Redis was called but Qdrant was not
    mock_redis.get.assert_called_with("memory:test-memory-id")


@pytest.mark.asyncio
async def test_retrieve_memory_from_qdrant(hippocampus, mock_qdrant, mock_redis, sample_memory):
    """Test memory retrieval from Qdrant when not in cache."""
    # Mock Redis cache miss
    mock_redis.get.return_value = None

    # Mock Qdrant retrieval
    mock_point = MagicMock()
    mock_point.id = "test-memory-id"
    mock_point.vector = [0.1] * 768
    mock_point.payload = {
        "title": "Test Memory",
        "text": "This is a test memory",
        "summary": "Test memory summary",
        "source": "test",
        "tags": ["test", "memory"],
        "created_at": datetime.utcnow().isoformat(),
        "retention_policy": "default",
        "status": "active",
        "hash": "test-hash",
    }
    mock_qdrant.retrieve.return_value = [mock_point]

    result = await hippocampus.retrieve_memory("test-memory-id")

    assert result is not None
    assert result.id == "test-memory-id"
    assert result.text == "This is a test memory"

    # Verify Qdrant was called
    mock_qdrant.retrieve.assert_called_with(collection_name="test_memories", ids=["test-memory-id"])

    # Verify Redis cache was updated
    mock_redis.setex.assert_called()


@pytest.mark.asyncio
async def test_retrieve_memory_not_found(hippocampus, mock_qdrant, mock_redis):
    """Test memory retrieval when memory doesn't exist."""
    # Mock cache miss
    mock_redis.get.return_value = None

    # Mock Qdrant not finding the memory
    mock_qdrant.retrieve.return_value = []

    result = await hippocampus.retrieve_memory("nonexistent-id")

    assert result is None


@pytest.mark.asyncio
async def test_retrieve_memory_qdrant_error(hippocampus, mock_qdrant, mock_redis):
    """Test memory retrieval with Qdrant error."""
    # Mock cache miss
    mock_redis.get.return_value = None

    # Mock Qdrant error
    mock_qdrant.retrieve.side_effect = Exception("Qdrant error")

    with patch("builtins.print") as mock_print:
        result = await hippocampus.retrieve_memory("error-memory")

    assert result is None
    mock_print.assert_called_with("Error retrieving memory from Hippocampus: Qdrant error")


@pytest.mark.asyncio
async def test_vector_search_success(hippocampus, mock_qdrant):
    """Test successful vector search."""
    # Mock search results
    mock_point1 = MagicMock()
    mock_point1.id = "memory1"
    mock_point1.vector = [0.1] * 768
    mock_point1.score = 0.9
    mock_point1.payload = {
        "title": "Search Result 1",
        "text": "Search result 1",
        "summary": "Summary 1",
        "source": "test",
        "tags": ["search"],
        "created_at": datetime.utcnow().isoformat(),
        "retention_policy": "default",
        "status": "active",
        "hash": "hash1",
    }

    mock_point2 = MagicMock()
    mock_point2.id = "memory2"
    mock_point2.vector = [0.2] * 768
    mock_point2.score = 0.8
    mock_point2.payload = {
        "title": "Search Result 2",
        "text": "Search result 2",
        "summary": "Summary 2",
        "source": "test",
        "tags": ["search"],
        "created_at": datetime.utcnow().isoformat(),
        "retention_policy": "default",
        "status": "active",
        "hash": "hash2",
    }

    mock_qdrant.search.return_value = [mock_point1, mock_point2]

    query_vector = [0.5] * 768
    results = await hippocampus.vector_search(
        query_vector=query_vector, top_k=10, filters={"tags": ["search"]}
    )

    assert len(results) == 2
    assert results[0]["memory"].id == "memory1"
    assert results[0]["score"] == 0.9
    assert results[1]["memory"].id == "memory2"
    assert results[1]["score"] == 0.8

    # Verify Qdrant search was called correctly
    mock_qdrant.search.assert_called_with(
        collection_name="test_memories",
        query_vector=query_vector,
        limit=10,
        query_filter={"tags": ["search"]},
    )


@pytest.mark.asyncio
async def test_vector_search_empty_results(hippocampus, mock_qdrant):
    """Test vector search with no results."""
    mock_qdrant.search.return_value = []

    results = await hippocampus.vector_search(query_vector=[0.5] * 768, top_k=5)

    assert len(results) == 0


@pytest.mark.asyncio
async def test_vector_search_error(hippocampus, mock_qdrant):
    """Test vector search with error."""
    mock_qdrant.search.side_effect = Exception("Search error")

    with patch("builtins.print") as mock_print:
        results = await hippocampus.vector_search(query_vector=[0.5] * 768, top_k=5)

    assert results == []
    mock_print.assert_called_with("Error in vector search: Search error")


@pytest.mark.asyncio
async def test_mark_quarantined_success(hippocampus, mock_qdrant, mock_redis):
    """Test successful memory quarantine."""
    result = await hippocampus.mark_quarantined("test-memory-id")

    assert result is True

    # Verify Qdrant payload update
    mock_qdrant.set_payload.assert_called_with(
        collection_name="test_memories",
        payload={"status": "quarantined"},
        points=["test-memory-id"],
    )

    # Verify Redis cache invalidation
    mock_redis.delete.assert_called_with("memory:test-memory-id")


@pytest.mark.asyncio
async def test_mark_quarantined_error(hippocampus, mock_qdrant, mock_redis):
    """Test memory quarantine with error."""
    mock_qdrant.set_payload.side_effect = Exception("Quarantine error")

    with patch("builtins.print") as mock_print:
        result = await hippocampus.mark_quarantined("error-memory")

    assert result is False
    mock_print.assert_called_with("Error marking memory as quarantined: Quarantine error")


@pytest.mark.asyncio
async def test_verify_integrity_success(hippocampus, mock_redis, sample_memory):
    """Test successful memory integrity verification."""
    # Mock memory retrieval
    mock_redis.get.return_value = sample_memory.model_dump_json()

    # Mock the verify_integrity method on the Memory class
    with patch.object(Memory, "verify_integrity", return_value=True):
        result = await hippocampus.verify_integrity("test-memory-id")

    assert result is True


@pytest.mark.asyncio
async def test_verify_integrity_memory_not_found(hippocampus, mock_redis):
    """Test integrity verification when memory not found."""
    mock_redis.get.return_value = None

    # Mock Qdrant also returning no results
    with patch.object(hippocampus, "retrieve_memory", return_value=None):
        result = await hippocampus.verify_integrity("nonexistent-id")

    assert result is False


@pytest.mark.asyncio
async def test_verify_integrity_verification_fails(hippocampus, mock_redis, sample_memory):
    """Test integrity verification when verification fails."""
    # Mock memory retrieval
    mock_redis.get.return_value = sample_memory.model_dump_json()

    # Mock the verify_integrity method to fail
    with patch.object(Memory, "verify_integrity", return_value=False):
        result = await hippocampus.verify_integrity("test-memory-id")

    assert result is False


def test_health_check_all_healthy(hippocampus, mock_qdrant, mock_redis):
    """Test health check when all services are healthy."""
    # Mock successful connections
    mock_qdrant.get_collections.return_value = MagicMock()
    mock_redis.ping.return_value = True

    health = hippocampus.health_check()

    assert health == {"qdrant_connected": True, "redis_connected": True}


def test_health_check_qdrant_unhealthy(hippocampus, mock_qdrant, mock_redis):
    """Test health check when Qdrant is unhealthy."""
    # Mock Qdrant connection failure
    mock_qdrant.get_collections.side_effect = Exception("Qdrant down")
    mock_redis.ping.return_value = True

    health = hippocampus.health_check()

    assert health == {"qdrant_connected": False, "redis_connected": True}


def test_health_check_redis_unhealthy(hippocampus, mock_qdrant, mock_redis):
    """Test health check when Redis is unhealthy."""
    # Mock Redis connection failure
    mock_qdrant.get_collections.return_value = MagicMock()
    mock_redis.ping.side_effect = Exception("Redis down")

    health = hippocampus.health_check()

    assert health == {"qdrant_connected": True, "redis_connected": False}


def test_health_check_all_unhealthy(hippocampus, mock_qdrant, mock_redis):
    """Test health check when all services are unhealthy."""
    # Mock both connection failures
    mock_qdrant.get_collections.side_effect = Exception("Qdrant down")
    mock_redis.ping.side_effect = Exception("Redis down")

    health = hippocampus.health_check()

    assert health == {"qdrant_connected": False, "redis_connected": False}


def test_check_qdrant_connection_success(hippocampus, mock_qdrant):
    """Test Qdrant connection check success."""
    mock_qdrant.get_collections.return_value = MagicMock()

    result = hippocampus._check_qdrant_connection()

    assert result is True


def test_check_qdrant_connection_failure(hippocampus, mock_qdrant):
    """Test Qdrant connection check failure."""
    mock_qdrant.get_collections.side_effect = Exception("Connection failed")

    result = hippocampus._check_qdrant_connection()

    assert result is False


def test_check_redis_connection_success(hippocampus, mock_redis):
    """Test Redis connection check success."""
    mock_redis.ping.return_value = True

    result = hippocampus._check_redis_connection()

    assert result is True


def test_check_redis_connection_failure(hippocampus, mock_redis):
    """Test Redis connection check failure."""
    mock_redis.ping.side_effect = Exception("Connection failed")

    result = hippocampus._check_redis_connection()

    assert result is False


def test_check_redis_connection_false_response(hippocampus, mock_redis):
    """Test Redis connection check with False response."""
    mock_redis.ping.return_value = False

    result = hippocampus._check_redis_connection()

    assert result is False


@pytest.mark.asyncio
async def test_vector_search_with_none_filters(hippocampus, mock_qdrant):
    """Test vector search with None filters."""
    mock_qdrant.search.return_value = []

    await hippocampus.vector_search(query_vector=[0.5] * 768, top_k=5, filters=None)

    # Verify search was called with None filters
    mock_qdrant.search.assert_called_with(
        collection_name="test_memories", query_vector=[0.5] * 768, limit=5, query_filter=None
    )


@pytest.mark.asyncio
async def test_store_memory_with_complex_tags(hippocampus, mock_qdrant, mock_redis):
    """Test storing memory with complex tags."""
    memory_with_tags = Memory(
        id="tagged-memory",
        title="Tagged Memory",
        text="Tagged content",
        summary="Tagged summary",
        source="test",
        tags=["python", "programming", "machine-learning", "AI"],
        embedding_v1=[0.3] * 768,
    )

    result = await hippocampus.store_memory(memory_with_tags)

    assert result is True

    # Verify the tags were included in the payload
    call_args = mock_qdrant.upsert.call_args
    point = call_args[1]["points"][0]
    assert "python" in point.payload["tags"]
    assert "machine-learning" in point.payload["tags"]


@pytest.mark.asyncio
async def test_vector_search_with_empty_payload(hippocampus, mock_qdrant):
    """Test vector search when point has empty payload."""
    # Mock point with minimal required payload
    mock_point = MagicMock()
    mock_point.id = "empty-payload-memory"
    mock_point.vector = [0.1] * 768
    mock_point.score = 0.7
    mock_point.payload = {
        "title": "Empty Payload Memory",
        "text": "Content",
        "summary": "Summary",
        "source": "test",
        "hash": "test-hash",
    }  # Minimal required fields

    mock_qdrant.search.return_value = [mock_point]

    results = await hippocampus.vector_search(query_vector=[0.5] * 768, top_k=1)

    assert len(results) == 1
    assert results[0]["memory"].id == "empty-payload-memory"
    assert results[0]["score"] == 0.7


@pytest.mark.asyncio
async def test_vector_search_with_null_payload(hippocampus, mock_qdrant):
    """Test vector search when point has null payload."""
    # This test checks error handling when payload is null
    mock_point = MagicMock()
    mock_point.id = "null-payload-memory"
    mock_point.vector = [0.1] * 768
    mock_point.score = 0.6
    mock_point.payload = None  # Null payload

    mock_qdrant.search.return_value = [mock_point]

    with patch("builtins.print") as mock_print:
        results = await hippocampus.vector_search(query_vector=[0.5] * 768, top_k=1)

    # Should return empty results due to validation error
    assert len(results) == 0
    # Should have printed an error
    assert mock_print.called
