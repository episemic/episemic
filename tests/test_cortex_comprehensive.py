"""Comprehensive tests for the cortex module with mocked PostgreSQL."""

import pytest
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime
import json

from episemic.cortex.cortex import Cortex
from episemic.models import Memory, MemoryLink, LinkType, MemoryStatus, RetentionPolicy


@pytest.fixture
def mock_psycopg2():
    """Mock psycopg2 and its connection/cursor objects."""
    with patch("episemic.cortex.cortex.psycopg2") as mock_pg:
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Setup connection context manager
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)

        # Setup cursor context manager
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        # Connect cursor to connection
        mock_conn.cursor.return_value = mock_cursor
        mock_pg.connect.return_value = mock_conn

        yield {"psycopg2": mock_pg, "connection": mock_conn, "cursor": mock_cursor}


@pytest.fixture
def cortex(mock_psycopg2):
    """Create a cortex instance with mocked database."""
    return Cortex(
        db_host="test_host",
        db_port=5432,
        db_name="test_db",
        db_user="test_user",
        db_password="test_password",
    )


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        id="test-memory-id",
        title="Test Memory",
        text="This is a test memory",
        summary="Test memory summary",
        source="test",
        tags=["test", "memory"],
        metadata={"category": "test"},
        links=[MemoryLink(target_id="linked-memory-id", type=LinkType.CITES, weight=0.8)],
    )


def test_cortex_initialization(mock_psycopg2):
    """Test cortex initialization and database setup."""
    cortex = Cortex(
        db_host="localhost",
        db_port=5432,
        db_name="episemic",
        db_user="postgres",
        db_password="postgres",
    )

    # Verify connection parameters
    assert cortex.connection_params == {
        "host": "localhost",
        "port": 5432,
        "database": "episemic",
        "user": "postgres",
        "password": "postgres",
    }

    # Verify database setup was called
    mock_psycopg2["psycopg2"].connect.assert_called()
    mock_psycopg2["cursor"].execute.assert_called()


def test_cortex_initialization_custom_params(mock_psycopg2):
    """Test cortex initialization with custom parameters."""
    cortex = Cortex(
        db_host="custom-host",
        db_port=5433,
        db_name="custom-db",
        db_user="custom-user",
        db_password="custom-pass",
    )

    expected_params = {
        "host": "custom-host",
        "port": 5433,
        "database": "custom-db",
        "user": "custom-user",
        "password": "custom-pass",
    }

    assert cortex.connection_params == expected_params


def test_get_connection(cortex, mock_psycopg2):
    """Test _get_connection method."""
    conn = cortex._get_connection()

    mock_psycopg2["psycopg2"].connect.assert_called_with(**cortex.connection_params)
    assert conn == mock_psycopg2["connection"]


@pytest.mark.asyncio
async def test_store_memory_success(cortex, mock_psycopg2, sample_memory):
    """Test successful memory storage."""
    result = await cortex.store_memory(sample_memory)

    assert result is True

    # Verify main memory insert was called
    calls = mock_psycopg2["cursor"].execute.call_args_list

    # Check that memory insert was called
    memory_insert_call = None
    for call in calls:
        if "INSERT INTO memories" in str(call):
            memory_insert_call = call
            break

    assert memory_insert_call is not None

    # Verify commit was called
    mock_psycopg2["connection"].commit.assert_called()


@pytest.mark.asyncio
async def test_store_memory_with_links(cortex, mock_psycopg2, sample_memory):
    """Test storing memory with links."""
    result = await cortex.store_memory(sample_memory)

    assert result is True

    # Verify link insert was called
    calls = mock_psycopg2["cursor"].execute.call_args_list
    link_insert_call = None
    for call in calls:
        if "INSERT INTO memory_links" in str(call):
            link_insert_call = call
            break

    assert link_insert_call is not None


@pytest.mark.asyncio
async def test_store_memory_exception_handling(cortex, mock_psycopg2):
    """Test memory storage exception handling."""
    # Make execute raise an exception
    mock_psycopg2["cursor"].execute.side_effect = Exception("Database error")

    memory = Memory(
        id="error-memory",
        title="Error Memory",
        text="Error content",
        summary="Error summary",
        source="test",
    )

    with patch("builtins.print") as mock_print:
        result = await cortex.store_memory(memory)

    assert result is False
    mock_print.assert_called_with("Error storing memory in Cortex: Database error")


@pytest.mark.asyncio
async def test_retrieve_memory_success(cortex, mock_psycopg2):
    """Test successful memory retrieval."""
    # Mock database row
    mock_row = {
        "id": "test-memory-id",
        "title": "Test Memory",
        "text": "Test content",
        "summary": "Test summary",
        "source": "test",
        "source_ref": None,
        "tags": ["test"],
        "metadata": {},
        "created_at": datetime.utcnow(),
        "ingested_at": datetime.utcnow(),
        "hash": "test-hash",
        "version": 1,
        "access_count": 0,
        "last_accessed": None,
        "retention_policy": "default",
        "status": "active",
        "checksum_status": "unknown",
        "embedding_v1": None,
        "embedding_v2": None,
    }

    # Mock links
    mock_links = [{"target_id": "linked-memory", "link_type": "cites", "weight": 0.8}]

    # Setup cursor to return our mock data
    mock_psycopg2["cursor"].fetchone.return_value = mock_row
    mock_psycopg2["cursor"].fetchall.return_value = mock_links

    result = await cortex.retrieve_memory("test-memory-id")

    assert result is not None
    assert result.id == "test-memory-id"
    assert result.title == "Test Memory"
    assert len(result.links) == 1
    assert result.links[0].target_id == "linked-memory"


@pytest.mark.asyncio
async def test_retrieve_memory_not_found(cortex, mock_psycopg2):
    """Test memory retrieval when memory not found."""
    mock_psycopg2["cursor"].fetchone.return_value = None

    result = await cortex.retrieve_memory("nonexistent-id")

    assert result is None


@pytest.mark.asyncio
async def test_retrieve_memory_exception_handling(cortex, mock_psycopg2):
    """Test memory retrieval exception handling."""
    mock_psycopg2["cursor"].execute.side_effect = Exception("Retrieval error")

    with patch("builtins.print") as mock_print:
        result = await cortex.retrieve_memory("error-memory")

    assert result is None
    mock_print.assert_called_with("Error retrieving memory from Cortex: Retrieval error")


@pytest.mark.asyncio
async def test_search_by_tags_success(cortex, mock_psycopg2):
    """Test successful tag-based search."""
    # Mock search results
    mock_rows = [
        {
            "id": "memory1",
            "title": "Memory 1",
            "text": "Content 1",
            "summary": "Summary 1",
            "source": "test",
            "source_ref": None,
            "tags": ["python", "programming"],
            "metadata": {},
            "created_at": datetime.utcnow(),
            "ingested_at": datetime.utcnow(),
            "hash": "hash1",
            "version": 1,
            "access_count": 0,
            "last_accessed": None,
            "retention_policy": "default",
            "status": "active",
            "checksum_status": "unknown",
            "embedding_v1": None,
            "embedding_v2": None,
        },
        {
            "id": "memory2",
            "title": "Memory 2",
            "text": "Content 2",
            "summary": "Summary 2",
            "source": "test",
            "source_ref": None,
            "tags": ["python", "tutorial"],
            "metadata": {},
            "created_at": datetime.utcnow(),
            "ingested_at": datetime.utcnow(),
            "hash": "hash2",
            "version": 1,
            "access_count": 0,
            "last_accessed": None,
            "retention_policy": "default",
            "status": "active",
            "checksum_status": "unknown",
            "embedding_v1": None,
            "embedding_v2": None,
        },
    ]

    mock_psycopg2["cursor"].fetchall.return_value = mock_rows

    results = await cortex.search_by_tags(["python"], limit=10)

    assert len(results) == 2
    assert results[0].id == "memory1"
    assert results[1].id == "memory2"
    assert "python" in results[0].tags
    assert "python" in results[1].tags


@pytest.mark.asyncio
async def test_search_by_tags_empty_results(cortex, mock_psycopg2):
    """Test tag search with no results."""
    mock_psycopg2["cursor"].fetchall.return_value = []

    results = await cortex.search_by_tags(["nonexistent"], limit=10)

    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_by_tags_exception_handling(cortex, mock_psycopg2):
    """Test tag search exception handling."""
    mock_psycopg2["cursor"].execute.side_effect = Exception("Search error")

    with patch("builtins.print") as mock_print:
        results = await cortex.search_by_tags(["test"])

    assert results == []
    mock_print.assert_called_with("Error searching by tags: Search error")


@pytest.mark.asyncio
async def test_get_memory_graph_success(cortex, mock_psycopg2):
    """Test successful memory graph retrieval."""
    # Mock nodes and edges
    mock_nodes = [
        {"id": "node1", "title": "Node 1", "depth": 0},
        {"id": "node2", "title": "Node 2", "depth": 1},
    ]

    mock_edges = [{"source_id": "node1", "target_id": "node2", "link_type": "cites", "weight": 0.8}]

    # Setup cursor to return nodes first, then edges
    mock_psycopg2["cursor"].fetchall.side_effect = [mock_nodes, mock_edges]

    result = await cortex.get_memory_graph("node1", depth=2)

    assert "nodes" in result
    assert "edges" in result
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1
    assert result["nodes"][0]["id"] == "node1"
    assert result["edges"][0]["source_id"] == "node1"


@pytest.mark.asyncio
async def test_get_memory_graph_exception_handling(cortex, mock_psycopg2):
    """Test memory graph exception handling."""
    mock_psycopg2["cursor"].execute.side_effect = Exception("Graph error")

    with patch("builtins.print") as mock_print:
        result = await cortex.get_memory_graph("error-node")

    assert result == {"nodes": [], "edges": []}
    mock_print.assert_called_with("Error getting memory graph: Graph error")


@pytest.mark.asyncio
async def test_increment_access_count_success(cortex, mock_psycopg2):
    """Test successful access count increment."""
    await cortex.increment_access_count("test-memory-id")

    # Verify the UPDATE query was called
    mock_psycopg2["cursor"].execute.assert_called()
    mock_psycopg2["connection"].commit.assert_called()

    # Check the SQL query
    calls = mock_psycopg2["cursor"].execute.call_args_list
    update_call = None
    for call in calls:
        if "UPDATE memories" in str(call):
            update_call = call
            break

    assert update_call is not None


@pytest.mark.asyncio
async def test_increment_access_count_exception_handling(cortex, mock_psycopg2):
    """Test access count increment exception handling."""
    mock_psycopg2["cursor"].execute.side_effect = Exception("Update error")

    with patch("builtins.print") as mock_print:
        await cortex.increment_access_count("error-memory")

    mock_print.assert_called_with("Error incrementing access count: Update error")


@pytest.mark.asyncio
async def test_mark_deleted_success(cortex, mock_psycopg2):
    """Test successful memory deletion marking."""
    await cortex.mark_deleted("test-memory-id")

    # Verify the UPDATE query was called
    mock_psycopg2["cursor"].execute.assert_called()
    mock_psycopg2["connection"].commit.assert_called()


@pytest.mark.asyncio
async def test_mark_deleted_exception_handling(cortex, mock_psycopg2):
    """Test memory deletion marking exception handling."""
    mock_psycopg2["cursor"].execute.side_effect = Exception("Delete error")

    with patch("builtins.print") as mock_print:
        await cortex.mark_deleted("error-memory")

    mock_print.assert_called_with("Error marking memory as deleted: Delete error")


def test_health_check_success(cortex, mock_psycopg2):
    """Test successful health check."""
    mock_psycopg2["cursor"].fetchone.return_value = (1,)

    result = cortex.health_check()

    assert result is True
    mock_psycopg2["cursor"].execute.assert_called_with("SELECT 1")


def test_health_check_failure(cortex, mock_psycopg2):
    """Test health check failure."""
    mock_psycopg2["cursor"].fetchone.return_value = None

    result = cortex.health_check()

    assert result is False


def test_health_check_exception_handling(cortex, mock_psycopg2):
    """Test health check exception handling."""
    mock_psycopg2["psycopg2"].connect.side_effect = Exception("Connection error")

    result = cortex.health_check()

    assert result is False


def test_database_setup_table_creation(mock_psycopg2):
    """Test that database setup creates the required tables."""
    cortex = Cortex()

    # Check that execute was called multiple times for table creation
    assert mock_psycopg2["cursor"].execute.call_count >= 3

    # Verify commit was called
    mock_psycopg2["connection"].commit.assert_called()


@pytest.mark.asyncio
async def test_store_memory_with_complex_metadata(cortex, mock_psycopg2):
    """Test storing memory with complex metadata."""
    memory = Memory(
        id="complex-memory",
        title="Complex Memory",
        text="Complex content",
        summary="Complex summary",
        source="test",
        metadata={"nested": {"key": "value"}, "list": [1, 2, 3], "string": "test"},
    )

    result = await cortex.store_memory(memory)

    assert result is True

    # Verify that the metadata was JSON serialized in the call
    calls = mock_psycopg2["cursor"].execute.call_args_list
    memory_insert_call = None
    for call in calls:
        if "INSERT INTO memories" in str(call):
            memory_insert_call = call
            break

    assert memory_insert_call is not None
    # The metadata should be JSON dumped
    args = memory_insert_call[0][1]
    # Find metadata in the arguments (it's the second to last argument before embeddings)
    metadata_arg = args[-3]  # metadata is 3rd from end (before two embedding fields)
    assert '"nested":' in metadata_arg  # Should be JSON string


@pytest.mark.asyncio
async def test_search_by_tags_with_custom_limit(cortex, mock_psycopg2):
    """Test tag search with custom limit."""
    mock_psycopg2["cursor"].fetchall.return_value = []

    await cortex.search_by_tags(["test"], limit=5)

    # Verify the limit was passed to the query
    call_args = mock_psycopg2["cursor"].execute.call_args
    assert call_args[0][1] == (["test"], 5)


def test_cortex_connection_parameters_validation(mock_psycopg2):
    """Test that cortex properly validates and stores connection parameters."""
    cortex = Cortex(
        db_host="test-host",
        db_port=9999,
        db_name="test-database",
        db_user="test-user",
        db_password="secret-password",
    )

    expected_params = {
        "host": "test-host",
        "port": 9999,
        "database": "test-database",
        "user": "test-user",
        "password": "secret-password",
    }

    assert cortex.connection_params == expected_params


@pytest.mark.asyncio
async def test_memory_retrieval_with_no_links(cortex, mock_psycopg2):
    """Test memory retrieval when memory has no links."""
    # Mock database row
    mock_row = {
        "id": "no-links-memory",
        "title": "No Links Memory",
        "text": "Content without links",
        "summary": "Summary",
        "source": "test",
        "source_ref": None,
        "tags": ["test"],
        "metadata": {},
        "created_at": datetime.utcnow(),
        "ingested_at": datetime.utcnow(),
        "hash": "test-hash",
        "version": 1,
        "access_count": 0,
        "last_accessed": None,
        "retention_policy": "default",
        "status": "active",
        "checksum_status": "unknown",
        "embedding_v1": None,
        "embedding_v2": None,
    }

    # Setup cursor to return memory but no links
    mock_psycopg2["cursor"].fetchone.return_value = mock_row
    mock_psycopg2["cursor"].fetchall.return_value = []  # No links

    result = await cortex.retrieve_memory("no-links-memory")

    assert result is not None
    assert result.id == "no-links-memory"
    assert len(result.links) == 0


@pytest.mark.asyncio
async def test_get_memory_graph_with_custom_depth(cortex, mock_psycopg2):
    """Test memory graph retrieval with custom depth."""
    mock_psycopg2["cursor"].fetchall.side_effect = [[], []]  # Empty nodes and edges

    await cortex.get_memory_graph("test-node", depth=5)

    # Verify depth parameter was passed correctly
    # Check all execute calls to find the one with the depth parameter
    execute_calls = mock_psycopg2["cursor"].execute.call_args_list
    found_depth_call = False
    for call in execute_calls:
        if len(call[0]) > 1 and call[0][1] == ("test-node", 5):
            found_depth_call = True
            break

    assert found_depth_call, "Expected call with depth parameter not found"
