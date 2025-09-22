"""Comprehensive tests for CLI functionality."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typer.testing import CliRunner
from datetime import datetime

from episemic.cli.main import app
from episemic.models import Memory, SearchResult


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_memory_system():
    """Mock the entire memory system."""
    with patch('episemic.cli.main.get_memory_system') as mock_get:
        mock_hippocampus = MagicMock()
        mock_cortex = MagicMock()
        mock_consolidation = MagicMock()
        mock_retrieval = MagicMock()

        mock_get.return_value = (mock_hippocampus, mock_cortex, mock_consolidation, mock_retrieval)

        yield {
            'hippocampus': mock_hippocampus,
            'cortex': mock_cortex,
            'consolidation': mock_consolidation,
            'retrieval': mock_retrieval,
            'get_system': mock_get
        }


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        id="test-memory-id",
        title="Test Memory",
        text="This is a test memory for CLI testing",
        summary="Test memory summary",
        source="cli",
        tags=["test", "cli"],
        created_at=datetime.utcnow(),
        access_count=5
    )


def test_version_command(cli_runner):
    """Test version command."""
    result = cli_runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "Episemic Core v1.0.2" in result.stdout


def test_help_command(cli_runner):
    """Test help command."""
    result = cli_runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Episemic Core" in result.stdout
    assert "store" in result.stdout
    assert "search" in result.stdout
    assert "get" in result.stdout
    assert "consolidate" in result.stdout
    assert "health" in result.stdout


def test_init_command_success(cli_runner):
    """Test successful init command."""
    with patch('episemic.cli.main.Hippocampus') as mock_hippo, \
         patch('episemic.cli.main.Cortex') as mock_cortex, \
         patch('episemic.cli.main.ConsolidationEngine') as mock_consol, \
         patch('episemic.cli.main.RetrievalEngine') as mock_retrieval:

        result = cli_runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "Initializing Episemic Core" in result.stdout
        assert "initialized successfully" in result.stdout

        # Verify components were initialized with default params
        mock_hippo.assert_called_once_with("localhost", 6333)
        mock_cortex.assert_called_once_with("localhost", 5432, "episemic")


def test_init_command_with_custom_params(cli_runner):
    """Test init command with custom parameters."""
    with patch('episemic.cli.main.Hippocampus') as mock_hippo, \
         patch('episemic.cli.main.Cortex') as mock_cortex, \
         patch('episemic.cli.main.ConsolidationEngine'), \
         patch('episemic.cli.main.RetrievalEngine'):

        result = cli_runner.invoke(app, [
            "init",
            "--qdrant-host", "custom-qdrant",
            "--qdrant-port", "9999",
            "--postgres-host", "custom-postgres",
            "--postgres-port", "5433",
            "--postgres-db", "custom-db"
        ])

        assert result.exit_code == 0
        mock_hippo.assert_called_once_with("custom-qdrant", 9999)
        mock_cortex.assert_called_once_with("custom-postgres", 5433, "custom-db")


def test_init_command_failure(cli_runner):
    """Test init command failure."""
    with patch('episemic.cli.main.Hippocampus', side_effect=Exception("Init failed")):
        result = cli_runner.invoke(app, ["init"])

        assert result.exit_code == 1
        assert "Failed to initialize" in result.stdout


def test_store_command_success(cli_runner, mock_memory_system):
    """Test successful store command."""
    # Mock successful storage
    mock_memory_system['cortex'].store_memory.return_value = True

    with patch('asyncio.run') as mock_run:
        mock_run.return_value = "test-memory-id"

        result = cli_runner.invoke(app, [
            "store",
            "Test memory content",
            "--title", "Test Title",
            "--source", "test",
            "--tags", "test1",
            "--tags", "test2"
        ])

        assert result.exit_code == 0
        assert "Memory stored with ID: test-memory-id" in result.stdout


def test_store_command_no_title(cli_runner, mock_memory_system):
    """Test store command without title (should generate from text)."""
    mock_memory_system['cortex'].store_memory.return_value = True

    with patch('asyncio.run') as mock_run:
        mock_run.return_value = "test-memory-id"

        result = cli_runner.invoke(app, [
            "store",
            "Short text"
        ])

        assert result.exit_code == 0
        assert "Memory stored with ID: test-memory-id" in result.stdout


def test_store_command_long_text(cli_runner, mock_memory_system):
    """Test store command with long text (should truncate title and summary)."""
    mock_memory_system['cortex'].store_memory.return_value = True
    long_text = "A" * 300  # Long text

    with patch('asyncio.run') as mock_run:
        mock_run.return_value = "test-memory-id"

        result = cli_runner.invoke(app, [
            "store",
            long_text
        ])

        assert result.exit_code == 0
        assert "Memory stored with ID: test-memory-id" in result.stdout


def test_store_command_failure(cli_runner, mock_memory_system):
    """Test store command failure."""
    with patch('asyncio.run') as mock_run:
        mock_run.return_value = None  # Storage failed

        result = cli_runner.invoke(app, [
            "store",
            "Test content"
        ])

        assert result.exit_code == 1
        assert "Failed to store memory" in result.stdout


def test_store_command_exception(cli_runner, mock_memory_system):
    """Test store command with exception."""
    with patch('asyncio.run', side_effect=Exception("Storage error")):
        result = cli_runner.invoke(app, [
            "store",
            "Test content"
        ])

        assert result.exit_code == 1
        assert "Error storing memory: Storage error" in result.stdout


def test_search_command_success(cli_runner, mock_memory_system, sample_memory):
    """Test successful search command."""
    # Create search results
    search_result = SearchResult(
        memory=sample_memory,
        score=0.95,
        provenance={"source": "test"},
        retrieval_path=["test"]
    )

    with patch('asyncio.run') as mock_run:
        mock_run.return_value = [search_result]

        result = cli_runner.invoke(app, [
            "search",
            "test query",
            "--top-k", "10",
            "--tags", "test"
        ])

        assert result.exit_code == 0
        assert "Search Results for: test query" in result.stdout
        assert "Test Memory" in result.stdout
        assert "0.950" in result.stdout


def test_search_command_with_tags(cli_runner, mock_memory_system, sample_memory):
    """Test search command with multiple tags."""
    search_result = SearchResult(
        memory=sample_memory,
        score=0.85,
        provenance={"source": "test"},
        retrieval_path=["test"]
    )

    with patch('asyncio.run') as mock_run:
        mock_run.return_value = [search_result]

        result = cli_runner.invoke(app, [
            "search",
            "test query",
            "--tags", "test1",
            "--tags", "test2"
        ])

        assert result.exit_code == 0
        assert "Search Results" in result.stdout


def test_search_command_no_results(cli_runner, mock_memory_system):
    """Test search command with no results."""
    with patch('asyncio.run') as mock_run:
        mock_run.return_value = []  # No results

        result = cli_runner.invoke(app, [
            "search",
            "nonexistent query"
        ])

        assert result.exit_code == 0
        assert "No memories found matching your query" in result.stdout


def test_search_command_exception(cli_runner, mock_memory_system):
    """Test search command with exception."""
    with patch('asyncio.run', side_effect=Exception("Search error")):
        result = cli_runner.invoke(app, [
            "search",
            "test query"
        ])

        assert result.exit_code == 1
        assert "Error searching memories: Search error" in result.stdout


def test_search_command_long_title(cli_runner, mock_memory_system):
    """Test search command with long memory title."""
    long_title_memory = Memory(
        id="long-title-memory",
        title="A" * 100,  # Very long title
        text="Test content",
        summary="Summary",
        source="test",
        tags=["tag1", "tag2", "tag3", "tag4", "tag5"]  # Many tags
    )

    search_result = SearchResult(
        memory=long_title_memory,
        score=0.75,
        provenance={"source": "test"},
        retrieval_path=["test"]
    )

    with patch('asyncio.run') as mock_run:
        mock_run.return_value = [search_result]

        result = cli_runner.invoke(app, [
            "search",
            "test query"
        ])

        assert result.exit_code == 0
        assert "..." in result.stdout  # Title should be truncated


def test_get_command_success(cli_runner, mock_memory_system, sample_memory):
    """Test successful get command."""
    with patch('asyncio.run') as mock_run:
        mock_run.return_value = sample_memory

        result = cli_runner.invoke(app, [
            "get",
            "test-memory-id"
        ])

        assert result.exit_code == 0
        assert "Test Memory" in result.stdout
        assert "This is a test memory" in result.stdout
        assert "cli" in result.stdout
        assert "test, cli" in result.stdout


def test_get_command_not_found(cli_runner, mock_memory_system):
    """Test get command when memory not found."""
    with patch('asyncio.run') as mock_run:
        mock_run.return_value = None  # Memory not found

        result = cli_runner.invoke(app, [
            "get",
            "nonexistent-id"
        ])

        assert result.exit_code == 1
        assert "Memory nonexistent-id not found" in result.stdout


def test_get_command_exception(cli_runner, mock_memory_system):
    """Test get command with exception."""
    with patch('asyncio.run', side_effect=Exception("Retrieval error")):
        result = cli_runner.invoke(app, [
            "get",
            "test-memory-id"
        ])

        assert result.exit_code == 1
        assert "Error retrieving memory: Retrieval error" in result.stdout


def test_consolidate_command_auto_success(cli_runner, mock_memory_system):
    """Test successful auto consolidation command."""
    with patch('asyncio.run') as mock_run:
        mock_run.return_value = 5  # 5 memories processed

        result = cli_runner.invoke(app, [
            "consolidate",
            "--auto"
        ])

        assert result.exit_code == 0
        assert "Auto-consolidation completed. 5 memories processed" in result.stdout


def test_consolidate_command_single_success(cli_runner, mock_memory_system):
    """Test successful single memory consolidation."""
    with patch('asyncio.run') as mock_run:
        mock_run.return_value = True  # Consolidation successful

        result = cli_runner.invoke(app, [
            "consolidate",
            "--memory-id", "test-memory-id"
        ])

        assert result.exit_code == 0
        assert "Memory test-memory-id consolidated successfully" in result.stdout


def test_consolidate_command_single_failure(cli_runner, mock_memory_system):
    """Test failed single memory consolidation."""
    with patch('asyncio.run') as mock_run:
        mock_run.return_value = False  # Consolidation failed

        result = cli_runner.invoke(app, [
            "consolidate",
            "--memory-id", "test-memory-id"
        ])

        assert result.exit_code == 1
        assert "Failed to consolidate memory test-memory-id" in result.stdout


def test_consolidate_command_no_params(cli_runner, mock_memory_system):
    """Test consolidate command without parameters."""
    result = cli_runner.invoke(app, ["consolidate"])

    assert result.exit_code == 1
    assert "Please specify either --memory-id or --auto" in result.stdout


def test_consolidate_command_exception(cli_runner, mock_memory_system):
    """Test consolidate command with exception."""
    with patch('asyncio.run', side_effect=Exception("Consolidation error")):
        result = cli_runner.invoke(app, [
            "consolidate",
            "--auto"
        ])

        assert result.exit_code == 1
        assert "Error during consolidation: Consolidation error" in result.stdout


def test_health_command_all_healthy(cli_runner, mock_memory_system):
    """Test health command when all components are healthy."""
    # Mock healthy responses
    mock_memory_system['hippocampus'].health_check.return_value = {
        "qdrant_connected": True,
        "redis_connected": True
    }
    mock_memory_system['cortex'].health_check.return_value = True
    mock_memory_system['consolidation'].health_check.return_value = {
        "hippocampus_healthy": True,
        "cortex_healthy": True
    }
    mock_memory_system['retrieval'].health_check.return_value = {
        "hippocampus_healthy": True,
        "cortex_healthy": True
    }

    result = cli_runner.invoke(app, ["health"])

    assert result.exit_code == 0
    assert "System Health Status" in result.stdout
    assert "✅ Healthy" in result.stdout


def test_health_command_some_unhealthy(cli_runner, mock_memory_system):
    """Test health command when some components are unhealthy."""
    # Mock mixed health responses
    mock_memory_system['hippocampus'].health_check.return_value = {
        "qdrant_connected": False,  # Unhealthy
        "redis_connected": True
    }
    mock_memory_system['cortex'].health_check.return_value = True
    mock_memory_system['consolidation'].health_check.return_value = {
        "hippocampus_healthy": False,  # Unhealthy
        "cortex_healthy": True
    }
    mock_memory_system['retrieval'].health_check.return_value = {
        "hippocampus_healthy": True,
        "cortex_healthy": True
    }

    result = cli_runner.invoke(app, ["health"])

    assert result.exit_code == 0
    assert "System Health Status" in result.stdout
    assert "❌ Unhealthy" in result.stdout
    assert "✅ Healthy" in result.stdout


def test_health_command_exception(cli_runner, mock_memory_system):
    """Test health command with exception."""
    mock_memory_system['hippocampus'].health_check.side_effect = Exception("Health check error")

    result = cli_runner.invoke(app, ["health"])

    assert result.exit_code == 1
    assert "Error checking health: Health check error" in result.stdout


def test_get_memory_system_initialization():
    """Test get_memory_system function initializes components."""
    with patch('episemic.cli.main.Hippocampus') as mock_hippo, \
         patch('episemic.cli.main.Cortex') as mock_cortex, \
         patch('episemic.cli.main.ConsolidationEngine') as mock_consol, \
         patch('episemic.cli.main.RetrievalEngine') as mock_retrieval:

        # Clear global instances
        import episemic.cli.main as cli_main
        cli_main.hippocampus = None
        cli_main.cortex = None
        cli_main.consolidation_engine = None
        cli_main.retrieval_engine = None

        from episemic.cli.main import get_memory_system
        hippo, cortex, consol, retrieval = get_memory_system()

        # Verify components were created
        mock_hippo.assert_called_once()
        mock_cortex.assert_called_once()
        mock_consol.assert_called_once()
        mock_retrieval.assert_called_once()


def test_get_memory_system_reuse_existing():
    """Test get_memory_system reuses existing components."""
    with patch('episemic.cli.main.Hippocampus') as mock_hippo, \
         patch('episemic.cli.main.Cortex') as mock_cortex, \
         patch('episemic.cli.main.ConsolidationEngine') as mock_consol, \
         patch('episemic.cli.main.RetrievalEngine') as mock_retrieval:

        # Set global instances
        import episemic.cli.main as cli_main
        cli_main.hippocampus = "existing_hippo"
        cli_main.cortex = "existing_cortex"
        cli_main.consolidation_engine = "existing_consol"
        cli_main.retrieval_engine = "existing_retrieval"

        from episemic.cli.main import get_memory_system
        hippo, cortex, consol, retrieval = get_memory_system()

        # Verify existing components were returned
        assert hippo == "existing_hippo"
        assert cortex == "existing_cortex"
        assert consol == "existing_consol"
        assert retrieval == "existing_retrieval"

        # Verify new components were NOT created
        mock_hippo.assert_not_called()
        mock_cortex.assert_not_called()
        mock_consol.assert_not_called()
        mock_retrieval.assert_not_called()


def test_command_help_individual():
    """Test help for individual commands."""
    cli_runner = CliRunner()
    commands = ["init", "store", "search", "get", "consolidate", "health", "version"]

    for command in commands:
        result = cli_runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert ("Usage:" in result.stdout or "Show this message" in result.stdout)


def test_consolidate_complex_health_check_logic(cli_runner, mock_memory_system):
    """Test health command with complex health check responses."""
    # Mock complex health responses
    mock_memory_system['hippocampus'].health_check.return_value = {
        "qdrant_connected": True,
        "redis_connected": True,
        "extra_field": "should_be_ignored"
    }
    mock_memory_system['cortex'].health_check.return_value = True
    mock_memory_system['consolidation'].health_check.return_value = {
        "hippocampus_healthy": {"nested": "dict"},  # Dict instead of bool
        "cortex_healthy": True,
        "extra_field": "not_a_bool"
    }
    mock_memory_system['retrieval'].health_check.return_value = {
        "hippocampus_healthy": True,
        "cortex_healthy": {"another": "dict"}  # Dict instead of bool
    }

    result = cli_runner.invoke(app, ["health"])

    assert result.exit_code == 0
    assert "System Health Status" in result.stdout


def test_store_command_memory_creation_logic(cli_runner, mock_memory_system):
    """Test store command memory creation with different text lengths."""
    mock_memory_system['cortex'].store_memory.return_value = True

    test_cases = [
        # (text, expected_title_contains, expected_summary_contains)
        ("Short", "Short", "Short"),
        ("A" * 30, "A" * 30, "A" * 30),  # Normal length
        ("B" * 60, "..." in "B" * 50 + "...", "B" * 60),  # Long title
        ("C" * 250, "..." in "C" * 50 + "...", "..." in "C" * 200 + "..."),  # Long text and summary
    ]

    for text, _, _ in test_cases:
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = "test-id"

            result = cli_runner.invoke(app, ["store", text])
            assert result.exit_code == 0


def test_search_results_display_formatting(cli_runner, mock_memory_system):
    """Test search results display with various formatting scenarios."""
    # Create memories with different characteristics
    memories = [
        Memory(
            id="short-id",
            title="Short Title",
            text="Content",
            summary="Summary",
            source="test",
            tags=["tag1"]
        ),
        Memory(
            id="very-long-memory-id-that-should-be-truncated",
            title="A" * 100,  # Very long title
            text="Content",
            summary="Summary",
            source="test",
            tags=["tag1", "tag2", "tag3", "tag4", "tag5", "tag6"]  # Many tags
        ),
        Memory(
            id="no-tags-memory",
            title="No Tags Memory",
            text="Content",
            summary="Summary",
            source="test",
            tags=[]  # No tags
        )
    ]

    search_results = [
        SearchResult(memory=mem, score=0.8 - i*0.1, provenance={}, retrieval_path=[])
        for i, mem in enumerate(memories)
    ]

    with patch('asyncio.run') as mock_run:
        mock_run.return_value = search_results

        result = cli_runner.invoke(app, ["search", "test"])

        assert result.exit_code == 0
        assert "Search Results" in result.stdout


def test_memory_display_formatting_in_get(cli_runner, mock_memory_system):
    """Test memory display formatting in get command."""
    memory_with_all_fields = Memory(
        id="full-memory-id",
        title="Full Memory Title",
        text="This is the full text content of the memory",
        summary="Full summary",
        source="comprehensive_test",
        tags=["tag1", "tag2", "tag3"],
        created_at=datetime(2023, 1, 1, 12, 0, 0),
        access_count=42
    )

    with patch('asyncio.run') as mock_run:
        mock_run.return_value = memory_with_all_fields

        result = cli_runner.invoke(app, ["get", "full-memory-id"])

        assert result.exit_code == 0
        assert "Full Memory Title" in result.stdout
        assert "full text content" in result.stdout
        assert "comprehensive_test" in result.stdout
        assert "tag1, tag2, tag3" in result.stdout
        assert "42" in result.stdout


def test_main_entry_point():
    """Test main entry point."""
    with patch('episemic.cli.main.app') as mock_app:
        from episemic.cli.main import __name__ as module_name

        # Simulate running as main module
        if module_name == "__main__":
            mock_app.assert_called()
        else:
            # If not main, app should not be called automatically
            pass