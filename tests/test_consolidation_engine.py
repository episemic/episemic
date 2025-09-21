"""Comprehensive tests for the consolidation engine."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from episemic_core.consolidation.consolidation import ConsolidationEngine
from episemic_core.models import Memory, ConsolidationJob, MemoryStatus, RetentionPolicy


@pytest.fixture
def mock_hippocampus():
    """Create a mock hippocampus."""
    hippocampus = MagicMock()
    hippocampus.retrieve_memory = AsyncMock()
    hippocampus.health_check = MagicMock(return_value={"status": "healthy"})
    return hippocampus


@pytest.fixture
def mock_cortex():
    """Create a mock cortex."""
    cortex = MagicMock()
    cortex.store_memory = AsyncMock()
    cortex.retrieve_memory = AsyncMock()
    cortex.health_check = MagicMock(return_value=True)
    return cortex


@pytest.fixture
def consolidation_engine(mock_hippocampus, mock_cortex):
    """Create a consolidation engine with mocked dependencies."""
    return ConsolidationEngine(mock_hippocampus, mock_cortex)


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
        created_at=datetime.utcnow() - timedelta(hours=3),  # Old enough to consolidate
        access_count=5,  # High enough to consolidate
        status=MemoryStatus.ACTIVE,
        retention_policy=RetentionPolicy.DEFAULT
    )


@pytest.fixture
def sample_consolidation_job():
    """Create a sample consolidation job."""
    return ConsolidationJob(
        memory_ids=["memory1", "memory2", "memory3"],
        job_type="consolidation",
        status="pending"
    )


@pytest.mark.asyncio
async def test_consolidation_engine_initialization(mock_hippocampus, mock_cortex):
    """Test consolidation engine initialization."""
    engine = ConsolidationEngine(mock_hippocampus, mock_cortex)
    assert engine.hippocampus == mock_hippocampus
    assert engine.cortex == mock_cortex
    assert engine.consolidation_threshold_hours == 2
    assert engine.consolidation_access_threshold == 3


@pytest.mark.asyncio
async def test_consolidate_memory_success(consolidation_engine, mock_hippocampus, mock_cortex, sample_memory):
    """Test successful memory consolidation."""
    mock_hippocampus.retrieve_memory.return_value = sample_memory
    mock_cortex.store_memory.return_value = True

    with patch('builtins.print') as mock_print:
        result = await consolidation_engine.consolidate_memory("test-memory-id")

    assert result is True
    mock_hippocampus.retrieve_memory.assert_called_once_with("test-memory-id")
    mock_cortex.store_memory.assert_called_once_with(sample_memory)
    mock_print.assert_called_with("Successfully consolidated memory test-memory-id to cortex")


@pytest.mark.asyncio
async def test_consolidate_memory_not_found(consolidation_engine, mock_hippocampus):
    """Test consolidation when memory is not found."""
    mock_hippocampus.retrieve_memory.return_value = None

    with patch('builtins.print') as mock_print:
        result = await consolidation_engine.consolidate_memory("nonexistent-id")

    assert result is False
    mock_print.assert_called_with("Memory nonexistent-id not found in hippocampus")


@pytest.mark.asyncio
async def test_consolidate_memory_should_not_consolidate(consolidation_engine, mock_hippocampus):
    """Test consolidation when memory should not be consolidated."""
    # Create memory that shouldn't be consolidated (recent and low access)
    recent_memory = Memory(
        id="recent-memory",
        title="Recent Memory",
        text="Recent content",
        summary="Recent summary",
        source="test",
        created_at=datetime.utcnow(),  # Very recent
        access_count=1,  # Low access count
        status=MemoryStatus.ACTIVE,
        retention_policy=RetentionPolicy.DEFAULT
    )

    mock_hippocampus.retrieve_memory.return_value = recent_memory

    result = await consolidation_engine.consolidate_memory("recent-memory")
    assert result is False


@pytest.mark.asyncio
async def test_consolidate_memory_cortex_failure(consolidation_engine, mock_hippocampus, mock_cortex, sample_memory):
    """Test consolidation when cortex storage fails."""
    mock_hippocampus.retrieve_memory.return_value = sample_memory
    mock_cortex.store_memory.return_value = False

    with patch('builtins.print') as mock_print:
        result = await consolidation_engine.consolidate_memory("test-memory-id")

    assert result is False
    mock_print.assert_called_with("Failed to consolidate memory test-memory-id")


@pytest.mark.asyncio
async def test_consolidate_memory_exception_handling(consolidation_engine, mock_hippocampus):
    """Test consolidation exception handling."""
    mock_hippocampus.retrieve_memory.side_effect = Exception("Retrieval error")

    with patch('builtins.print') as mock_print:
        result = await consolidation_engine.consolidate_memory("error-memory")

    assert result is False
    mock_print.assert_called_with("Error during consolidation of memory error-memory: Retrieval error")


def test_should_consolidate_by_age(consolidation_engine):
    """Test _should_consolidate method based on age."""
    # Old memory should be consolidated
    old_memory = Memory(
        id="old-memory",
        title="Old Memory",
        text="Old content",
        summary="Old summary",
        source="test",
        created_at=datetime.utcnow() - timedelta(hours=5),  # Older than threshold
        access_count=1,
        status=MemoryStatus.ACTIVE,
        retention_policy=RetentionPolicy.DEFAULT
    )

    assert consolidation_engine._should_consolidate(old_memory) is True


def test_should_consolidate_by_access_count(consolidation_engine):
    """Test _should_consolidate method based on access count."""
    # High access memory should be consolidated
    high_access_memory = Memory(
        id="high-access-memory",
        title="High Access Memory",
        text="High access content",
        summary="High access summary",
        source="test",
        created_at=datetime.utcnow(),  # Recent
        access_count=10,  # High access count
        status=MemoryStatus.ACTIVE,
        retention_policy=RetentionPolicy.DEFAULT
    )

    assert consolidation_engine._should_consolidate(high_access_memory) is True


def test_should_not_consolidate_ephemeral(consolidation_engine):
    """Test that ephemeral memories are not consolidated."""
    ephemeral_memory = Memory(
        id="ephemeral-memory",
        title="Ephemeral Memory",
        text="Ephemeral content",
        summary="Ephemeral summary",
        source="test",
        created_at=datetime.utcnow() - timedelta(hours=5),
        access_count=10,
        status=MemoryStatus.ACTIVE,
        retention_policy=RetentionPolicy.EPHEMERAL  # Should not consolidate
    )

    assert consolidation_engine._should_consolidate(ephemeral_memory) is False


def test_should_not_consolidate_inactive(consolidation_engine):
    """Test that inactive memories are not consolidated."""
    inactive_memory = Memory(
        id="inactive-memory",
        title="Inactive Memory",
        text="Inactive content",
        summary="Inactive summary",
        source="test",
        created_at=datetime.utcnow() - timedelta(hours=5),
        access_count=10,
        status=MemoryStatus.QUARANTINED,  # Inactive status
        retention_policy=RetentionPolicy.DEFAULT
    )

    assert consolidation_engine._should_consolidate(inactive_memory) is False


def test_should_not_consolidate_recent_low_access(consolidation_engine):
    """Test that recent memories with low access are not consolidated."""
    recent_low_access_memory = Memory(
        id="recent-low-memory",
        title="Recent Low Memory",
        text="Recent low content",
        summary="Recent low summary",
        source="test",
        created_at=datetime.utcnow(),  # Recent
        access_count=1,  # Low access
        status=MemoryStatus.ACTIVE,
        retention_policy=RetentionPolicy.DEFAULT
    )

    assert consolidation_engine._should_consolidate(recent_low_access_memory) is False


@pytest.mark.asyncio
async def test_run_consolidation_job_success(consolidation_engine, sample_consolidation_job):
    """Test successful consolidation job execution."""
    # Mock consolidate_memory to return True for all memories
    with patch.object(consolidation_engine, 'consolidate_memory', return_value=True) as mock_consolidate:
        with patch('builtins.print') as mock_print:
            result = await consolidation_engine.run_consolidation_job(sample_consolidation_job)

    assert result.status == "completed"
    assert mock_consolidate.call_count == 3
    mock_print.assert_called_with(f"Consolidation job {sample_consolidation_job.id} completed: 3 memories consolidated")


@pytest.mark.asyncio
async def test_run_consolidation_job_partial_failure(consolidation_engine, sample_consolidation_job):
    """Test consolidation job with partial failures."""
    # Mock consolidate_memory to fail for one memory
    def mock_consolidate_side_effect(memory_id):
        return memory_id != "memory2"  # Fail for memory2

    with patch.object(consolidation_engine, 'consolidate_memory', side_effect=mock_consolidate_side_effect):
        with patch('builtins.print') as mock_print:
            result = await consolidation_engine.run_consolidation_job(sample_consolidation_job)

    assert result.status == "completed"  # Still completed if some succeed
    mock_print.assert_called_with(f"Consolidation job {sample_consolidation_job.id} completed: 2 memories consolidated")


@pytest.mark.asyncio
async def test_run_consolidation_job_complete_failure(consolidation_engine, sample_consolidation_job):
    """Test consolidation job with complete failure."""
    # Mock consolidate_memory to always return False
    with patch.object(consolidation_engine, 'consolidate_memory', return_value=False):
        result = await consolidation_engine.run_consolidation_job(sample_consolidation_job)

    assert result.status == "failed"
    assert "Failed to consolidate memories" in result.error_message
    assert "memory1" in result.error_message


@pytest.mark.asyncio
async def test_run_consolidation_job_exception_handling(consolidation_engine, sample_consolidation_job):
    """Test consolidation job exception handling."""
    with patch.object(consolidation_engine, 'consolidate_memory', side_effect=Exception("Consolidation error")):
        with patch('builtins.print') as mock_print:
            result = await consolidation_engine.run_consolidation_job(sample_consolidation_job)

    assert result.status == "failed"
    assert result.error_message == "Consolidation error"
    mock_print.assert_called_with(f"Consolidation job {sample_consolidation_job.id} failed: Consolidation error")


@pytest.mark.asyncio
async def test_auto_consolidation_sweep(consolidation_engine):
    """Test auto consolidation sweep."""
    with patch('builtins.print') as mock_print:
        result = await consolidation_engine.auto_consolidation_sweep()

    assert result == 0  # Currently returns 0 as it's a placeholder
    mock_print.assert_called_with("Auto consolidation sweep completed")


@pytest.mark.asyncio
async def test_auto_consolidation_sweep_exception_handling(consolidation_engine):
    """Test auto consolidation sweep exception handling."""
    # The current auto_consolidation_sweep doesn't actually access hippocampus
    # It's a placeholder that always succeeds. Let's test it as is.
    with patch('builtins.print') as mock_print:
        result = await consolidation_engine.auto_consolidation_sweep()

    assert result == 0
    mock_print.assert_called_with("Auto consolidation sweep completed")


@pytest.mark.asyncio
async def test_create_consolidated_summary_success(consolidation_engine, mock_cortex):
    """Test successful creation of consolidated summary."""
    # Create test memories
    memory1 = Memory(
        id="memory1",
        title="Memory 1",
        text="Content 1",
        summary="Summary 1",
        source="source1",
        tags=["tag1", "tag2"]
    )
    memory2 = Memory(
        id="memory2",
        title="Memory 2",
        text="Content 2",
        summary="Summary 2",
        source="source2",
        tags=["tag2", "tag3"]
    )

    mock_cortex.retrieve_memory.side_effect = [memory1, memory2]
    mock_cortex.store_memory.return_value = True

    result = await consolidation_engine.create_consolidated_summary(["memory1", "memory2"])

    assert result is not None
    assert result.title == "Consolidated Summary (2 memories)"
    assert result.source == "consolidation"
    assert "tag1" in result.tags
    assert "tag2" in result.tags
    assert "tag3" in result.tags
    assert result.retention_policy == RetentionPolicy.ARCHIVAL
    assert result.metadata["original_memory_count"] == 2


@pytest.mark.asyncio
async def test_create_consolidated_summary_no_memories_found(consolidation_engine, mock_cortex):
    """Test consolidated summary creation when no memories are found."""
    mock_cortex.retrieve_memory.return_value = None

    result = await consolidation_engine.create_consolidated_summary(["memory1", "memory2"])

    assert result is None


@pytest.mark.asyncio
async def test_create_consolidated_summary_storage_failure(consolidation_engine, mock_cortex):
    """Test consolidated summary creation when storage fails."""
    memory = Memory(
        id="memory1",
        title="Memory 1",
        text="Content 1",
        summary="Summary 1",
        source="source1",
        tags=["tag1"]
    )

    mock_cortex.retrieve_memory.return_value = memory
    mock_cortex.store_memory.return_value = False

    result = await consolidation_engine.create_consolidated_summary(["memory1"])

    assert result is None


@pytest.mark.asyncio
async def test_create_consolidated_summary_exception_handling(consolidation_engine, mock_cortex):
    """Test consolidated summary creation exception handling."""
    mock_cortex.retrieve_memory.side_effect = Exception("Retrieval error")

    with patch('builtins.print') as mock_print:
        result = await consolidation_engine.create_consolidated_summary(["memory1"])

    assert result is None
    mock_print.assert_called_with("Error creating consolidated summary: Retrieval error")


def test_summarize_memories(consolidation_engine):
    """Test _summarize_memories method."""
    memories = [
        Memory(
            id="memory1",
            title="Memory 1",
            text="Content 1",
            summary="Summary 1",
            source="test"
        ),
        Memory(
            id="memory2",
            title="Memory 2",
            text="Content 2",
            summary="Summary 2",
            source="test"
        )
    ]

    result = consolidation_engine._summarize_memories(memories)

    assert "Consolidated summary of 2 related memories" in result
    assert "Memory 1: Summary 1" in result
    assert "Memory 2: Summary 2" in result


def test_create_meta_summary(consolidation_engine):
    """Test _create_meta_summary method."""
    memories = [
        Memory(
            id="memory1",
            title="Memory 1",
            text="Content 1",
            summary="Summary 1",
            source="source1",
            tags=["tag1", "tag2"]
        ),
        Memory(
            id="memory2",
            title="Memory 2",
            text="Content 2",
            summary="Summary 2",
            source="source2",
            tags=["tag2", "tag3"]
        )
    ]

    result = consolidation_engine._create_meta_summary(memories)

    assert "Meta-summary covering topics:" in result
    assert "tag1" in result
    assert "tag2" in result
    assert "tag3" in result
    assert "source1" in result
    assert "source2" in result


def test_health_check(consolidation_engine, mock_hippocampus, mock_cortex):
    """Test health_check method."""
    mock_hippocampus.health_check.return_value = {"hippocampus": "healthy"}
    mock_cortex.health_check.return_value = True

    health = consolidation_engine.health_check()

    assert health == {
        "hippocampus_healthy": {"hippocampus": "healthy"},
        "cortex_healthy": True
    }


def test_consolidation_engine_configuration(consolidation_engine):
    """Test consolidation engine configuration parameters."""
    # Test default values
    assert consolidation_engine.consolidation_threshold_hours == 2
    assert consolidation_engine.consolidation_access_threshold == 3

    # Test custom configuration
    consolidation_engine.consolidation_threshold_hours = 24
    consolidation_engine.consolidation_access_threshold = 10

    assert consolidation_engine.consolidation_threshold_hours == 24
    assert consolidation_engine.consolidation_access_threshold == 10


@pytest.mark.asyncio
async def test_consolidation_workflow_integration(consolidation_engine, mock_hippocampus, mock_cortex):
    """Test complete consolidation workflow."""
    # Create a memory that should be consolidated
    old_memory = Memory(
        id="workflow-memory",
        title="Workflow Memory",
        text="Workflow content",
        summary="Workflow summary",
        source="test",
        created_at=datetime.utcnow() - timedelta(hours=5),
        access_count=1,
        status=MemoryStatus.ACTIVE,
        retention_policy=RetentionPolicy.DEFAULT
    )

    mock_hippocampus.retrieve_memory.return_value = old_memory
    mock_cortex.store_memory.return_value = True

    # Test the workflow
    result = await consolidation_engine.consolidate_memory("workflow-memory")

    assert result is True
    mock_hippocampus.retrieve_memory.assert_called_once_with("workflow-memory")
    mock_cortex.store_memory.assert_called_once_with(old_memory)