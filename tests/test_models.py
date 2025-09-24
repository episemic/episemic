"""Tests for Episemic Core models."""

from datetime import datetime

from episemic.models import LinkType, Memory, MemoryLink


def test_memory_creation():
    memory = Memory(
        title="Test Memory",
        text="This is a test memory",
        summary="Test summary",
        source="test",
        tags=["test", "memory"],
    )

    assert memory.title == "Test Memory"
    assert memory.text == "This is a test memory"
    assert memory.source == "test"
    assert memory.tags == ["test", "memory"]
    assert memory.access_count == 0
    assert memory.version == 1


def test_memory_hash_computation():
    memory = Memory(
        title="Test Memory", text="This is a test memory", summary="Test summary", source="test"
    )

    expected_hash = memory.compute_hash()
    assert memory.hash == expected_hash
    assert len(memory.hash) == 64  # SHA256 hex length


def test_memory_integrity_verification():
    memory = Memory(
        title="Test Memory", text="This is a test memory", summary="Test summary", source="test"
    )

    assert memory.verify_integrity() is True

    # Modify text without updating hash
    memory.text = "Modified text"
    assert memory.verify_integrity() is False


def test_memory_access_increment():
    memory = Memory(
        title="Test Memory", text="This is a test memory", summary="Test summary", source="test"
    )

    initial_count = memory.access_count
    initial_accessed = memory.last_accessed

    memory.increment_access()

    assert memory.access_count == initial_count + 1
    assert memory.last_accessed != initial_accessed
    assert isinstance(memory.last_accessed, datetime)


def test_memory_link():
    link = MemoryLink(target_id="test-target-id", type=LinkType.CITES, weight=0.8)

    assert link.target_id == "test-target-id"
    assert link.type == LinkType.CITES
    assert link.weight == 0.8
