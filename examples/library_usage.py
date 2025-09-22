"""
Examples of using Episemic Core as a Python library.

This file demonstrates various ways to use Episemic Core programmatically.
"""

import asyncio
from typing import List

from episemic_core import Episemic, create_memory_system
from episemic_core.api import EpistemicAPI
from episemic_core.config import EpistemicConfig


async def basic_usage_example():
    """Basic usage with default configuration."""
    print("=== Basic Usage Example ===")

    # Method 1: Simple initialization
    api = EpistemicAPI()
    await api.initialize()

    # Store some memories
    memory_id1 = await api.store_memory(
        "Artificial Intelligence is transforming how we work and live.",
        title="AI Impact",
        tags=["ai", "technology", "future"],
        source="example"
    )

    memory_id2 = await api.store_memory(
        "Machine learning models require large amounts of training data.",
        title="ML Training",
        tags=["ml", "data", "training"],
        source="example"
    )

    print(f"Stored memories: {memory_id1}, {memory_id2}")

    # Search for memories
    results = await api.search("artificial intelligence", top_k=5)
    print(f"Search results: {len(results)} memories found")

    for result in results:
        print(f"  - {result.memory.title} (score: {result.score:.3f})")

    # Get a specific memory
    memory = await api.get_memory(memory_id1)
    if memory:
        print(f"Retrieved memory: {memory.title}")

    # Get related memories
    related = await api.get_related_memories(memory_id1)
    print(f"Related memories: {len(related)} found")


async def context_manager_example():
    """Using Episemic API as an async context manager."""
    print("\n=== Context Manager Example ===")

    async with EpistemicAPI() as api:
        # Store a memory
        memory_id = await api.store_memory(
            "Context managers provide clean resource management.",
            title="Python Context Managers",
            tags=["python", "programming", "patterns"]
        )

        # Search immediately
        results = await api.search("context manager", top_k=3)
        print(f"Found {len(results)} relevant memories")


async def custom_configuration_example():
    """Using custom configuration."""
    print("\n=== Custom Configuration Example ===")

    # Using configuration class
    config = EpistemicConfig(
        debug=True
    )
    # Note: For this example, we'll use DuckDB-only mode
    # which doesn't require external services

    api = EpistemicAPI(config)
    await api.initialize()

    # Check health with debug info
    health = await api.health_check()
    print(f"System health: {health}")


async def environment_config_example():
    """Using environment-based configuration."""
    print("\n=== Environment Configuration Example ===")

    # This will read from environment variables like:
    # QDRANT_HOST, POSTGRES_HOST, POSTGRES_PASSWORD, etc.
    config = EpistemicConfig.from_env()

    api = EpistemicAPI(config)
    # Note: This might fail if environment vars aren't set
    try:
        await api.initialize()
        print("Successfully initialized with environment config")
    except Exception as e:
        print(f"Failed to initialize with env config: {e}")


async def advanced_features_example():
    """Demonstrating advanced features."""
    print("\n=== Advanced Features Example ===")

    api = EpistemicAPI()
    await api.initialize()

    # Store memory with custom metadata
    memory_id = await api.store_memory(
        "Neural networks are inspired by biological brain structures.",
        title="Neural Networks",
        tags=["ai", "neuroscience", "deep-learning"],
        metadata={
            "author": "AI Researcher",
            "confidence": 0.95,
            "references": ["paper1.pdf", "paper2.pdf"]
        },
        # Control where to store
        store_in_hippocampus=True,
        store_in_cortex=True
    )

    # Search with tag filtering
    results = await api.search(
        "brain networks",
        top_k=10,
        tags=["neuroscience"]  # Only search memories with this tag
    )

    print(f"Tagged search found {len(results)} memories")

    # Manual consolidation
    success = await api.consolidate_memory(memory_id)
    print(f"Manual consolidation: {'Success' if success else 'Failed'}")

    # Auto consolidation sweep
    processed = await api.run_auto_consolidation()
    print(f"Auto consolidation processed {processed} memories")


async def bulk_operations_example():
    """Demonstrating bulk memory operations."""
    print("\n=== Bulk Operations Example ===")

    api = EpistemicAPI()
    await api.initialize()

    # Store multiple memories
    memory_topics = [
        ("Python is a versatile programming language", "Python Basics", ["python", "programming"]),
        ("React is a JavaScript library for building UIs", "React Framework", ["javascript", "react", "ui"]),
        ("Docker containers provide application isolation", "Docker Containers", ["docker", "devops", "containers"]),
        ("Kubernetes orchestrates containerized applications", "Kubernetes", ["kubernetes", "devops", "orchestration"]),
        ("GraphQL provides flexible API queries", "GraphQL APIs", ["graphql", "api", "database"]),
    ]

    memory_ids = []
    for text, title, tags in memory_topics:
        memory_id = await api.store_memory(text, title=title, tags=tags)
        memory_ids.append(memory_id)

    print(f"Stored {len(memory_ids)} memories")

    # Search for programming-related memories
    programming_results = await api.search("programming", top_k=10, tags=["programming"])
    print(f"Programming memories: {len(programming_results)}")

    # Search for DevOps-related memories
    devops_results = await api.search("deployment", top_k=10, tags=["devops"])
    print(f"DevOps memories: {len(devops_results)}")

    # Get related memories for the first one
    if memory_ids:
        related = await api.get_related_memories(memory_ids[0], max_related=3)
        print(f"Related to first memory: {len(related)} found")


async def convenience_function_example():
    """Using convenience functions for quick setup."""
    print("\n=== Convenience Functions Example ===")

    # Quick setup with default config
    api = await create_memory_system()

    # Store and search in one go
    await api.store_memory(
        "Episemic provides brain-inspired memory for AI agents",
        title="About Episemic",
        tags=["episemic", "memory", "ai"]
    )

    results = await api.search("episemic memory")
    print(f"Quick search found {len(results)} memories")


async def error_handling_example():
    """Demonstrating proper error handling."""
    print("\n=== Error Handling Example ===")

    config = EpistemicConfig(debug=True)
    api = EpistemicAPI(config)

    # Try to use API before initialization
    try:
        await api.store_memory("This will fail")
    except RuntimeError as e:
        print(f"Expected error: {e}")

    # Initialize properly
    await api.initialize()

    # Now it should work
    memory_id = await api.store_memory("This should work")
    print(f"After initialization: {memory_id is not None}")


async def main():
    """Run all examples."""
    print("Episemic Core Library Usage Examples")
    print("=====================================")

    examples = [
        basic_usage_example,
        context_manager_example,
        custom_configuration_example,
        environment_config_example,
        advanced_features_example,
        bulk_operations_example,
        convenience_function_example,
        error_handling_example,
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Example failed (this is expected if services aren't running): {e}")

    print("\n=== Examples completed ===")


if __name__ == "__main__":
    asyncio.run(main())