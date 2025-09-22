"""
Simple examples of using Episemic.

This shows the super clean, easy-to-use API.
"""

import asyncio

# Import the simple API
from episemic import Episemic, EpistemicSync, create_memory_system


async def basic_example():
    """The simplest possible usage."""
    print("=== Basic Async Example ===")

    # Create and start memory system
    async with Episemic() as episemic:
        # Store some memories
        memory1 = await episemic.remember(
            "Python is a great programming language",
            title="About Python",
            tags=["programming", "python"]
        )

        memory2 = await episemic.remember(
            "Machine learning models need training data",
            title="ML Requirements",
            tags=["ml", "training"]
        )

        print(f"Stored: {memory1}")
        print(f"Stored: {memory2}")

        # Search for memories
        results = await episemic.recall("programming language")
        print(f"\nSearch results: {len(results)} found")

        for result in results:
            print(f"  - {result.memory.title} (score: {result.score:.3f})")

        # Get a specific memory
        if memory1:
            retrieved = await episemic.get(memory1.id)
            print(f"\nRetrieved: {retrieved.title}")

        # Find related memories
        if memory1:
            related = await episemic.find_related(memory1.id, limit=3)
            print(f"\nRelated memories: {len(related)} found")


async def custom_config_example():
    """Using custom configuration."""
    print("\n=== Custom Configuration Example ===")

    # Custom configuration (if you have different servers)
    episemic = Episemic(
        postgres_host="localhost",
        postgres_db="my_episemic_db",
        debug=True
    )

    await episemic.start()

    # Store with metadata
    memory = await episemic.remember(
        "Important research findings about neural networks",
        title="Neural Network Research",
        tags=["research", "ai", "neural-networks"],
        metadata={
            "source": "research_paper.pdf",
            "author": "Dr. Smith",
            "confidence": 0.95
        }
    )

    if memory:
        print(f"Stored with metadata: {memory.title}")
        print(f"Metadata: {memory.metadata}")

    # Check system health
    healthy = await episemic.health()
    print(f"System healthy: {healthy}")


def sync_example():
    """Using the synchronous API (for non-async code)."""
    print("\n=== Synchronous Example ===")

    # For code that can't use async/await
    episemic = EpistemicSync(debug=True)
    episemic.start()

    # Store memories
    memory = episemic.remember(
        "Synchronous programming is sometimes easier",
        title="Sync Programming",
        tags=["programming", "sync"]
    )

    print(f"Stored synchronously: {memory}")

    # Search memories
    results = episemic.recall("programming")
    print(f"Found {len(results)} memories")

    # Get system health
    healthy = episemic.health()
    print(f"System healthy: {healthy}")


async def quick_setup_example():
    """Using the quick setup function."""
    print("\n=== Quick Setup Example ===")

    # One-line setup
    episemic = await create_memory_system(debug=True)

    # Immediately start using it
    memory = await episemic.remember(
        "Quick setup makes it easy to get started",
        tags=["setup", "easy"]
    )

    results = await episemic.recall("easy")
    print(f"Quick setup found {len(results)} memories")


async def real_world_example():
    """A more realistic example with error handling."""
    print("\n=== Real World Example ===")

    try:
        async with Episemic() as episemic:
            # Store various types of information
            memories = []

            # Store a note
            note = await episemic.remember(
                "Remember to follow up with the client about the proposal",
                title="Client Follow-up",
                tags=["work", "todo", "client"]
            )
            if note:
                memories.append(note)

            # Store a learning
            learning = await episemic.remember(
                "FastAPI is great for building APIs quickly in Python",
                title="FastAPI Learning",
                tags=["programming", "api", "python"],
                metadata={"learned_from": "tutorial", "difficulty": "easy"}
            )
            if learning:
                memories.append(learning)

            # Store a fact
            fact = await episemic.remember(
                "The human brain has approximately 86 billion neurons",
                title="Brain Facts",
                tags=["science", "neuroscience", "facts"]
            )
            if fact:
                memories.append(fact)

            print(f"Stored {len(memories)} memories")

            # Search for work-related items
            work_items = await episemic.recall("work", tags=["work"])
            print(f"\nWork items: {len(work_items)}")

            # Search for programming knowledge
            programming = await episemic.recall("programming", tags=["programming"])
            print(f"Programming knowledge: {len(programming)}")

            # Find things related to the brain/AI
            ai_related = await episemic.recall("brain neural")
            print(f"AI/Brain related: {len(ai_related)}")

            # Demonstrate memory relationships
            if fact:
                related_to_fact = await episemic.find_related(fact.id)
                print(f"Related to brain fact: {len(related_to_fact)}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your database services are running!")


async def main():
    """Run all examples."""
    print("Episemic - Simple Memory System Examples")
    print("=" * 40)

    examples = [
        basic_example,
        custom_config_example,
        quick_setup_example,
        real_world_example,
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Example failed (services may not be running): {e}")

    # Run sync example separately
    try:
        sync_example()
    except Exception as e:
        print(f"Sync example failed: {e}")

    print("\n=== All examples completed ===")


if __name__ == "__main__":
    asyncio.run(main())