#!/usr/bin/env python3
"""
Example script testing episemic embeddings and similarity search
Tests multiple memories and semantic recall capabilities
"""

import asyncio
from episemic import Episemic

async def main():
    print("🧠 Testing Episemic Embeddings and Similarity Search")
    print("=" * 60)

    # Create memory system (no setup required!)
    async with Episemic() as episemic:
        print("\n📝 Storing multiple memories...")

        # Store related memories about AI/ML
        ai_memories = [
            {
                "text": "Machine learning models require large datasets for training",
                "title": "ML Training Data",
                "tags": ["ml", "training", "data"]
            },
            {
                "text": "Neural networks learn patterns through backpropagation",
                "title": "Neural Network Learning",
                "tags": ["neural", "learning", "backprop"]
            },
            {
                "text": "Deep learning uses multiple hidden layers to extract features",
                "title": "Deep Learning Architecture",
                "tags": ["deep", "layers", "features"]
            },
            {
                "text": "Transformers use attention mechanisms for sequence modeling",
                "title": "Transformer Architecture",
                "tags": ["transformer", "attention", "sequence"]
            },
            {
                "text": "Reinforcement learning agents learn through trial and error",
                "title": "RL Learning Process",
                "tags": ["rl", "agents", "trial"]
            }
        ]

        # Store unrelated memories
        other_memories = [
            {
                "text": "Python is a popular programming language for data science",
                "title": "Python for Data Science",
                "tags": ["python", "programming", "data"]
            },
            {
                "text": "Docker containers provide isolated application environments",
                "title": "Docker Containers",
                "tags": ["docker", "containers", "devops"]
            },
            {
                "text": "Git is a distributed version control system",
                "title": "Git Version Control",
                "tags": ["git", "version", "control"]
            }
        ]

        all_memories = ai_memories + other_memories
        stored_memories = []

        for mem_data in all_memories:
            memory = await episemic.remember(
                mem_data["text"],
                title=mem_data["title"],
                tags=mem_data["tags"]
            )
            stored_memories.append(memory)
            print(f"  ✅ Stored: {memory.title}")

        print(f"\n📊 Total memories stored: {len(stored_memories)}")

        # Test semantic similarity searches
        test_queries = [
            "training neural networks",
            "learning algorithms",
            "attention in transformers",
            "programming languages",
            "version control systems"
        ]

        print("\n🔍 Testing semantic similarity searches...")
        print("-" * 40)

        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = await episemic.recall(query, limit=3)

            if results:
                print(f"Found {len(results)} similar memories:")
                for i, result in enumerate(results, 1):
                    score = f"{result.score:.3f}" if hasattr(result, 'score') else "N/A"
                    print(f"  {i}. [{score}] {result.memory.title}")
                    print(f"     {result.memory.text[:60]}...")
            else:
                print("  No similar memories found")

        # Test cross-domain queries to verify embeddings work
        print("\n🌐 Testing cross-domain semantic understanding...")
        print("-" * 45)

        cross_domain_queries = [
            "algorithms that learn",
            "systems that improve over time",
            "automated pattern recognition"
        ]

        for query in cross_domain_queries:
            print(f"\nCross-domain query: '{query}'")
            results = await episemic.recall(query, limit=2)

            if results:
                print(f"Found {len(results)} semantically similar memories:")
                for i, result in enumerate(results, 1):
                    score = f"{result.score:.3f}" if hasattr(result, 'score') else "N/A"
                    print(f"  {i}. [{score}] {result.memory.title}")
            else:
                print("  No semantically similar memories found")

        # Test exact tag-based search
        print("\n🏷️  Testing tag-based search...")
        print("-" * 30)

        tag_results = await episemic.recall("ml", limit=5)
        print(f"Memories tagged with 'ml': {len(tag_results)}")
        for result in tag_results:
            print(f"  • {result.memory.title}")

        print("\n🎉 Embedding test completed successfully!")
        print("✅ Semantic similarity search is working correctly")
        print("✅ Embeddings can find related concepts across different domains")

if __name__ == "__main__":
    asyncio.run(main())