# Episemic Usage Guide

Episemic provides a simple, brain-inspired memory system for AI agents. You can use it both as a command-line tool and as a Python library.

## 🚀 Library Usage (Recommended)

### Super Simple API

```python
import asyncio
from episemic_core import Episemic

async def main():
    # Create and start memory system
    async with Episemic() as episemic:
        # Store memories
        memory = await episemic.remember(
            "Python is great for AI development",
            title="About Python",
            tags=["programming", "python", "ai"]
        )

        # Search memories
        results = await episemic.recall("python programming")

        # Get specific memory
        retrieved = await episemic.get(memory.id)

        print(f"Stored: {memory.title}")
        print(f"Found {len(results)} related memories")

asyncio.run(main())
```

### For Non-Async Code

```python
from episemic_core import EpistemicSync

# Synchronous API
episemic = EpistemicSync()
episemic.start()

# Store and search
memory = episemic.remember("Important information", tags=["work"])
results = episemic.recall("important")

print(f"Found {len(results)} memories")
```

### Configuration

```python
from episemic_core import Episemic

# Custom database settings
episemic = Episemic(
    postgres_host="your-postgres-host",
    postgres_db="your_database",
    postgres_user="your_user",
    postgres_password="your_password",
    qdrant_host="your-qdrant-host",
    redis_host="your-redis-host",
    debug=True
)

await episemic.start()
```

### Complete Example

```python
import asyncio
from episemic_core import Episemic

async def knowledge_base_example():
    async with Episemic() as episemic:
        # Store different types of knowledge

        # Technical knowledge
        await episemic.remember(
            "FastAPI is a modern web framework for Python APIs",
            title="FastAPI Framework",
            tags=["python", "web", "api", "framework"],
            metadata={"category": "technical", "difficulty": "intermediate"}
        )

        # Project information
        await episemic.remember(
            "The Q3 project deadline is October 15th",
            title="Q3 Project Deadline",
            tags=["project", "deadline", "q3"],
            metadata={"priority": "high", "due_date": "2024-10-15"}
        )

        # Research findings
        await episemic.remember(
            "Studies show that retrieval practice improves long-term retention",
            title="Learning Research",
            tags=["research", "learning", "memory", "education"],
            metadata={"source": "cognitive_science_journal", "confidence": 0.9}
        )

        # Search for different types of information

        # Technical queries
        tech_results = await episemic.recall("python framework", tags=["python"])
        print(f"Technical results: {len(tech_results)}")

        # Project queries
        project_results = await episemic.recall("deadline", tags=["project"])
        print(f"Project results: {len(project_results)}")

        # Research queries
        research_results = await episemic.recall("memory learning", tags=["research"])
        print(f"Research results: {len(research_results)}")

        # Find related memories
        if tech_results:
            related = await episemic.find_related(tech_results[0].memory.id)
            print(f"Related to technical: {len(related)}")

asyncio.run(knowledge_base_example())
```

## 🖥️ Command Line Usage

### Installation

```bash
git clone <repository>
cd episemic
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install poetry
poetry install
```

### Basic Commands

```bash
# Initialize the system
episemic init

# Store memories
episemic store "Important information" --title "My Note" --tags work important

# Search memories
episemic search "important" --top-k 5 --tags work

# Get a specific memory
episemic get <memory-id>

# Run consolidation
episemic consolidate --auto

# Check system health
episemic health

# Show version
episemic version
```

## 🛠️ API Reference

### Main Class: `Episemic`

```python
class Episemic:
    def __init__(self, **config_kwargs)
    async def start(self) -> bool
    async def remember(self, text: str, title: str = None, tags: list = None, metadata: dict = None) -> Memory
    async def recall(self, query: str, limit: int = 10, tags: list = None) -> list[SearchResult]
    async def get(self, memory_id: str) -> Memory
    async def find_related(self, memory_id: str, limit: int = 5) -> list[SearchResult]
    async def forget(self, memory_id: str) -> bool
    async def consolidate(self) -> int
    async def health(self) -> bool
```

### Memory Object

```python
class Memory:
    @property
    def id(self) -> str           # Unique identifier
    @property
    def text(self) -> str         # Memory content
    @property
    def title(self) -> str        # Memory title
    @property
    def tags(self) -> list[str]   # Associated tags
    @property
    def created_at(self) -> str   # Creation timestamp (ISO format)
    @property
    def metadata(self) -> dict    # Additional data
```

### Search Result

```python
class SearchResult:
    memory: Memory    # The found memory
    score: float      # Relevance score (0.0 to 1.0)
```

## 🔧 Configuration Options

Pass these as keyword arguments to `Episemic()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `postgres_host` | "localhost" | PostgreSQL server host |
| `postgres_port` | 5432 | PostgreSQL server port |
| `postgres_db` | "episemic" | Database name |
| `postgres_user` | "postgres" | Database user |
| `postgres_password` | "postgres" | Database password |
| `qdrant_host` | "localhost" | Qdrant vector DB host |
| `qdrant_port` | 6333 | Qdrant vector DB port |
| `redis_host` | "localhost" | Redis cache host |
| `redis_port` | 6379 | Redis cache port |
| `debug` | False | Enable debug output |

## 🌟 Key Features

- **🎯 Simple API**: Just `remember()` and `recall()` - that's it!
- **🧠 Brain-Inspired**: Mimics how human memory works (fast + slow storage)
- **🔍 Smart Search**: Vector similarity + tags + graph relationships
- **⚡ Fast Retrieval**: Optimized for quick access to recent memories
- **💾 Durable Storage**: Long-term persistence with integrity checking
- **🔧 Easy Setup**: Minimal configuration required
- **🚀 Async & Sync**: Works in both async and traditional Python code
- **📊 Rich Metadata**: Store additional context with any memory
- **🏥 Self-Healing**: Automatic error detection and recovery

## 🎛️ Environment Variables

You can also configure using environment variables:

```bash
export POSTGRES_HOST=my-postgres-server
export POSTGRES_PASSWORD=my-password
export QDRANT_HOST=my-qdrant-server
export EPISEMIC_DEBUG=true
```

## 🚨 Error Handling

```python
async with Episemic() as episemic:
    try:
        memory = await episemic.remember("Important data")
        if memory:
            print(f"Stored: {memory.id}")
        else:
            print("Failed to store memory")

        results = await episemic.recall("important")
        print(f"Found {len(results)} memories")

    except Exception as e:
        print(f"Error: {e}")

    # Check if system is healthy
    if await episemic.health():
        print("System is working properly")
    else:
        print("Some services may be down")
```

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL (for persistent storage)
- Qdrant (for vector similarity search)
- Redis (for caching - optional but recommended)

## 🔗 See Also

- [`examples/simple_usage.py`](examples/simple_usage.py) - More detailed examples
- [`BLUEPRINT.md`](BLUEPRINT.md) - System architecture and design
- [`docs/LIBRARY_USAGE.md`](docs/LIBRARY_USAGE.md) - Advanced library usage (internal APIs)