# Using Episemic as a Python Library

Episemic can be used both as a CLI tool and as a Python library. This guide shows you how to integrate it into your Python applications.

## Quick Start

### Basic Usage

```python
import asyncio
from episemic import EpistemicAPI

async def main():
    # Initialize the memory system
    api = EpistemicAPI()
    await api.initialize()

    # Store a memory
    memory_id = await api.store_memory(
        "Machine learning models need lots of training data",
        title="ML Training Requirements",
        tags=["ml", "data", "training"]
    )

    # Search for memories
    results = await api.search("training data", top_k=5)

    # Get a specific memory
    memory = await api.get_memory(memory_id)

    print(f"Found {len(results)} relevant memories")

asyncio.run(main())
```

### Using Context Manager (Recommended)

```python
import asyncio
from episemic import EpistemicAPI

async def main():
    async with EpistemicAPI() as api:
        # Store memories
        await api.store_memory("Important information here")

        # Search
        results = await api.search("important", top_k=10)

        # System automatically cleans up when done

asyncio.run(main())
```

## Configuration

### Default Configuration

By default, Episemic Core expects:
- **Qdrant**: localhost:6333
- **PostgreSQL**: localhost:5432 (database: episemic, user: postgres)
- **Redis**: localhost:6379

### Custom Configuration

```python
from episemic import EpistemicAPI, EpistemicConfig

# Method 1: Using configuration object
config = EpistemicConfig(
    qdrant={"host": "my-qdrant-host", "port": 6333},
    postgresql={"host": "my-postgres-host", "database": "my_db"},
    redis={"host": "my-redis-host"},
    debug=True
)

api = EpistemicAPI(config)
await api.initialize()
```

### Environment Variables

```python
from episemic import create_config_from_env, EpistemicAPI

# Set environment variables:
# QDRANT_HOST=my-qdrant-host
# POSTGRES_HOST=my-postgres-host
# POSTGRES_PASSWORD=my-password
# EPISEMIC_DEBUG=true

config = create_config_from_env()
api = EpistemicAPI(config)
await api.initialize()
```

### Configuration Options

```python
from episemic import EpistemicConfig

config = EpistemicConfig(
    # Database connections
    qdrant={"host": "localhost", "port": 6333, "collection_name": "memories"},
    postgresql={"host": "localhost", "port": 5432, "database": "episemic"},
    redis={"host": "localhost", "port": 6379, "ttl": 3600},

    # Feature toggles
    enable_hippocampus=True,    # Fast vector storage
    enable_cortex=True,         # Persistent relational storage
    enable_consolidation=True,  # Memory consolidation
    enable_retrieval=True,      # Multi-path retrieval

    # Consolidation settings
    consolidation={
        "threshold_hours": 2,           # Consolidate memories older than 2 hours
        "access_threshold": 3,          # Consolidate frequently accessed memories
        "auto_consolidation_enabled": True
    },

    # Debug settings
    debug=False,
    log_level="INFO"
)
```

## API Reference

### Core Methods

#### `store_memory()`
```python
memory_id = await api.store_memory(
    text="Your memory content",
    title="Optional title",
    source="your_app",
    tags=["tag1", "tag2"],
    embedding=None,  # Optional pre-computed embedding
    metadata={"key": "value"},  # Optional metadata
    store_in_hippocampus=True,  # Store in fast layer
    store_in_cortex=True        # Store in persistent layer
)
```

#### `search()`
```python
results = await api.search(
    query="your search query",
    top_k=10,
    tags=["filter_by_tag"],  # Optional tag filtering
    include_quarantined=False,
    embedding=None  # Optional pre-computed query embedding
)

# Each result has:
# - result.memory: The Memory object
# - result.score: Relevance score (0.0 to 1.0)
# - result.provenance: How it was found
# - result.retrieval_path: Search path taken
```

#### `get_memory()`
```python
memory = await api.get_memory(memory_id)
if memory:
    print(f"Title: {memory.title}")
    print(f"Text: {memory.text}")
    print(f"Tags: {memory.tags}")
    print(f"Created: {memory.created_at}")
```

#### `get_related_memories()`
```python
related = await api.get_related_memories(
    memory_id="some-memory-id",
    max_related=5
)
```

### Memory Consolidation

```python
# Manual consolidation
success = await api.consolidate_memory(memory_id)

# Auto consolidation sweep
processed_count = await api.run_auto_consolidation()
```

### Health Monitoring

```python
health = await api.health_check()
print(health)
# {
#   'hippocampus_qdrant_connected': True,
#   'hippocampus_redis_connected': True,
#   'cortex_healthy': True,
#   'consolidation_hippocampus_healthy': {...},
#   'consolidation_cortex_healthy': True,
#   'retrieval_hippocampus_healthy': {...},
#   'retrieval_cortex_healthy': True
# }
```

## Advanced Usage

### Custom Memory Storage

```python
# Store only in fast layer (temporary)
memory_id = await api.store_memory(
    "Temporary note",
    store_in_hippocampus=True,
    store_in_cortex=False
)

# Store only in persistent layer
memory_id = await api.store_memory(
    "Important permanent record",
    store_in_hippocampus=False,
    store_in_cortex=True
)
```

### Rich Metadata

```python
memory_id = await api.store_memory(
    "Research paper summary",
    title="Attention is All You Need",
    tags=["transformers", "attention", "nlp"],
    metadata={
        "authors": ["Vaswani", "Shazeer", "Parmar"],
        "year": 2017,
        "venue": "NIPS",
        "doi": "10.5555/3295222.3295349",
        "confidence": 0.95,
        "summary_type": "abstract"
    }
)
```

### Selective Component Initialization

```python
# Only enable specific components
config = EpistemicConfig(
    enable_hippocampus=True,   # Fast retrieval
    enable_cortex=False,       # Disable persistent storage
    enable_consolidation=False, # Disable consolidation
    enable_retrieval=True      # Keep retrieval
)

api = EpistemicAPI(config)
await api.initialize()
# Now only hippocampus + retrieval are active
```

### Integration with Embedding Models

```python
# If you have your own embedding model
def compute_embedding(text: str) -> List[float]:
    # Your embedding logic here
    return [0.1, 0.2, 0.3, ...]  # 768-dimensional vector

text = "Some important information"
embedding = compute_embedding(text)

# Store with pre-computed embedding
memory_id = await api.store_memory(
    text=text,
    embedding=embedding
)

# Search with pre-computed query embedding
query = "important information"
query_embedding = compute_embedding(query)
results = await api.search(
    query=query,
    embedding=query_embedding
)
```

## Error Handling

```python
from episemic import EpistemicAPI

async def robust_usage():
    api = EpistemicAPI()

    try:
        await api.initialize()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    try:
        memory_id = await api.store_memory("Test memory")
        if memory_id:
            print("Memory stored successfully")
        else:
            print("Failed to store memory")
    except Exception as e:
        print(f"Storage error: {e}")

    try:
        results = await api.search("test")
        print(f"Found {len(results)} memories")
    except Exception as e:
        print(f"Search error: {e}")
```

## Performance Considerations

### Batch Operations

```python
# Store multiple memories efficiently
memory_ids = []
for text, title, tags in your_data:
    memory_id = await api.store_memory(text, title=title, tags=tags)
    memory_ids.append(memory_id)

# Or use asyncio.gather for parallel storage
import asyncio

tasks = [
    api.store_memory(text, title=title, tags=tags)
    for text, title, tags in your_data
]
memory_ids = await asyncio.gather(*tasks)
```

### Memory Management

```python
# For long-running applications, periodically run consolidation
async def periodic_consolidation(api: EpistemicAPI):
    while True:
        count = await api.run_auto_consolidation()
        print(f"Consolidated {count} memories")
        await asyncio.sleep(3600)  # Run every hour

# Start background task
asyncio.create_task(periodic_consolidation(api))
```

## Examples

See the [`examples/library_usage.py`](../examples/library_usage.py) file for comprehensive examples of all features.

## Next Steps

- Check out the [CLI documentation](../README.md#cli-usage) for command-line usage
- Review the [architecture documentation](../BLUEPRINT.md) for system internals
- Explore the [API source code](../episemic/api.py) for detailed implementation