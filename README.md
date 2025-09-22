# Episemic Core 🧠

**Episemic Core** is a brain-inspired memory system for AI agents that enables intelligent **encoding, storage, consolidation, and retrieval** of information. Inspired by human cognition, Episemic combines **episodic and semantic memory**, **replay-based consolidation**, and **associative retrieval** to create context-aware AI systems.

[![PyPI version](https://badge.fury.io/py/episemic-core.svg)](https://badge.fury.io/py/episemic-core)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Features

- **🧠 Brain-Inspired Architecture**
  - Episodic Memory (Hippocampus-like): High-fidelity experiences
  - Semantic Memory (Cortex-like): Consolidated, structured knowledge
- **🔄 Replay & Consolidation**
  - Prioritized experience sampling
  - Distillation from episodic → semantic memory
- **🎯 Associative Retrieval**
  - Vector similarity search
  - Tag-based filtering
  - Graph-based contextual relationships
- **⚡ Zero-Config Setup**
  - Works out of the box with DuckDB
  - No external dependencies required
  - Scales to production with Qdrant + PostgreSQL

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install episemic-core
```

### From Test PyPI (Latest Development)

```bash
pip install --index-url https://test.pypi.org/simple/ episemic-core
```

### Requirements

- **Python 3.11 or higher**
- **No external databases required** (uses DuckDB by default)

---

## 🚀 Quick Start

### Simple Python API

```python
import asyncio
from episemic_core import Episemic

async def main():
    # Create memory system (no setup required!)
    async with Episemic() as episemic:
        # Store a memory
        memory = await episemic.remember(
            "Machine learning models need training data",
            title="ML Requirements",
            tags=["ml", "training", "data"]
        )

        # Search for memories
        results = await episemic.recall("training data")

        # Get specific memory
        retrieved = await episemic.get(memory.id)

        print(f"Stored: {memory.title}")
        print(f"Found {len(results)} related memories")

# Run the example
asyncio.run(main())
```

### Synchronous API (Non-async code)

```python
from episemic_core import EpistemicSync

# Initialize memory system
episemic = EpistemicSync()
episemic.start()

# Store memories
memory = episemic.remember(
    "Important information to remember",
    title="Key Insight",
    tags=["important", "work"]
)

# Search memories
results = episemic.recall("important information")

print(f"Found {len(results)} memories")
```

### Command Line Interface

```bash
# Install creates the 'episemic' command
pip install episemic-core

# Store a memory
episemic store "This is my first memory" --title "First Memory" --tags ai memory

# Search for memories
episemic search "memory" --top-k 5 --tags ai

# Retrieve a specific memory
episemic get <memory-id>

# Check system health
episemic health

# Show version
episemic version
```

---

## 💡 Key Features

### 🎯 Simple API
- **Just two main methods**: `remember()` to store, `recall()` to search
- **Automatic optimization** for fast retrieval and long-term storage
- **Rich metadata support** with tags, titles, and custom data

### ⚡ Zero-Config Setup
- **Works immediately** - no database setup required
- **DuckDB backend** provides local, file-based storage
- **Automatic fallback** to in-memory storage if needed

### 🔍 Intelligent Search
- **Vector similarity** using sentence transformers
- **Tag-based filtering** for precise results
- **Graph relationships** for contextual connections
- **Hybrid search** combining multiple strategies

### 🚀 Async & Sync Support
- **Native async/await** for modern Python applications
- **Synchronous wrapper** for traditional codebases
- **Context manager support** for automatic resource cleanup

---

## 📚 Examples

### Basic Usage

```python
from episemic_core import Episemic

async def basic_example():
    async with Episemic() as episemic:
        # Store various types of information
        await episemic.remember(
            "Python is a versatile programming language",
            title="About Python",
            tags=["programming", "python"]
        )

        await episemic.remember(
            "FastAPI is great for building APIs quickly",
            title="FastAPI Benefits",
            tags=["python", "api", "web"],
            metadata={"framework": "FastAPI", "type": "web"}
        )

        # Search with different strategies
        python_results = await episemic.recall("programming language")
        api_results = await episemic.recall("API", tags=["web"])

        print(f"Found {len(python_results)} programming-related memories")
        print(f"Found {len(api_results)} API-related memories")
```

### Advanced Configuration

```python
from episemic_core import Episemic, EpistemicConfig

# Custom configuration
config = EpistemicConfig(
    debug=True,
    duckdb={"db_path": "./my_memories.db"}
)

async with Episemic(config=config) as episemic:
    # Store with rich metadata
    memory = await episemic.remember(
        "Neural networks require careful hyperparameter tuning",
        title="ML Best Practices",
        tags=["ml", "neural-networks", "optimization"],
        metadata={
            "source": "research_paper.pdf",
            "confidence": 0.95,
            "author": "Dr. Smith"
        }
    )

    # Find related memories
    related = await episemic.find_related(memory.id, limit=5)
    print(f"Found {len(related)} related memories")
```

### CLI Examples

```bash
# Store a note with tags
episemic store "Remember to backup the database daily" \
  --title "Backup Reminder" \
  --tags maintenance backup database

# Search with filters
episemic search "backup" --tags maintenance --top-k 10

# Search for recent memories
episemic search "database" --limit 5

# Get system information
episemic health
```

---

## ⚙️ Configuration

### Storage Options

**Episemic** supports two storage modes:

#### 🦆 DuckDB (Default)
- **Zero setup** - works immediately after installation
- **Local file storage** with automatic creation
- **Perfect for development**, prototypes, and small applications
- **Built-in vector search** using sentence transformers

```python
# Uses DuckDB by default - no configuration needed
async with Episemic() as episemic:
    await episemic.remember("Hello world!")
```

#### ⚡ Production Setup (Qdrant + PostgreSQL)
- **High performance** vector search with Qdrant
- **Rich relational features** with PostgreSQL
- **Horizontal scaling** support
- **Production-ready** with high availability

```python
from episemic_core import Episemic, EpistemicConfig

config = EpistemicConfig(
    prefer_qdrant=True,
    qdrant={"host": "your-qdrant-host", "port": 6333},
    postgresql={"host": "your-postgres-host", "database": "episemic_prod"}
)

async with Episemic(config=config) as episemic:
    await episemic.remember("Production memory!")
```

### Environment Variables

```bash
# DuckDB configuration (default)
export DUCKDB_PATH="./memories.db"
export DUCKDB_MODEL="all-MiniLM-L12-v2"

# Qdrant configuration (optional)
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export EPISEMIC_PREFER_QDRANT="true"

# PostgreSQL configuration (optional)
export POSTGRES_HOST="localhost"
export POSTGRES_DB="episemic"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="your-password"

# Debug mode
export EPISEMIC_DEBUG="true"
```

---

## 📖 Documentation

- **[API Documentation](docs/index.html)** - Complete API reference
- **[Examples](examples/)** - Working code examples
- **[Contributing Guide](CONTRIBUTING.md)** - Development setup and guidelines

---

## 🛠️ CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `store` | Store a new memory | `episemic store "Text content" --title "Title" --tags tag1 tag2` |
| `search`/`recall` | Search for memories | `episemic search "query" --top-k 10 --tags ai` |
| `get` | Retrieve memory by ID | `episemic get abc123...` |
| `health` | Check system health | `episemic health` |
| `version` | Show version info | `episemic version` |

---

## 🔧 Development

Want to contribute or run from source? See our **[Contributing Guide](CONTRIBUTING.md)** for:

- Development environment setup
- Running tests and quality checks
- Project structure and architecture
- Contribution workflow

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Episemic Core is inspired by research in cognitive science and neuroscience, particularly:
- **Hippocampal-cortical memory systems**
- **Replay-based memory consolidation**
- **Associative memory networks**

---

## 📞 Support

- **📋 Issues**: [GitHub Issues](https://github.com/episemic/episemic-core/issues)
- **📖 Documentation**: [API Docs](docs/index.html)
- **💬 Discussions**: [GitHub Discussions](https://github.com/episemic/episemic-core/discussions)

---

**Get started with intelligent memory for your AI agents today!**

```bash
pip install episemic-core
```