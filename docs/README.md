# 📚 Episemic Core Documentation

Welcome to the Episemic Core documentation! This directory contains comprehensive API documentation generated using Python's `pydoc` tool.

## 🌐 Main Documentation

- **[📖 Documentation Index](index.html)** - Start here! Interactive HTML documentation with examples
- **[🎯 Usage Guide](../USAGE.md)** - Complete usage examples and configuration
- **[🧠 Architecture Blueprint](../BLUEPRINT.md)** - Brain-inspired system design

## 📋 Quick Navigation

### 👤 For Users (Simple API)
- [🧠 Episemic Class](api/episemic_core.simple.html#Episemic) - Main async interface
- [🔄 EpistemicSync Class](api/episemic_core.simple.html#EpistemicSync) - Sync interface
- [💭 Memory Object](api/episemic_core.simple.html#Memory) - Memory data structure
- [🔍 SearchResult](api/episemic_core.simple.html#SearchResult) - Search results

### ⚙️ For Developers (Internal API)
- [🔧 Configuration System](api/episemic_core.config.html) - Database and service config
- [🏗️ Data Models](api/episemic_core.models.html) - Internal Pydantic models
- [🎯 High-Level API](api/episemic_core.api.html) - EpistemicAPI class
- [💻 CLI Interface](api/episemic_core.cli.main.html) - Command-line tools

### 🧠 Core Components (Brain Architecture)
- [⚡ Hippocampus](api/episemic_core.hippocampus.hippocampus.html) - Fast vector storage
- [🏛️ Cortex](api/episemic_core.cortex.cortex.html) - Long-term relational storage
- [🔄 Consolidation](api/episemic_core.consolidation.consolidation.html) - Memory transfer engine
- [🎯 Retrieval](api/episemic_core.retrieval.retrieval.html) - Multi-path search system

## 🚀 Quick Start

```python
from episemic_core import Episemic

async with Episemic() as episemic:
    # Store a memory
    memory = await episemic.remember("Your text here", tags=["important"])

    # Search memories
    results = await episemic.recall("search query")

    print(f"Found {len(results)} memories")
```

## 🔄 Regenerating Documentation

To update the documentation after code changes:

```bash
# Using Make
make docs

# Or directly
python generate_docs.py

# View in browser
make docs-view
```

## 📁 Documentation Structure

```
docs/
├── index.html              # Main documentation portal
├── README.md               # This file
├── LIBRARY_USAGE.md        # Advanced library usage guide
├── api/                    # Generated API documentation
│   ├── episemic_core.html  # Main module
│   ├── episemic_core.simple.html  # User-facing API
│   ├── episemic_core.api.html     # Internal API
│   ├── episemic_core.config.html  # Configuration
│   ├── episemic_core.models.html  # Data models
│   └── [other modules...]
└── generate_docs.py        # Documentation generator script
```

## 🎯 What to Read First

1. **New Users**: Start with [index.html](index.html) and the simple API docs
2. **Integrators**: Check out [USAGE.md](../USAGE.md) for configuration examples
3. **Developers**: Review [BLUEPRINT.md](../BLUEPRINT.md) for architecture understanding
4. **Contributors**: Read the internal API documentation in the `api/` folder

## 📖 Documentation Features

- **🎨 Interactive HTML**: Rich, styled documentation with navigation
- **🔍 Search-friendly**: All classes and methods are documented
- **📱 Responsive**: Works on desktop and mobile
- **🔗 Cross-linked**: Easy navigation between related components
- **📝 Examples**: Code examples throughout
- **🎯 User-focused**: Separate docs for users vs. developers

## 🆘 Need Help?

- Check the [main README](../README.md) for installation and setup
- See [examples/](../examples/) for working code samples
- Read the [USAGE.md](../USAGE.md) guide for comprehensive examples
- Look at the API documentation for specific method details

---

*Documentation generated using Python's `pydoc` tool | Last updated: $(date)*