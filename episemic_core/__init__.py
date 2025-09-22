"""Episemic - Simple memory system for AI agents."""

__version__ = "0.1.2"
__author__ = "Aditya Karnam"
__email__ = "akarnam37@gmail.com"

# Simple, user-friendly API
from .simple import Episemic, EpistemicSync, Memory, SearchResult, create_memory_system

__all__ = [
    "Episemic",          # Main async interface
    "EpistemicSync",     # Sync interface for non-async code
    "Memory",            # Memory object
    "SearchResult",      # Search result with memory + score
    "create_memory_system",  # Quick setup function
]
