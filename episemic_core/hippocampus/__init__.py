"""Hippocampus module - Fast memory storage and retrieval."""

try:
    from .hippocampus import Hippocampus
    __all__ = ["Hippocampus"]
except ImportError:
    # Qdrant/Redis dependencies not available
    Hippocampus = None
    __all__ = []
