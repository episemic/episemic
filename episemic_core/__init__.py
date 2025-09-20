"""Episemic Core - A brain-inspired memory system for AI agents."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .consolidation import ConsolidationEngine
from .cortex import Cortex
from .hippocampus import Hippocampus
from .retrieval import RetrievalEngine

__all__ = [
    "Hippocampus",
    "Cortex",
    "ConsolidationEngine",
    "RetrievalEngine",
]
