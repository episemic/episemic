"""Cortex module - Long-term memory storage and semantic relationships."""

try:
    from .cortex import Cortex
    __all__ = ["Cortex"]
except ImportError:
    # psycopg2 not available
    Cortex = None
    __all__ = []
