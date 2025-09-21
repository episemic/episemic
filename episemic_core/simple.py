"""Simple, user-friendly API for Episemic Core."""

import asyncio
from typing import Any

from .api import EpistemicAPI
from .config import EpistemicConfig


class Memory:
    """A simple memory object that users interact with."""

    def __init__(self, internal_memory):
        self._internal = internal_memory

    @property
    def id(self) -> str:
        """Unique memory identifier."""
        return self._internal.id

    @property
    def text(self) -> str:
        """The memory content."""
        return self._internal.text

    @property
    def title(self) -> str:
        """Memory title."""
        return self._internal.title

    @property
    def tags(self) -> list[str]:
        """Memory tags."""
        return self._internal.tags

    @property
    def created_at(self) -> str:
        """When the memory was created (ISO format)."""
        return self._internal.created_at.isoformat()

    @property
    def metadata(self) -> dict:
        """Additional memory metadata."""
        return self._internal.metadata

    def __str__(self) -> str:
        return f"Memory('{self.title}' - {len(self.text)} chars)"

    def __repr__(self) -> str:
        return f"Memory(id='{self.id[:8]}...', title='{self.title}')"


class SearchResult:
    """A search result with memory and relevance score."""

    def __init__(self, internal_result):
        self._internal = internal_result
        self.memory = Memory(internal_result.memory)
        self.score = internal_result.score

    def __str__(self) -> str:
        return f"SearchResult({self.memory.title}, score={self.score:.3f})"


class Episemic:
    """
    Simple memory system for AI agents.

    Example:
        >>> # Basic usage
        >>> episemic = Episemic()
        >>> await episemic.start()
        >>>
        >>> # Store a memory
        >>> memory = await episemic.remember("Important information", tags=["work"])
        >>>
        >>> # Search memories
        >>> results = await episemic.recall("important")
        >>>
        >>> # Get a specific memory
        >>> memory = await episemic.get(memory_id)
    """

    def __init__(self, **config_kwargs):
        """
        Initialize Episemic memory system.

        Args:
            **config_kwargs: Configuration options such as:
                - qdrant_host: Qdrant server host (default: "localhost")
                - qdrant_port: Qdrant server port (default: 6333)
                - postgres_host: PostgreSQL host (default: "localhost")
                - postgres_db: PostgreSQL database (default: "episemic")
                - postgres_user: PostgreSQL user (default: "postgres")
                - postgres_password: PostgreSQL password (default: "postgres")
                - redis_host: Redis host (default: "localhost")
                - redis_port: Redis port (default: 6379)
                - debug: Enable debug mode (default: False)
        """
        # Convert simple kwargs to config
        config_dict = {}

        if any(k.startswith('qdrant_') for k in config_kwargs):
            config_dict['qdrant'] = {}
            if 'qdrant_host' in config_kwargs:
                config_dict['qdrant']['host'] = config_kwargs.pop('qdrant_host')
            if 'qdrant_port' in config_kwargs:
                config_dict['qdrant']['port'] = config_kwargs.pop('qdrant_port')

        if any(k.startswith('postgres_') for k in config_kwargs):
            config_dict['postgresql'] = {}
            if 'postgres_host' in config_kwargs:
                config_dict['postgresql']['host'] = config_kwargs.pop('postgres_host')
            if 'postgres_db' in config_kwargs:
                config_dict['postgresql']['database'] = config_kwargs.pop('postgres_db')
            if 'postgres_user' in config_kwargs:
                config_dict['postgresql']['user'] = config_kwargs.pop('postgres_user')
            if 'postgres_password' in config_kwargs:
                config_dict['postgresql']['password'] = config_kwargs.pop('postgres_password')

        if any(k.startswith('redis_') for k in config_kwargs):
            config_dict['redis'] = {}
            if 'redis_host' in config_kwargs:
                config_dict['redis']['host'] = config_kwargs.pop('redis_host')
            if 'redis_port' in config_kwargs:
                config_dict['redis']['port'] = config_kwargs.pop('redis_port')

        # Add remaining kwargs directly
        config_dict.update(config_kwargs)

        self._config = EpistemicConfig(**config_dict) if config_dict else EpistemicConfig()
        self._api = EpistemicAPI(self._config)
        self._started = False

    async def start(self) -> bool:
        """
        Start the memory system.

        Returns:
            True if started successfully, False otherwise.
        """
        if self._started:
            return True

        try:
            success = await self._api.initialize()
            # Always mark as started if we get this far, even if some services fail
            self._started = True
            return success
        except Exception as e:
            if self._config.debug:
                print(f"Warning: Some services failed to start: {e}")
            # Still mark as started for basic functionality
            self._started = True
            return False

    async def remember(
        self,
        text: str,
        title: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Memory | None:
        """
        Store a new memory.

        Args:
            text: The content to remember
            title: Optional title for the memory
            tags: Optional list of tags
            metadata: Optional additional data

        Returns:
            Memory object if successful, None otherwise.
        """
        self._check_started()

        memory_id = await self._api.store_memory(
            text=text,
            title=title,
            tags=tags or [],
            metadata=metadata or {}
        )

        if memory_id:
            internal_memory = await self._api.get_memory(memory_id)
            return Memory(internal_memory) if internal_memory else None

        return None

    async def recall(
        self,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Search for memories.

        Args:
            query: What to search for
            limit: Maximum number of results
            tags: Optional tag filters

        Returns:
            List of search results with relevance scores.
        """
        self._check_started()

        results = await self._api.search(
            query=query,
            top_k=limit,
            tags=tags
        )

        return [SearchResult(result) for result in results]

    async def get(self, memory_id: str) -> Memory | None:
        """
        Get a specific memory by ID.

        Args:
            memory_id: The memory identifier

        Returns:
            Memory object if found, None otherwise.
        """
        self._check_started()

        internal_memory = await self._api.get_memory(memory_id)
        return Memory(internal_memory) if internal_memory else None

    async def find_related(self, memory_id: str, limit: int = 5) -> list[SearchResult]:
        """
        Find memories related to a specific memory.

        Args:
            memory_id: The reference memory ID
            limit: Maximum number of related memories

        Returns:
            List of related memories.
        """
        self._check_started()

        results = await self._api.get_related_memories(memory_id, limit)
        return [SearchResult(result) for result in results]

    async def forget(self, memory_id: str) -> bool:
        """
        Remove a memory from the system.

        Args:
            memory_id: The memory to remove

        Returns:
            True if successful, False otherwise.
        """
        self._check_started()

        # Mark as deleted in cortex
        if self._api.cortex:
            await self._api.cortex.mark_deleted(memory_id)
            return True

        return False

    async def consolidate(self) -> int:
        """
        Run memory consolidation to optimize storage.

        Returns:
            Number of memories processed.
        """
        self._check_started()

        return await self._api.run_auto_consolidation()

    async def health(self) -> bool:
        """
        Check if the memory system is healthy.

        Returns:
            True if all components are working, False otherwise.
        """
        if not self._started:
            return False

        health_status = await self._api.health_check()
        return all(status for status in health_status.values() if isinstance(status, bool))

    def _check_started(self):
        """Ensure the system has been started."""
        if not self._started:
            raise RuntimeError("Memory system not started. Call await episemic.start() first.")

    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed
        pass


# Convenience function for even simpler usage
async def create_memory_system(**config_kwargs) -> Episemic:
    """
    Create and start a memory system in one call.

    Args:
        **config_kwargs: Configuration options

    Returns:
        Started Episemic instance.
    """
    episemic = Episemic(**config_kwargs)
    await episemic.start()
    return episemic


# Synchronous wrapper for non-async environments
class EpistemicSync:
    """
    Synchronous wrapper for Episemic (for non-async code).

    Example:
        >>> episemic = EpistemicSync()
        >>> episemic.start()
        >>> memory = episemic.remember("Important info")
        >>> results = episemic.recall("important")
    """

    def __init__(self, **config_kwargs):
        self._async_episemic = Episemic(**config_kwargs)
        self._loop = None

    def _run_async(self, coro):
        """Run async function in sync context."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        return self._loop.run_until_complete(coro)

    def start(self) -> bool:
        """Start the memory system."""
        return self._run_async(self._async_episemic.start())

    def remember(
        self,
        text: str,
        title: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Memory | None:
        """Store a new memory."""
        return self._run_async(
            self._async_episemic.remember(text, title, tags, metadata)
        )

    def recall(
        self,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for memories."""
        return self._run_async(
            self._async_episemic.recall(query, limit, tags)
        )

    def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory."""
        return self._run_async(self._async_episemic.get(memory_id))

    def find_related(self, memory_id: str, limit: int = 5) -> list[SearchResult]:
        """Find related memories."""
        return self._run_async(
            self._async_episemic.find_related(memory_id, limit)
        )

    def forget(self, memory_id: str) -> bool:
        """Remove a memory."""
        return self._run_async(self._async_episemic.forget(memory_id))

    def consolidate(self) -> int:
        """Run memory consolidation."""
        return self._run_async(self._async_episemic.consolidate())

    def health(self) -> bool:
        """Check system health."""
        return self._run_async(self._async_episemic.health())