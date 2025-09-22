"""High-level API for Episemic Core - Simple library interface."""


from .config import EpistemicConfig
from .hippocampus import Hippocampus
from .hippocampus.duckdb_hippocampus import DuckDBHippocampus
from .models import Memory, SearchQuery, SearchResult
from .retrieval import RetrievalEngine

# Optional imports - these packages are only needed if cortex is enabled
try:
    from .consolidation import ConsolidationEngine
    from .cortex import Cortex
    CORTEX_AVAILABLE = True
except ImportError:
    ConsolidationEngine = None
    Cortex = None
    CORTEX_AVAILABLE = False


class EpistemicAPI:
    """
    High-level API for Episemic Core memory system.

    This class provides a simple interface for using Episemic as a Python library.

    Example:
        >>> api = EpistemicAPI()
        >>> await api.initialize()
        >>>
        >>> # Store a memory
        >>> memory_id = await api.store_memory(
        ...     "This is important information",
        ...     title="Important Note",
        ...     tags=["important", "note"]
        ... )
        >>>
        >>> # Search for memories
        >>> results = await api.search("important information", top_k=5)
        >>>
        >>> # Get a specific memory
        >>> memory = await api.get_memory(memory_id)
    """

    def __init__(self, config: EpistemicConfig | None = None):
        """
        Initialize Episemic API.

        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or EpistemicConfig()
        self._initialized = False

        # Core components
        self.hippocampus: Hippocampus | DuckDBHippocampus | None = None
        self.cortex: Cortex | None = None
        self.consolidation_engine: ConsolidationEngine | None = None
        self.retrieval_engine: RetrievalEngine | None = None

    async def initialize(self) -> bool:
        """
        Initialize all system components.

        Returns:
            True if initialization successful, False otherwise.
        """
        initialization_success = True

        # Initialize hippocampus (required for basic functionality)
        if self.config.enable_hippocampus:
            try:
                # Try to determine which storage backend to use
                use_duckdb = await self._should_use_duckdb()

                if use_duckdb:
                    # Use DuckDB fallback
                    self.hippocampus = DuckDBHippocampus(
                        db_path=self.config.duckdb.db_path,
                        model_name=self.config.duckdb.model_name
                    )
                    if self.config.debug:
                        print("✓ Using DuckDB for hippocampus storage")
                else:
                    # Use Qdrant + Redis
                    self.hippocampus = Hippocampus(
                        qdrant_host=self.config.qdrant.host,
                        qdrant_port=self.config.qdrant.port,
                        redis_host=self.config.redis.host,
                        redis_port=self.config.redis.port,
                        collection_name=self.config.qdrant.collection_name,
                    )
                    if self.config.debug:
                        print("✓ Using Qdrant + Redis for hippocampus storage")
            except Exception as e:
                if self.config.debug:
                    print(f"✗ Hippocampus initialization failed: {e}")
                initialization_success = False

        # Initialize cortex (optional - may fail if PostgreSQL not available or dependencies missing)
        if self.config.enable_cortex and CORTEX_AVAILABLE:
            try:
                self.cortex = Cortex(
                    db_host=self.config.postgresql.host,
                    db_port=self.config.postgresql.port,
                    db_name=self.config.postgresql.database,
                    db_user=self.config.postgresql.user,
                    db_password=self.config.postgresql.password,
                )
                if self.config.debug:
                    print("✓ Using PostgreSQL for cortex storage")
            except Exception as e:
                if self.config.debug:
                    print(f"⚠ Cortex initialization failed (PostgreSQL unavailable): {e}")
                    print("⚠ Continuing with hippocampus-only mode")
                # Don't mark as failure - DuckDB-only mode is valid
        elif self.config.enable_cortex and not CORTEX_AVAILABLE:
            if self.config.debug:
                print("⚠ Cortex dependencies not available (missing psycopg2)")
                print("⚠ Continuing with hippocampus-only mode")

        # Initialize consolidation engine (only if both hippocampus and cortex available)
        if (self.config.enable_consolidation and self.hippocampus and self.cortex
            and CORTEX_AVAILABLE and ConsolidationEngine is not None):
            try:
                self.consolidation_engine = ConsolidationEngine(
                    self.hippocampus,
                    self.cortex
                )
                # Set consolidation parameters
                self.consolidation_engine.consolidation_threshold_hours = self.config.consolidation.threshold_hours
                self.consolidation_engine.consolidation_access_threshold = self.config.consolidation.access_threshold
                if self.config.debug:
                    print("✓ Consolidation engine initialized")
            except Exception as e:
                if self.config.debug:
                    print(f"⚠ Consolidation engine initialization failed: {e}")
        elif self.config.enable_consolidation and not CORTEX_AVAILABLE:
            if self.config.debug:
                print("⚠ Consolidation requires cortex dependencies - disabled")

        # Initialize retrieval engine (works with just hippocampus if cortex unavailable)
        if self.config.enable_retrieval and self.hippocampus:
            try:
                self.retrieval_engine = RetrievalEngine(
                    self.hippocampus,
                    self.cortex  # May be None, that's OK
                )
                if self.config.debug:
                    print("✓ Retrieval engine initialized")
            except Exception as e:
                if self.config.debug:
                    print(f"⚠ Retrieval engine initialization failed: {e}")

        # Mark as initialized if we at least have hippocampus
        if self.hippocampus:
            self._initialized = True
            if self.config.debug:
                print("✓ API initialization completed")
            return True
        else:
            if self.config.debug:
                print("✗ API initialization failed - no storage backend available")
            return False

    async def _should_use_duckdb(self) -> bool:
        """
        Determine whether to use DuckDB or Qdrant based on configuration and availability.

        Returns:
            True if should use DuckDB, False if should use Qdrant.
        """
        # If explicitly configured to use DuckDB fallback, use it
        if self.config.use_duckdb_fallback and not self.config.prefer_qdrant:
            return True

        # If prefer_qdrant is set, try to connect to Qdrant first
        if self.config.prefer_qdrant:
            try:
                # Try to create a test connection to Qdrant
                from qdrant_client import QdrantClient
                client = QdrantClient(
                    host=self.config.qdrant.host,
                    port=self.config.qdrant.port,
                    timeout=5  # Short timeout for quick check
                )
                # Try a simple health check
                client.get_collections()
                return False  # Qdrant is available
            except Exception:
                # Qdrant not available, fall back to DuckDB
                if self.config.debug:
                    print("Qdrant not available, falling back to DuckDB")
                return True

        # Default to DuckDB (no external dependencies)
        return True

    async def store_memory(
        self,
        text: str,
        title: str | None = None,
        source: str = "api",
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        metadata: dict | None = None,
        store_in_hippocampus: bool = True,
        store_in_cortex: bool = True,
    ) -> str | None:
        """
        Store a new memory in the system.

        Args:
            text: The main content of the memory
            title: Optional title for the memory
            source: Source identifier (default: "api")
            tags: Optional list of tags
            embedding: Optional pre-computed embedding vector
            metadata: Optional additional metadata
            store_in_hippocampus: Whether to store in fast hippocampus layer
            store_in_cortex: Whether to store in persistent cortex layer

        Returns:
            Memory ID if successful, None otherwise.
        """
        self._check_initialized()

        memory = Memory(
            title=title or text[:50] + "..." if len(text) > 50 else text,
            text=text,
            summary=text[:200] + "..." if len(text) > 200 else text,
            source=source,
            tags=tags or [],
            embedding_v1=embedding,
            metadata=metadata or {},
        )

        success = True

        # Store in hippocampus (fast layer)
        if store_in_hippocampus and self.hippocampus:
            if not await self.hippocampus.store_memory(memory):
                success = False

        # Store in cortex (persistent layer)
        if store_in_cortex and self.cortex:
            if not await self.cortex.store_memory(memory):
                success = False

        return memory.id if success else None

    async def search(
        self,
        query: str,
        top_k: int = 10,
        tags: list[str] | None = None,
        include_quarantined: bool = False,
        embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """
        Search for memories using multiple retrieval strategies.

        Args:
            query: Search query text
            top_k: Maximum number of results to return
            tags: Optional tag filters
            include_quarantined: Whether to include quarantined memories
            embedding: Optional pre-computed query embedding

        Returns:
            List of search results with relevance scores.
        """
        self._check_initialized()

        if not self.retrieval_engine:
            # Fallback to direct hippocampus search if retrieval engine not available
            if self.hippocampus:
                try:
                    # Generate embedding for the query
                    if hasattr(self.hippocampus, 'get_embedding'):
                        embedding = await self.hippocampus.get_embedding(query)
                        search_results = await self.hippocampus.vector_search(
                            embedding, top_k, {"tags": tags[0]} if tags else None
                        )
                        # Convert to SearchResult format
                        from .models import SearchResult, Memory
                        results = []
                        for result in search_results:
                            memory = Memory(
                                id=result["id"],
                                text=result["content"],
                                summary=result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                                title=result.get("title", ""),
                                source=result.get("source", ""),
                                tags=result.get("tags", []),
                                metadata=result.get("metadata", {}),
                                created_at=result.get("created_at"),
                                access_count=result.get("access_count", 0),
                                last_accessed=result.get("last_accessed")
                            )
                            search_result = SearchResult(
                                memory=memory,
                                score=result.get("similarity", 1.0),
                                context="",
                                metadata=result.get("metadata", {})
                            )
                            results.append(search_result)
                        return results
                except Exception as e:
                    if self.config.debug:
                        print(f"Fallback search failed: {e}")
            return []

        search_query = SearchQuery(
            query=query,
            top_k=top_k,
            filters={"tags": tags} if tags else {},
            include_quarantined=include_quarantined,
        )

        # Add embedding if provided
        if embedding:
            search_query.embedding = embedding

        return await self.retrieval_engine.search(search_query)

    async def get_memory(self, memory_id: str) -> Memory | None:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: The memory ID to retrieve

        Returns:
            Memory object if found, None otherwise.
        """
        self._check_initialized()

        if not self.retrieval_engine:
            return None

        return await self.retrieval_engine.retrieve_by_id(memory_id)

    async def get_related_memories(
        self,
        memory_id: str,
        max_related: int = 5
    ) -> list[SearchResult]:
        """
        Get memories related to a specific memory.

        Args:
            memory_id: The reference memory ID
            max_related: Maximum number of related memories to return

        Returns:
            List of related memories with relevance scores.
        """
        self._check_initialized()

        if not self.retrieval_engine:
            return []

        return await self.retrieval_engine.get_related_memories(memory_id, max_related)

    async def consolidate_memory(self, memory_id: str) -> bool:
        """
        Manually trigger consolidation for a specific memory.

        Args:
            memory_id: The memory ID to consolidate

        Returns:
            True if consolidation successful, False otherwise.
        """
        self._check_initialized()

        if not self.consolidation_engine:
            return False

        return await self.consolidation_engine.consolidate_memory(memory_id)

    async def run_auto_consolidation(self) -> int:
        """
        Run automatic consolidation sweep.

        Returns:
            Number of memories processed.
        """
        self._check_initialized()

        if not self.consolidation_engine:
            return 0

        return await self.consolidation_engine.auto_consolidation_sweep()

    async def health_check(self) -> dict[str, bool]:
        """
        Check the health status of all components.

        Returns:
            Dictionary with health status of each component.
        """
        self._check_initialized()

        health_status = {}

        if self.hippocampus:
            health_status.update({
                f"hippocampus_{k}": v
                for k, v in self.hippocampus.health_check().items()
            })

        if self.cortex:
            health_status["cortex_healthy"] = self.cortex.health_check()

        if self.consolidation_engine:
            consolidation_health = self.consolidation_engine.health_check()
            health_status.update({
                f"consolidation_{k}": v
                for k, v in consolidation_health.items()
            })

        if self.retrieval_engine:
            retrieval_health = self.retrieval_engine.health_check()
            health_status.update({
                f"retrieval_{k}": v
                for k, v in retrieval_health.items()
            })

        return health_status

    def _check_initialized(self):
        """Check if the API has been initialized."""
        if not self._initialized:
            raise RuntimeError(
                "EpistemicAPI not initialized. Call await api.initialize() first."
            )

    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed
        pass


# Convenience functions for quick usage
async def create_memory_system(config: EpistemicConfig | None = None) -> EpistemicAPI:
    """
    Create and initialize a memory system.

    Args:
        config: Optional configuration

    Returns:
        Initialized EpistemicAPI instance.
    """
    api = EpistemicAPI(config)
    await api.initialize()
    return api


def create_config_from_env() -> EpistemicConfig:
    """Create configuration from environment variables."""
    return EpistemicConfig.from_env()


def create_config(**kwargs) -> EpistemicConfig:
    """Create configuration with custom parameters."""
    return EpistemicConfig(**kwargs)
