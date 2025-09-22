"""Retrieval engine for multi-path memory search and context composition."""


from ..cortex import Cortex
from ..hippocampus import Hippocampus
from ..models import Memory, SearchQuery, SearchResult


class RetrievalEngine:
    def __init__(self, hippocampus: Hippocampus, cortex: Cortex | None = None):
        self.hippocampus = hippocampus
        self.cortex = cortex

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        try:
            results = []

            # Path 1: Search in hippocampus (DuckDB or Qdrant)
            if hasattr(query, 'embedding') and query.embedding:
                # Vector search with embeddings
                hippocampus_results = await self.hippocampus.vector_search(
                    query_vector=query.embedding,
                    top_k=query.top_k,
                    filters=self._build_search_filters(query.filters)
                )
            else:
                # Fallback to text-based search in DuckDB
                hippocampus_results = await self.hippocampus.vector_search(
                    query_vector=[],  # Empty vector for text search
                    top_k=query.top_k,
                    filters=self._build_search_filters(query.filters)
                )

            # Convert hippocampus results to SearchResult objects
            for result in hippocampus_results:
                try:
                    # Handle different result formats from different hippocampus implementations
                    if isinstance(result, dict):
                        if "memory" in result:
                            memory = result["memory"]
                            score = result.get("score", result.get("similarity", 1.0))
                        else:
                            # DuckDB format - convert dict to Memory object
                            from ..models import Memory
                            content = result["content"]
                            memory = Memory(
                                id=result["id"],
                                text=content,
                                summary=content[:200] + "..." if len(content) > 200 else content,
                                title=result.get("title", ""),
                                source=result.get("source", "hippocampus"),
                                tags=result.get("tags", []),
                                metadata=result.get("metadata", {})
                            )
                            score = result.get("similarity", 1.0)

                        search_result = SearchResult(
                            memory=memory,
                            score=score,
                            provenance={"source": "hippocampus", "method": "vector_similarity"},
                            retrieval_path=["hippocampus", "vector_search"]
                        )
                        results.append(search_result)
                except Exception as e:
                    print(f"Error processing hippocampus result: {e}")
                    continue

            # Path 2: Tag-based search in cortex (semantic) - only if cortex available
            if self.cortex and query.filters.get("tags"):
                try:
                    cortex_results = await self.cortex.search_by_tags(
                        tags=query.filters["tags"],
                        limit=query.top_k
                    )

                    for memory in cortex_results:
                        search_result = SearchResult(
                            memory=memory,
                            score=self._calculate_tag_relevance_score(memory, query.filters["tags"]),
                            provenance={"source": "cortex", "method": "tag_search"},
                            retrieval_path=["cortex", "tag_search"]
                        )
                        results.append(search_result)
                except Exception as e:
                    print(f"Error in cortex search: {e}")

            # Path 3: Graph traversal for contextual memories - only if cortex available
            if self.cortex and query.filters.get("context_memory_id"):
                try:
                    context_results = await self._search_by_context(
                        query.filters["context_memory_id"],
                        query.top_k
                    )
                    results.extend(context_results)
                except Exception as e:
                    print(f"Error in context search: {e}")

            # Deduplicate and rank results
            results = self._deduplicate_results(results)
            results = self._rank_results(results, query)

            # Update access counts for retrieved memories - only if cortex available
            if self.cortex:
                for result in results[:query.top_k]:
                    try:
                        await self.cortex.increment_access_count(result.memory.id)
                    except Exception:
                        pass  # Ignore access count update errors

            return results[:query.top_k]

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    async def retrieve_by_id(self, memory_id: str) -> Memory | None:
        # Try hippocampus first (faster)
        memory = await self.hippocampus.retrieve_memory(memory_id)
        if memory:
            if self.cortex:
                await self.cortex.increment_access_count(memory_id)
            return memory

        # Fallback to cortex if available
        if self.cortex:
            memory = await self.cortex.retrieve_memory(memory_id)
            if memory:
                await self.cortex.increment_access_count(memory_id)
                return memory

        return None

    async def get_related_memories(self, memory_id: str, max_related: int = 5) -> list[SearchResult]:
        try:
            # Get the base memory
            base_memory = await self.retrieve_by_id(memory_id)
            if not base_memory:
                return []

            results = []

            # Find memories with shared tags using hippocampus (DuckDB search)
            if base_memory.tags:
                # Use hippocampus to find memories with similar tags
                hippocampus_results = await self.hippocampus.vector_search(
                    query_vector=[],  # No vector search, just tag-based
                    top_k=max_related * 2,  # Get more to filter out the original
                    filters={"tags": base_memory.tags}
                )

                for result in hippocampus_results:
                    try:
                        if result["id"] != memory_id:  # Exclude the original memory
                            from ..models import Memory
                            content = result["content"]
                            memory = Memory(
                                id=result["id"],
                                text=content,
                                summary=content[:200] + "..." if len(content) > 200 else content,
                                title=result.get("title", ""),
                                source=result.get("source", "hippocampus"),
                                tags=result.get("tags", []),
                                metadata=result.get("metadata", {})
                            )

                            score = self._calculate_tag_overlap_score(base_memory.tags, memory.tags)
                            search_result = SearchResult(
                                memory=memory,
                                score=score,
                                provenance={"source": "hippocampus", "method": "tag_overlap"},
                                retrieval_path=["hippocampus", "related_search", "tag_overlap"]
                            )
                            results.append(search_result)
                    except Exception as e:
                        print(f"Error processing related memory result: {e}")
                        continue

            # If cortex is available, also search there
            if self.cortex and base_memory.tags:
                try:
                    tag_matches = await self.cortex.search_by_tags(
                        tags=base_memory.tags,
                        limit=max_related * 2
                    )

                    for memory in tag_matches:
                        if memory.id != memory_id:
                            score = self._calculate_tag_overlap_score(base_memory.tags, memory.tags)
                            result = SearchResult(
                                memory=memory,
                                score=score,
                                provenance={"source": "cortex", "method": "tag_overlap"},
                                retrieval_path=["cortex", "related_search", "tag_overlap"]
                            )
                            results.append(result)

                    # Get graph-connected memories
                    graph_data = await self.cortex.get_memory_graph(memory_id, depth=2)
                    for node in graph_data["nodes"]:
                        if node["id"] != memory_id:
                            retrieved_memory = await self.cortex.retrieve_memory(node["id"])
                            if retrieved_memory:
                                result = SearchResult(
                                    memory=retrieved_memory,
                                    score=0.8,  # High score for directly linked memories
                                    provenance={"source": "cortex", "method": "graph_traversal"},
                                    retrieval_path=["cortex", "related_search", "graph_traversal"]
                                )
                                results.append(result)
                except Exception as e:
                    print(f"Error in cortex related search: {e}")

            # Deduplicate and rank
            results = self._deduplicate_results(results)
            results.sort(key=lambda x: x.score, reverse=True)

            return results[:max_related]

        except Exception as e:
            print(f"Error getting related memories: {e}")
            return []

    async def _search_by_context(self, context_memory_id: str, top_k: int) -> list[SearchResult]:
        try:
            graph_data = await self.cortex.get_memory_graph(context_memory_id, depth=1)
            results = []

            for node in graph_data["nodes"]:
                if node["id"] != context_memory_id:
                    memory = await self.cortex.retrieve_memory(node["id"])
                    if memory:
                        result = SearchResult(
                            memory=memory,
                            score=0.7,  # Context relevance score
                            provenance={"source": "cortex", "method": "context_search"},
                            retrieval_path=["cortex", "context_search", "graph_traversal"]
                        )
                        results.append(result)

            return results[:top_k]

        except Exception as e:
            print(f"Error in context search: {e}")
            return []

    def _build_search_filters(self, filters: dict) -> dict | None:
        """Build search filters compatible with both DuckDB and Qdrant backends."""
        if not filters:
            return None

        search_filter = {}

        if filters.get("tags"):
            search_filter["tags"] = filters["tags"]

        if filters.get("source"):
            search_filter["source"] = filters["source"]

        if filters.get("retention_policy"):
            search_filter["retention_policy"] = filters["retention_policy"]

        return search_filter if search_filter else None

    def _build_qdrant_filters(self, filters: dict) -> dict | None:
        """Legacy method for Qdrant-specific filters."""
        if not filters:
            return None

        qdrant_filter = {}

        if filters.get("tags"):
            qdrant_filter["tags"] = {"any": filters["tags"]}

        if filters.get("source"):
            qdrant_filter["source"] = filters["source"]

        if filters.get("retention_policy"):
            qdrant_filter["retention_policy"] = filters["retention_policy"]

        return qdrant_filter if qdrant_filter else None

    def _calculate_tag_relevance_score(self, memory: Memory, query_tags: list[str]) -> float:
        if not memory.tags or not query_tags:
            return 0.0

        overlap = set(memory.tags) & set(query_tags)
        return len(overlap) / len(set(memory.tags) | set(query_tags))

    def _calculate_tag_overlap_score(self, tags1: list[str], tags2: list[str]) -> float:
        if not tags1 or not tags2:
            return 0.0

        overlap = set(tags1) & set(tags2)
        union = set(tags1) | set(tags2)
        return len(overlap) / len(union) if union else 0.0

    def _deduplicate_results(self, results: list[SearchResult]) -> list[SearchResult]:
        seen_ids = set()
        deduplicated = []

        for result in results:
            if result.memory.id not in seen_ids:
                seen_ids.add(result.memory.id)
                deduplicated.append(result)

        return deduplicated

    def _rank_results(self, results: list[SearchResult], query: SearchQuery) -> list[SearchResult]:
        # Simple ranking by score, but could be enhanced with more sophisticated ranking
        results.sort(key=lambda x: x.score, reverse=True)

        # Boost recently accessed memories slightly
        for result in results:
            if result.memory.access_count > 5:
                result.score *= 1.1

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def health_check(self) -> dict:
        health = {
            "hippocampus_healthy": self.hippocampus.health_check() if hasattr(self.hippocampus, 'health_check') else True,
        }

        if self.cortex:
            health["cortex_healthy"] = self.cortex.health_check() if hasattr(self.cortex, 'health_check') else True
        else:
            health["cortex_healthy"] = None  # Not available

        return health
