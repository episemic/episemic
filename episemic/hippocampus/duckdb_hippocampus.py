"""DuckDB-based hippocampus implementation for local fallback storage."""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
from sentence_transformers import SentenceTransformer

from ..models import Memory


class DuckDBHippocampus:
    """DuckDB-based hippocampus for local vector storage without external dependencies."""

    def __init__(self, db_path: str | None = None, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize DuckDB hippocampus with local storage.

        Args:
            db_path: Path to DuckDB file. If None, uses in-memory database.
            model_name: Sentence transformer model for embeddings.
        """
        self.db_path = db_path or ":memory:"
        self.model_name = model_name
        self.model = None
        self._initialized = False

        # Create data directory if using file storage
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def _ensure_initialized(self):
        """Ensure the database and model are initialized."""
        if self._initialized:
            return

        # Initialize model in thread executor to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            self.model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self.model_name, token=False)
            )
        except Exception as e:
            print(f"Warning: Failed to load sentence transformer model '{self.model_name}': {e}")
            print("Falling back to simple text-based storage without embeddings")
            self.model = None

        # Initialize database
        self.conn = duckdb.connect(self.db_path)

        # Create tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id VARCHAR PRIMARY KEY,
                content TEXT NOT NULL,
                title VARCHAR,
                source VARCHAR,
                tags TEXT[], -- Array of strings
                metadata TEXT, -- JSON string
                embedding FLOAT[], -- Array of floats
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_quarantined BOOLEAN DEFAULT FALSE
            )
        """)

        # Note: DuckDB doesn't support HNSW indexes yet, but we can still do similarity search
        # The vector search will use array_cosine_similarity function

        self._initialized = True

    async def store_memory(self, memory: Memory) -> bool:
        """Store a memory in DuckDB with vector embedding."""
        try:
            await self._ensure_initialized()

            # Generate embedding if model is available
            embedding = None
            if self.model:
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None,
                    lambda: self.model.encode(memory.text).tolist()
                )
            else:
                # No embedding model available, use empty array
                embedding = []

            # Convert metadata to JSON string
            metadata_json = json.dumps(memory.metadata) if memory.metadata else None

            # Insert into database
            self.conn.execute("""
                INSERT INTO memories (
                    id, content, title, source, tags, metadata, embedding,
                    created_at, access_count, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                memory.id,
                memory.text,
                memory.title,
                memory.source,
                memory.tags,
                metadata_json,
                embedding,
                memory.created_at,
                memory.access_count,
                memory.last_accessed
            ])

            return True
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False

    async def retrieve_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID."""
        try:
            await self._ensure_initialized()

            result = self.conn.execute("""
                SELECT id, content, title, source, tags, metadata,
                       created_at, access_count, last_accessed, is_quarantined
                FROM memories
                WHERE id = ? AND is_quarantined = FALSE
            """, [memory_id]).fetchone()

            if not result:
                return None

            # Parse metadata
            metadata = json.loads(result[5]) if result[5] else {}

            return Memory(
                id=result[0],
                text=result[1],
                summary=result[1][:200] + "..." if len(result[1]) > 200 else result[1],
                title=result[2],
                source=result[3],
                tags=result[4] or [],
                metadata=metadata,
                created_at=result[6],
                access_count=result[7],
                last_accessed=result[8]
            )
        except Exception as e:
            print(f"Error retrieving memory: {e}")
            return None

    async def vector_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None
    ) -> list[dict]:
        """Perform vector similarity search."""
        try:
            await self._ensure_initialized()

            # If no embeddings are available, fallback to basic text search
            if not self.model or not query_vector:
                print(f"DEBUG: Falling back to text search - model: {self.model is not None}, query_vector: {len(query_vector) if query_vector else 0}")
                return await self._fallback_text_search(query_vector, top_k, filters)

            # Build filter conditions
            where_clause = "WHERE is_quarantined = FALSE AND len(embedding) > 0"
            params = []

            if filters:
                if "tags" in filters:
                    # Handle tag filtering for DuckDB - check if any of the query tags are in the memory tags
                    tag_filter = filters["tags"]
                    if isinstance(tag_filter, list):
                        # If multiple tags, check if any match
                        tag_conditions = []
                        for tag in tag_filter:
                            tag_conditions.append("? = ANY(tags)")
                            params.append(tag)
                        where_clause += " AND (" + " OR ".join(tag_conditions) + ")"
                    else:
                        # Single tag
                        where_clause += " AND ? = ANY(tags)"
                        params.append(tag_filter)
                if "source" in filters:
                    where_clause += " AND source = ?"
                    params.append(filters["source"])

            # Get all memories with embeddings and calculate similarity in Python
            query = f"""
                SELECT id, content, title, source, tags, metadata,
                       created_at, access_count, last_accessed, embedding
                FROM memories
                {where_clause}
            """

            all_results = self.conn.execute(query, params).fetchall()

            # Calculate cosine similarity in Python for each result
            scored_results = []
            for row in all_results:
                embedding = row[9]  # embedding column
                if embedding and len(embedding) > 0:
                    similarity = self._calculate_cosine_similarity(query_vector, embedding)
                else:
                    similarity = 0.0

                scored_results.append({
                    "id": row[0],
                    "content": row[1],
                    "title": row[2],
                    "source": row[3],
                    "tags": row[4] or [],
                    "metadata": json.loads(row[5]) if row[5] else {},
                    "created_at": row[6],
                    "access_count": row[7],
                    "last_accessed": row[8],
                    "similarity": similarity
                })

            # Sort by similarity and return top_k
            scored_results.sort(key=lambda x: x["similarity"], reverse=True)
            return scored_results[:top_k]
        except Exception as e:
            print(f"Error in vector search: {e}")
            return await self._fallback_text_search(query_vector, top_k, filters)

    def _calculate_cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import math

            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return 0.0

            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))

            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))

            # Avoid division by zero
            if magnitude1 == 0.0 or magnitude2 == 0.0:
                return 0.0

            # Calculate cosine similarity
            similarity = dot_product / (magnitude1 * magnitude2)

            # Clamp to [0, 1] range for consistency
            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.0

    async def _fallback_text_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None
    ) -> list[dict]:
        """Fallback text-based search when embeddings are not available."""
        try:
            await self._ensure_initialized()

            # Build filter conditions
            where_clause = "WHERE is_quarantined = FALSE"
            params = []

            if filters:
                if "tags" in filters:
                    tag_filter = filters["tags"]
                    if isinstance(tag_filter, list):
                        tag_conditions = []
                        for tag in tag_filter:
                            tag_conditions.append("? = ANY(tags)")
                            params.append(tag)
                        where_clause += " AND (" + " OR ".join(tag_conditions) + ")"
                    else:
                        where_clause += " AND ? = ANY(tags)"
                        params.append(tag_filter)
                if "source" in filters:
                    where_clause += " AND source = ?"
                    params.append(filters["source"])

            # Simple text-based ordering (fallback when no embeddings)
            query = f"""
                SELECT id, content, title, source, tags, metadata,
                       created_at, access_count, last_accessed,
                       0.5 AS similarity
                FROM memories
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
            """

            params = params + [top_k]
            results = self.conn.execute(query, params).fetchall()

            return [
                {
                    "id": row[0],
                    "content": row[1],
                    "title": row[2],
                    "source": row[3],
                    "tags": row[4] or [],
                    "metadata": json.loads(row[5]) if row[5] else {},
                    "created_at": row[6],
                    "access_count": row[7],
                    "last_accessed": row[8],
                    "similarity": row[9]
                }
                for row in results
            ]
        except Exception as e:
            print(f"Error in fallback text search: {e}")
            return []

    async def mark_quarantined(self, memory_id: str) -> bool:
        """Mark a memory as quarantined."""
        try:
            await self._ensure_initialized()

            self.conn.execute("""
                UPDATE memories
                SET is_quarantined = TRUE
                WHERE id = ?
            """, [memory_id])

            return True
        except Exception as e:
            print(f"Error quarantining memory: {e}")
            return False

    async def verify_integrity(self, memory_id: str) -> bool:
        """Verify memory integrity."""
        try:
            await self._ensure_initialized()

            result = self.conn.execute("""
                SELECT COUNT(*) FROM memories WHERE id = ?
            """, [memory_id]).fetchone()

            return result[0] > 0
        except Exception:
            return False

    def health_check(self) -> dict[str, bool]:
        """Check health of DuckDB storage."""
        try:
            if not self._initialized:
                return {
                    "duckdb": False,
                    "model": False,
                    "embeddings": False
                }

            # Test database connection
            self.conn.execute("SELECT 1").fetchone()

            # Test model
            test_embedding = self.model.encode("test").tolist()

            return {
                "duckdb": True,
                "model": True,
                "embeddings": len(test_embedding) > 0
            }
        except Exception:
            return {
                "duckdb": False,
                "model": False,
                "embeddings": False
            }

    async def get_memory_count(self) -> int:
        """Get total number of non-quarantined memories."""
        try:
            await self._ensure_initialized()

            result = self.conn.execute("""
                SELECT COUNT(*) FROM memories WHERE is_quarantined = FALSE
            """).fetchone()

            return result[0] if result else 0
        except Exception:
            return 0

    async def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        await self._ensure_initialized()

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.model.encode(text).tolist()
        )
        return embedding

    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn is not None:
            self.conn.close()
            self.conn = None