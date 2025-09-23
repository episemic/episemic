"""Hippocampus implementation - Fast, writable, vector-indexed store for recent memories."""

try:
    import redis
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    redis = None
    QdrantClient = None
    Distance = None
    PointStruct = None
    VectorParams = None
    QDRANT_AVAILABLE = False

from ..models import Memory, MemoryStatus


class Hippocampus:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        collection_name: str = "episodic_memories",
    ):
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.collection_name = collection_name
        self._setup_collection()

    def _setup_collection(self):
        collections = self.qdrant_client.get_collections()
        collection_exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if not collection_exists:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    async def store_memory(self, memory: Memory) -> bool:
        try:
            if not memory.embedding_v1:
                raise ValueError("Memory must have embedding_v1 to store in Hippocampus")

            point = PointStruct(
                id=memory.id,
                vector=memory.embedding_v1,
                payload={
                    "text": memory.text,
                    "summary": memory.summary,
                    "source": memory.source,
                    "tags": memory.tags,
                    "created_at": memory.created_at.isoformat(),
                    "retention_policy": memory.retention_policy.value,
                    "status": memory.status.value,
                },
            )

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            self.redis_client.setex(
                f"memory:{memory.id}",
                3600,  # 1 hour TTL for fast access
                memory.model_dump_json()
            )

            return True

        except Exception as e:
            print(f"Error storing memory in Hippocampus: {e}")
            return False

    async def retrieve_memory(self, memory_id: str) -> Memory | None:
        cached = self.redis_client.get(f"memory:{memory_id}")
        if cached:
            return Memory.model_validate_json(str(cached))

        try:
            points = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id]
            )

            if points:
                point = points[0]
                memory_data = point.payload or {}
                memory_data["id"] = str(point.id)
                memory_data["embedding_v1"] = point.vector

                memory = Memory.model_validate(memory_data)
                self.redis_client.setex(
                    f"memory:{memory_id}",
                    3600,
                    memory.model_dump_json()
                )
                return memory

        except Exception as e:
            print(f"Error retrieving memory from Hippocampus: {e}")

        return None

    async def vector_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None
    ) -> list[dict]:
        try:
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filters,
            )

            results = []
            for point in search_result:
                memory_data = point.payload or {}
                memory_data["id"] = str(point.id)
                memory_data["embedding_v1"] = point.vector

                results.append({
                    "memory": Memory.model_validate(memory_data),
                    "score": point.score,
                })

            return results

        except Exception as e:
            print(f"Error in vector search: {e}")
            return []

    async def mark_quarantined(self, memory_id: str) -> bool:
        try:
            self.qdrant_client.set_payload(
                collection_name=self.collection_name,
                payload={"status": MemoryStatus.QUARANTINED.value},
                points=[memory_id],
            )

            self.redis_client.delete(f"memory:{memory_id}")
            return True

        except Exception as e:
            print(f"Error marking memory as quarantined: {e}")
            return False

    async def verify_integrity(self, memory_id: str) -> bool:
        memory = await self.retrieve_memory(memory_id)
        if not memory:
            return False

        return memory.verify_integrity()

    async def get_all_memory_ids(self) -> list[str]:
        """Get all memory IDs from active memories."""
        try:
            # Get all points from the collection
            points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Adjust based on expected memory count
            )[0]

            memory_ids = []
            for point in points:
                # Check if memory exists in Redis and is not quarantined
                memory_key = f"memory:{point.id}"
                if self.redis_client.exists(memory_key):
                    memory_data = self.redis_client.hgetall(memory_key)
                    if memory_data.get("status") != "quarantined":
                        memory_ids.append(str(point.id))

            return memory_ids
        except Exception:
            return []

    async def remove_memory(self, memory_id: str) -> bool:
        """Remove a memory completely from the hippocampus."""
        try:
            # Remove from Qdrant
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=[memory_id]
            )

            # Remove from Redis
            self.redis_client.delete(f"memory:{memory_id}")
            return True
        except Exception:
            return False

    def health_check(self) -> dict[str, bool]:
        return {
            "qdrant_connected": self._check_qdrant_connection(),
            "redis_connected": self._check_redis_connection(),
        }

    def _check_qdrant_connection(self) -> bool:
        try:
            self.qdrant_client.get_collections()
            return True
        except Exception:
            return False

    def _check_redis_connection(self) -> bool:
        try:
            return bool(self.redis_client.ping())
        except Exception:
            return False
