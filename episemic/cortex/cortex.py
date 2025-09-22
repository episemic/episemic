"""Cortex implementation - Durable, richly-linked, relational archive of memories."""

import json

import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor

from ..models import Memory, MemoryLink


class Cortex:
    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "episemic",
        db_user: str = "postgres",
        db_password: str = "postgres",
    ):
        self.connection_params = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user,
            "password": db_password,
        }
        self._setup_database()

    def _get_connection(self):
        return psycopg2.connect(**self.connection_params)

    def _setup_database(self):
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id VARCHAR(36) PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        source VARCHAR(255) NOT NULL,
                        source_ref TEXT,
                        title TEXT NOT NULL,
                        text TEXT NOT NULL,
                        summary TEXT,
                        hash VARCHAR(64) NOT NULL,
                        version INTEGER DEFAULT 1,
                        tags TEXT[] DEFAULT '{}',
                        access_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP,
                        retention_policy VARCHAR(20) DEFAULT 'default',
                        status VARCHAR(20) DEFAULT 'active',
                        checksum_status VARCHAR(20) DEFAULT 'unknown',
                        metadata JSONB DEFAULT '{}',
                        embedding_v1 FLOAT8[],
                        embedding_v2 FLOAT8[]
                    );
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memory_links (
                        id SERIAL PRIMARY KEY,
                        source_id VARCHAR(36) REFERENCES memories(id),
                        target_id VARCHAR(36) REFERENCES memories(id),
                        link_type VARCHAR(50) NOT NULL,
                        weight FLOAT DEFAULT 0.5,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
                    CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags);
                    CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
                    CREATE INDEX IF NOT EXISTS idx_memory_links_source ON memory_links(source_id);
                    CREATE INDEX IF NOT EXISTS idx_memory_links_target ON memory_links(target_id);
                """)

                conn.commit()

    async def store_memory(self, memory: Memory) -> bool:
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO memories (
                            id, created_at, ingested_at, source, source_ref, title, text,
                            summary, hash, version, tags, access_count, last_accessed,
                            retention_policy, status, checksum_status, metadata,
                            embedding_v1, embedding_v2
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            text = EXCLUDED.text,
                            summary = EXCLUDED.summary,
                            hash = EXCLUDED.hash,
                            version = EXCLUDED.version + 1,
                            tags = EXCLUDED.tags,
                            metadata = EXCLUDED.metadata,
                            embedding_v1 = EXCLUDED.embedding_v1,
                            embedding_v2 = EXCLUDED.embedding_v2
                    """, (
                        memory.id, memory.created_at, memory.ingested_at,
                        memory.source, memory.source_ref, memory.title, memory.text,
                        memory.summary, memory.hash, memory.version, memory.tags,
                        memory.access_count, memory.last_accessed,
                        memory.retention_policy.value, memory.status.value,
                        memory.checksum_status.value, json.dumps(memory.metadata),
                        memory.embedding_v1, memory.embedding_v2
                    ))

                    for link in memory.links:
                        cursor.execute("""
                            INSERT INTO memory_links (source_id, target_id, link_type, weight)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (memory.id, link.target_id, link.type.value, link.weight))

                    conn.commit()
                    return True

        except Exception as e:
            print(f"Error storing memory in Cortex: {e}")
            return False

    async def retrieve_memory(self, memory_id: str) -> Memory | None:
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT * FROM memories WHERE id = %s AND status != 'deleted'
                    """, (memory_id,))

                    row = cursor.fetchone()
                    if not row:
                        return None

                    cursor.execute("""
                        SELECT target_id, link_type, weight
                        FROM memory_links
                        WHERE source_id = %s
                    """, (memory_id,))

                    links = [
                        MemoryLink(
                            target_id=link_row["target_id"],
                            type=link_row["link_type"],
                            weight=link_row["weight"]
                        )
                        for link_row in cursor.fetchall()
                    ]

                    memory_data = dict(row)
                    memory_data["links"] = links
                    return Memory.model_validate(memory_data)

        except Exception as e:
            print(f"Error retrieving memory from Cortex: {e}")
            return None

    async def search_by_tags(self, tags: list[str], limit: int = 10) -> list[Memory]:
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT * FROM memories
                        WHERE tags && %s AND status = 'active'
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (tags, limit))

                    memories = []
                    for row in cursor.fetchall():
                        memory_data = dict(row)
                        memory_data["links"] = []  # Would need separate query for links
                        memories.append(Memory.model_validate(memory_data))

                    return memories

        except Exception as e:
            print(f"Error searching by tags: {e}")
            return []

    async def get_memory_graph(self, memory_id: str, depth: int = 2) -> dict:
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        WITH RECURSIVE memory_graph AS (
                            SELECT id, title, 0 as depth
                            FROM memories
                            WHERE id = %s

                            UNION ALL

                            SELECT m.id, m.title, mg.depth + 1
                            FROM memories m
                            JOIN memory_links ml ON m.id = ml.target_id
                            JOIN memory_graph mg ON ml.source_id = mg.id
                            WHERE mg.depth < %s
                        )
                        SELECT * FROM memory_graph
                    """, (memory_id, depth))

                    nodes = cursor.fetchall()

                    cursor.execute("""
                        SELECT ml.source_id, ml.target_id, ml.link_type, ml.weight
                        FROM memory_links ml
                        WHERE ml.source_id = ANY(%s) OR ml.target_id = ANY(%s)
                    """, ([node["id"] for node in nodes], [node["id"] for node in nodes]))

                    edges = cursor.fetchall()

                    return {
                        "nodes": [dict(node) for node in nodes],
                        "edges": [dict(edge) for edge in edges]
                    }

        except Exception as e:
            print(f"Error getting memory graph: {e}")
            return {"nodes": [], "edges": []}

    async def increment_access_count(self, memory_id: str):
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE memories
                        SET access_count = access_count + 1,
                            last_accessed = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (memory_id,))
                    conn.commit()

        except Exception as e:
            print(f"Error incrementing access count: {e}")

    async def mark_deleted(self, memory_id: str):
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE memories
                        SET status = 'deleted'
                        WHERE id = %s
                    """, (memory_id,))
                    conn.commit()

        except Exception as e:
            print(f"Error marking memory as deleted: {e}")

    def health_check(self) -> bool:
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return cursor.fetchone() is not None
        except Exception:
            return False
