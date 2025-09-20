"""Data models for the Episemic memory system."""

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    DELETED = "deleted"


class ChecksumStatus(str, Enum):
    OK = "ok"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class RetentionPolicy(str, Enum):
    EPHEMERAL = "ephemeral"
    DEFAULT = "default"
    ARCHIVAL = "archival"


class LinkType(str, Enum):
    CITES = "cites"
    CONTEXT = "context"
    PERSON = "person"
    PROJECT = "project"
    DERIVED_FROM = "derived_from"
    SOURCE = "source"


class MemoryLink(BaseModel):
    target_id: str
    type: LinkType
    weight: float = Field(ge=0.0, le=1.0, default=0.5)


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    source: str
    source_ref: str | None = None
    title: str
    text: str
    summary: str
    embedding_v1: list[float] | None = None
    embedding_v2: list[float] | None = None
    hash: str = ""
    version: int = 1
    links: list[MemoryLink] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    access_count: int = 0
    last_accessed: datetime | None = None
    retention_policy: RetentionPolicy = RetentionPolicy.DEFAULT
    status: MemoryStatus = MemoryStatus.ACTIVE
    checksum_status: ChecksumStatus = ChecksumStatus.UNKNOWN
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:
        if not self.hash:
            self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        content = f"{self.text}{self.summary}{self.source}{self.source_ref or ''}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        return self.hash == self.compute_hash()

    def increment_access(self):
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class SearchQuery(BaseModel):
    query: str
    top_k: int = 10
    filters: dict[str, Any] = Field(default_factory=dict)
    include_quarantined: bool = False


class SearchResult(BaseModel):
    memory: Memory
    score: float
    provenance: dict[str, Any] = Field(default_factory=dict)
    retrieval_path: list[str] = Field(default_factory=list)


class ConsolidationJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    memory_ids: list[str]
    job_type: str = "consolidation"
    status: str = "pending"
    result_memory_id: str | None = None
    error_message: str | None = None
