# Episemic — hippocampus + cortex memory architecture (conceptual blueprint)

Nice — here’s a practical, brain-inspired memory architecture for **Episemic** that explicitly avoids the “failure modes” we learned from Alzheimer’s. I’ll give: components, data models, ingestion & retrieval pipelines, redundancy & integrity, decay/forgetting policies, monitoring & tests, tech choices (fits your Qdrant + Postgres notes), and an actionable roadmap + example code snippets.

---

# 1 — High level design (analogy)

* **Hippocampus (Fast layer)**: fast, writable, vector-indexed store for *recent* memories and short-term context — low-latency lookups for composing answers. (Qdrant + Redis cache)
* **Cortex (Slow / consolidated layer)**: durable, richly-linked, relational archive of memories — long-term recall and durable semantics. (Postgres + object store)
* **Synaptic graph (Relational graph layer)**: explicit graph of relations between memories (edges: context, person, project, derived-from, source). (Postgres with graph extension or a graph DB overlay)
* **Proteostasis & Integrity engine**: periodic integrity checks, hashes, versioning, automated repair (reindexing, multi-path links) to prevent corruption.
* **Controller / Orchestrator**: service that manages ingestion, consolidation (hippocampus → cortex), eviction, and monitoring.
* **Observability & Safety**: metrics, alerts, and self-healing (reindex, rehydrate).

---

# 2 — Key design principles (from Alzheimer’s)

1. **Redundancy of retrieval paths** — each memory is reachable by timestamp, embedding, semantic tags, links.
2. **Explicit linking** — memories degrade less when richly connected.
3. **Integrity-first ingestion** — validate & hash on write; reject or quarantine corrupt inputs.
4. **Separation of concerns** — fast writes vs durable consolidation pipelines.
5. **Graceful decay instead of catastrophic deletion** — soft-delete + staged GC + retention policies.
6. **Monitoring & early detection** — detect “synapse loss” (missing links / retrieval failures) and auto-repair.

---

# 3 — Data model (conceptual)

Memory object (single canonical record):

```json
{
  "id": "uuid",
  "created_at": "2025-09-18T...",
  "ingested_at": "2025-09-18T...",
  "source": "rss|web|user|note",
  "source_ref": "url or feed-id",
  "title": "string",
  "text": "full text / markdown",
  "summary": "short extract",
  "embedding_v1": [float,...],        // vector for Qdrant
  "embedding_v2": [float,...],        // alternate model for redundancy
  "hash": "sha256(...)",              // integrity
  "version": 1,
  "links": [
     {"target_id":"uuid", "type":"cites|context|person|project", "weight": 0.9}
  ],
  "tags": ["AI","research"],
  "access_count": 12,
  "last_accessed": "2025-09-18T...",
  "retention_policy": "ephemeral|default|archival",
  "status": "active|quarantined|deleted",
  "checksum_status": "ok|mismatch|unknown"
}
```

* Store canonical record in **Postgres** (JSONB for text + meta).
* Store vectors in **Qdrant**, keyed by `id` and duplicated embeddings for model-robustness.
* Object files (original attachments) in S3-compatible store, referenced by `source_ref`.

---

# 4 — Ingestion pipeline (step-by-step)

1. **Pre-validate**: dedupe by source + hash. If near-duplicate (fuzzy match), create `alias` link instead of new memory.
2. **Canonicalize**: normalize text (strip noise), create `summary`.
3. **Compute multi-model embeddings**: produce `embedding_v1`, `embedding_v2` (different models).
4. **Compute hash & signature**: `sha256(text + summary + meta)`.
5. **Write to Hippocampus (fast)**:

   * Insert in Postgres `memories_tmp` (low-latency table).
   * Upsert embedding in Qdrant with metadata.
6. **Link extraction**: NLP passes to extract mentions/entities/projects — create edges.
7. **Integrity check**: verify write success to both Postgres and Qdrant — if mismatch, mark `quarantined`.
8. **Consolidation job**: asynchronously (but executed by orchestrator immediately as job) move stable items to Cortex (main Postgres + graph) after N confirmations / time window.
9. **Index & backups**: scheduled multi-region backups; snapshot vector store and DB.

(Orchestrator must guarantee idempotency — use write tokens and optimistic concurrency.)

---

# 5 — Retrieval strategies

1. **Multi-path retrieval**:

   * Query Qdrant by vector similarity (embedding\_v1); get top-K candidates.
   * Cross-check Qdrant results with Postgres links (e.g., if query includes project tag, filter).
   * If confidence low or mismatch between embedding models, fallback to embedding\_v2 and semantic tag search.
2. **Contextual composition**:

   * Maintain short-term session context in fast cache (Redis). Compose retrieval using recency + semantic + graph hops (e.g., return memories within 1–2 hops of last read items).
3. **Integrity-aware ranking**:

   * Penalize items with integrity warnings or low access\_count.
4. **Explainability**:

   * Return provenance (source\_ref, created\_at, link path) with each retrieved memory.

---

# 6 — Redundancy & integrity measures

* **Dual embeddings**: store two different embedding vector variants (different models or random seeds). If one vector is corrupted or model becomes deprecated, the other can serve.
* **Cross-check on read**: when reading, compare stored hash to recomputed hash; if mismatch → repair from backup or fetch original source and reingest.
* **Replication**: Qdrant + Postgres replicated across zones. Regular snapshot of vectors and DB.
* **Link redundancy**: create at least 2 unique link routes to each memory: semantic tag + explicit graph edge + chronological index.
* **Versioning**: every memory update creates immutably-stored version; current pointer references latest.
* **Quarantine & healing**:

  * On integrity failure, mark memory `quarantined` and trigger `repair` job: re-fetch source or re-generate embedding from stored raw; if not possible, mark `degraded` with reason.

---

# 7 — Decay, forgetting, and retention design

* **Three-tier retention**:

  1. **Ephemeral (hippocampal)**: kept for 7–30 days for quick context; high mutation rate.
  2. **Default**: kept in cortex for months; access-driven promotion/demotion.
  3. **Archival**: cold storage for long-term (years); compressed representations.
* **Soft-deletion + GC**:

  * Soft-delete flips `status` to `deleted` and disables retrieval.
  * After retention window, GC physically removes, but keep hashed backups for N days.
* **Controlled decay**:

  * Use decay curves (Ebbinghaus-like or customized) to lower ranking over time but preserve high-linked memories.
  * Promote memories on access (reinforcement learning / simple rule: +access\_count, bump to default/archival if > threshold).
* **Knowledge consolidation job**:

  * Periodic process that consolidates episodic items into generalizable summaries (like hippocampus→cortex consolidation). The consolidation outputs a new long-term `memory` that cites original items.

---

# 8 — Monitoring, checks & tests

* **Metrics**:

  * ingestion\_success\_rate, qdrant\_write\_latency, qdrant\_vector\_count, postgresql\_integrity\_errors, retrieval\_confidence\_score, quarantine\_count.
* **Health checks**:

  * Endpoint that validates: sample fetch from Postgres, sample vector similarity, recompute hash for a random sample.
* **Automated tests**:

  * **Canary reads**: regularly run synthetic queries; expected results must include anchors.
  * **Round-trip test**: ingest test doc → retrieve it via vector search and tag search.
  * **Link consistency test**: for N random items, ensure their links exist and are reachable within 3 hops.
* **Alerts**:

  * High quarantine rate → alert ops.
  * Drop in retrieval confidence → alert and trigger reindex job.
* **Auditing**:

  * Immutable activity log for ingestions/repairs (append-only).

---

# 9 — Failure modes & mitigations (explicit)

1. **Vector store corruption**

   * Mitigation: dual embeddings, snapshots, read-time cross-checks.
2. **Link loss (synapse loss)**

   * Mitigation: automated graph rebuild from NLP entity extraction; create alternate tag-based paths.
3. **Bad ingestion (garbage in)**

   * Mitigation: pre-validate & quarantine noisy inputs, human review workflow.
4. **Stale model drift**

   * Mitigation: keep multiple embedding models; re-embed on-demand; tag model version.
5. **Catastrophic deletion**

   * Mitigation: multi-stage soft-delete and backup retention; required multi-party un-delete for important items.

---

# 10 — APIs (minimal)

POST /memories

* Request: raw text + source metadata
* Response: memory id, status

GET /memories/{id}

* Response: canonical record + provenance

POST /search

* Body: { "query": "text", "top\_k": 10, "filters": {"tags":\[], "projects":\[]} }
* Response: ranked memories + provenance + retrieval confidence

POST /consolidate/\:job

* Trigger immediate consolidation or use scheduler

GET /health

* Returns health checks and metrics

---

# 11 — Example Python-ish pseudocode (ingest + integrity)

```python
import hashlib, uuid, time
from typing import Dict

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def ingest(raw_text: str, meta: Dict):
    mid = str(uuid.uuid4())
    summary = summarize(raw_text)                # your summarizer
    text_norm = normalize(raw_text)
    h = sha256_text(text_norm + summary)
    emb1 = embed_model_v1(text_norm)
    emb2 = embed_model_v2(text_norm)

    # write to Postgres temp (hippocampus)
    pg_insert_temp(mid, {
        "text": text_norm,
        "summary": summary,
        "hash": h,
        "meta": meta,
        "created_at": now_iso(),
    })

    # write vectors to Qdrant (idempotent)
    qdrant_upsert(mid, vector=emb1, metadata={"model":"v1"})
    qdrant_upsert(mid+"_v2", vector=emb2, metadata={"model":"v2"})

    # verify writes
    if not verify_postgres(mid) or not verify_qdrant(mid):
        mark_quarantined(mid)
        alert("ingest_write_failed", {"id": mid})
        return {"id":mid, "status":"quarantined"}

    schedule_consolidation(mid)  # orchestrator handles moving to cortex
    return {"id": mid, "status": "accepted"}
```

---

# 12 — Tech stack suggestions (fits your project)

* Language: **Python** (you’re already there)
* Vector DB: **Qdrant** (primary fast retrieval)
* Relational DB: **Postgres** (canonical memory store + JSONB)
* Cache / Session store: **Redis**
* Object storage: **S3 / MinIO**
* Orchestration: **Celery / Prefect / Dagster** for consolidation jobs
* Graph queries: Postgres + **pggraph**/extension or add **Neo4j** if needed
* Monitoring: **Prometheus + Grafana**
* Backups: scheduled Postgres + Qdrant snapshots, stored in S3
* Embeddings: expose multiple model endpoints (local and cloud) with versioning

---

# 13 — Roadmap (practical milestones)

1. **Week 0–1** — Minimal hippocampus prototype

   * Ingest pipeline: Postgres temp table + Qdrant insert + integrity check
   * Basic API: ingest, retrieve by id, vector search

2. **Week 2–3** — Consolidation + graph

   * Consolidation worker that moves stable memories to main Postgres
   * Basic link extraction + store graph edges

3. **Week 4** — Redundancy & health

   * Dual embeddings, hash verification, quarantine logic
   * Canary tests + metrics dashboard

4. **Week 5–6** — Decay policies & consolidation summaries

   * Implement retention tiers, consolidation job producing compressed long-term memories

5. **Week 7+** — Robustness & scaling

   * Snapshots, distributed Qdrant, re-embedding on model upgrade, UX for human review of quarantined items

---

# 14 — Minimal experimental evaluation plan

* Run synthetic dataset (10k notes):

  * Test retrieval recall\@K before/after consolidation.
  * Simulate `synapse loss`: delete random links and measure retrieval drop — test auto-repair algorithm.
  * Corrupt random memory hashes and test quarantine + healing pipeline.
* Measure: ingestion\_latency, retrieval\_latency, retrieval\_precision, quarantine\_rate, successful\_repair\_rate.

---

# 15 — Quick checklist to avoid Alzheimer-like failure

* [x] Multiple retrieval paths per memory
* [x] Versioned, hashed storage + re-check on read
* [x] Rich semantic links and graph
* [x] Soft-delete + staged GC
* [x] Consolidation job (short → long term)
* [x] Observability + automated repair

---
