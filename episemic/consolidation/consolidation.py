"""Consolidation engine for transferring memories from hippocampus to cortex."""

from datetime import datetime, timedelta

try:
    from ..cortex import Cortex
except ImportError:
    Cortex = None

from ..hippocampus import Hippocampus
from ..models import ConsolidationJob, Memory, MemoryStatus, RetentionPolicy


class ConsolidationEngine:
    def __init__(self, hippocampus: Hippocampus, cortex: Cortex):
        self.hippocampus = hippocampus
        self.cortex = cortex
        self.consolidation_threshold_hours = 2
        self.consolidation_access_threshold = 3

    async def consolidate_memory(self, memory_id: str) -> bool:
        try:
            memory = await self.hippocampus.retrieve_memory(memory_id)
            if not memory:
                print(f"Memory {memory_id} not found in hippocampus")
                return False

            if not self._should_consolidate(memory):
                return False

            success = await self.cortex.store_memory(memory)
            if success:
                print(f"Successfully consolidated memory {memory_id} to cortex")
                return True
            else:
                print(f"Failed to consolidate memory {memory_id}")
                return False

        except Exception as e:
            print(f"Error during consolidation of memory {memory_id}: {e}")
            return False

    def _should_consolidate(self, memory: Memory) -> bool:
        time_threshold = datetime.utcnow() - timedelta(hours=self.consolidation_threshold_hours)

        if memory.retention_policy == RetentionPolicy.EPHEMERAL:
            return False

        if memory.status != MemoryStatus.ACTIVE:
            return False

        if memory.created_at < time_threshold:
            return True

        if memory.access_count >= self.consolidation_access_threshold:
            return True

        return False

    async def run_consolidation_job(self, job: ConsolidationJob) -> ConsolidationJob:
        try:
            consolidated_count = 0
            failed_memories = []

            for memory_id in job.memory_ids:
                success = await self.consolidate_memory(memory_id)
                if success:
                    consolidated_count += 1
                else:
                    failed_memories.append(memory_id)

            if consolidated_count > 0:
                job.status = "completed"
                print(f"Consolidation job {job.id} completed: {consolidated_count} memories consolidated")
            else:
                job.status = "failed"
                job.error_message = f"Failed to consolidate memories: {failed_memories}"

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            print(f"Consolidation job {job.id} failed: {e}")

        return job

    async def auto_consolidation_sweep(self) -> int:
        try:
            # This would need to be implemented to scan hippocampus for eligible memories
            # For now, this is a placeholder that would need hippocampus scanning capability

            print("Auto consolidation sweep completed")
            return 0

        except Exception as e:
            print(f"Error during auto consolidation sweep: {e}")
            return 0

    async def create_consolidated_summary(self, memory_ids: list[str]) -> Memory | None:
        try:
            memories = []
            for memory_id in memory_ids:
                memory = await self.cortex.retrieve_memory(memory_id)
                if memory:
                    memories.append(memory)

            if not memories:
                return None

            # Create a consolidated summary memory
            consolidated_text = self._summarize_memories(memories)
            consolidated_tags = list({tag for memory in memories for tag in memory.tags})

            summary_memory = Memory(
                source="consolidation",
                source_ref=f"consolidated_from_{len(memories)}_memories",
                title=f"Consolidated Summary ({len(memories)} memories)",
                text=consolidated_text,
                summary=self._create_meta_summary(memories),
                tags=consolidated_tags,
                retention_policy=RetentionPolicy.ARCHIVAL,
                metadata={
                    "consolidated_from": memory_ids,
                    "consolidation_timestamp": datetime.utcnow().isoformat(),
                    "original_memory_count": len(memories)
                }
            )

            # Store the consolidated memory in cortex
            success = await self.cortex.store_memory(summary_memory)
            if success:
                return summary_memory

            return None

        except Exception as e:
            print(f"Error creating consolidated summary: {e}")
            return None

    def _summarize_memories(self, memories: list[Memory]) -> str:
        # Simple concatenation - in practice, this would use an LLM
        combined_text = "\n\n".join([f"Memory {i+1}: {memory.summary}" for i, memory in enumerate(memories)])
        return f"Consolidated summary of {len(memories)} related memories:\n\n{combined_text}"

    def _create_meta_summary(self, memories: list[Memory]) -> str:
        topics = set()
        sources = set()

        for memory in memories:
            topics.update(memory.tags)
            sources.add(memory.source)

        return f"Meta-summary covering topics: {', '.join(topics)} from sources: {', '.join(sources)}"

    def health_check(self) -> dict:
        return {
            "hippocampus_healthy": self.hippocampus.health_check(),
            "cortex_healthy": self.cortex.health_check(),
        }
