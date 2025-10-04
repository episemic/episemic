# SPEC.md - EPISEMIC-II Extensions

**Research Implementation Specification**  
**Project**: Extending EPISEMIC with Intentional and Interactional Memory Dimensions  
**Version**: 1.0  
**Date**: October 2025

---

## 1. Overview

This specification describes the implementation of **EPISEMIC-II** extensions to the existing EPISEMIC framework. The goal is to add **intention tracking** (goals, motivations, constraints) and **interaction pattern modeling** (communication styles, engagement signals) as first-class memory primitives while maintaining backward compatibility with the current system.

**Research Hypothesis**: Agents with explicit intention and interaction memory will demonstrate measurable improvements in goal alignment, proactive assistance, and user satisfaction compared to fact-only memory systems.

**Key Design Principles**:
- Extend, don't replace - all existing APIs remain unchanged
- Maintain dual-backend support (DuckDB + Qdrant/PostgreSQL)
- Enable/disable via configuration flags for A/B testing
- Research-first: instrumentation for evaluation built-in

---

## 2. Architecture Overview

### 2.1 Existing System (Keep Intact)

```
Current EPISEMIC:
├── Hippocampus (episodic memory - recent experiences)
│   ├── DuckDB backend (default, zero-config)
│   └── Qdrant + Redis backend (production)
├── Cortex (semantic memory - consolidated knowledge)
│   ├── DuckDB backend (default)
│   └── PostgreSQL backend (production)
├── Consolidation Engine (hippocampus → cortex)
└── Retrieval Engine (multi-strategy search)
```

### 2.2 New Extensions (EPISEMIC-II)

```
EPISEMIC-II Extensions:
├── Intention Store (new component)
│   ├── Intention extraction from conversations
│   ├── Graph storage (goals, motivations, constraints)
│   ├── Relationship tracking (enables, blocks, requires, conflicts_with)
│   └── Temporal state management (active, dormant, completed, abandoned)
├── Interaction Store (new component)
│   ├── Pattern extraction from message sequences
│   ├── Communication style vectors
│   ├── Engagement signal tracking
│   └── Mode transition detection
├── Enhanced Retrieval (extend existing)
│   ├── Intention-aware memory ranking
│   ├── Goal-context filtering
│   └── Interaction-adapted responses
└── Evaluation Framework (new)
    ├── Ground truth annotation tools
    ├── Inference quality metrics
    ├── A/B testing infrastructure
    └── Performance benchmarking
```

---

## 3. Data Models

### 3.1 Intention Model

**File**: `episemic/models.py` (extend existing)

```python
from enum import Enum
from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel, Field

class IntentionType(str, Enum):
    """Types of intentions"""
    GOAL = "goal"                    # "I want to learn Rust"
    MOTIVATION = "motivation"        # "because it's fast and safe"
    CONSTRAINT = "constraint"        # "must finish by Friday"
    PLAN = "plan"                    # "will study 2 hours daily"
    PREFERENCE = "preference"        # "prefer hands-on examples"

class IntentionStatus(str, Enum):
    """Lifecycle status of intentions"""
    ACTIVE = "active"                # Currently pursuing
    DORMANT = "dormant"              # Paused but may resume
    COMPLETED = "completed"          # Successfully achieved
    ABANDONED = "abandoned"          # No longer pursuing

class Intention(BaseModel):
    """Intention memory record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: IntentionType
    content: str                     # Natural language description
    confidence: float = 1.0          # System confidence (0.0-1.0)
    status: IntentionStatus = IntentionStatus.ACTIVE
    
    # Temporal tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Context
    extracted_from_memory_id: Optional[str] = None  # Source memory
    related_entity_ids: List[str] = Field(default_factory=list)
    
    # User/agent association
    user_id: Optional[str] = None
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=dict)

class IntentionRelationship(BaseModel):
    """Relationships between intentions"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_intention_id: str
    target_intention_id: str
    relationship_type: str           # enables, blocks, requires, conflicts_with
    weight: float = 1.0              # Strength of relationship
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)
```

### 3.2 Interaction Model

**File**: `episemic/models.py` (extend existing)

```python
class InteractionPattern(BaseModel):
    """Interaction pattern record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    
    # Communication style (vector representation)
    detail_preference: float = 0.0   # -1 (concise) to +1 (detailed)
    technicality: float = 0.5        # 0 (layperson) to 1 (expert)
    formality: float = 0.5           # 0 (casual) to 1 (formal)
    
    # Engagement metrics
    avg_message_length: float = 0.0
    follow_up_rate: float = 0.0      # avg follow-ups per topic
    topic_persistence: float = 0.0   # avg turns on same topic
    question_frequency: float = 0.0
    
    # Mode tracking
    current_mode: str = "exploration"  # exploration, implementation, evaluation
    mode_history: List[Dict] = Field(default_factory=list)  # [{mode, timestamp, trigger}]
    
    # Temporal
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    sample_count: int = 0            # Number of interactions observed
    
    # Metadata
    metadata: Dict = Field(default_factory=dict)

class InteractionSession(BaseModel):
    """Single interaction session snapshot"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    
    # Session metrics
    message_count: int = 0
    topics_discussed: List[str] = Field(default_factory=list)
    avg_response_time: float = 0.0
    engagement_score: float = 0.0
    
    # Detected patterns
    detected_mode: str = "exploration"
    style_vector: Dict[str, float] = Field(default_factory=dict)
    
    # Memory associations
    memory_ids_accessed: List[str] = Field(default_factory=list)
    intentions_referenced: List[str] = Field(default_factory=list)
```

---

## 4. Component Implementation

### 4.1 Intention Store

**New File**: `episemic/intention/intention_store.py`

**Responsibilities**:
- Extract intentions from conversation/memory text
- Store intentions with relationships
- Manage intention lifecycle (active → dormant → completed/abandoned)
- Query intentions by type, status, user, etc.
- Detect intention conflicts

**Key Methods**:
```python
class IntentionStore:
    async def extract_intentions(
        self, 
        text: str, 
        memory_id: str,
        context: Optional[str] = None
    ) -> List[Intention]
    
    async def store_intention(self, intention: Intention) -> str
    
    async def add_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        rel_type: str
    ) -> IntentionRelationship
    
    async def get_intentions(
        self, 
        user_id: Optional[str] = None,
        status: Optional[IntentionStatus] = None,
        type: Optional[IntentionType] = None
    ) -> List[Intention]
    
    async def get_related_intentions(
        self, 
        intention_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2
    ) -> List[Tuple[Intention, str, int]]  # (intention, rel_type, depth)
    
    async def detect_conflicts(
        self, 
        intention_id: str
    ) -> List[Tuple[Intention, str]]  # (conflicting_intention, reason)
    
    async def update_status(
        self, 
        intention_id: str, 
        new_status: IntentionStatus
    ) -> bool
```

**Backend Support**:
- **DuckDB**: Single table with JSONB columns for graph relationships
- **PostgreSQL**: Separate tables for intentions and relationships with proper foreign keys

### 4.2 Interaction Store

**New File**: `episemic/interaction/interaction_store.py`

**Responsibilities**:
- Extract interaction patterns from message sequences
- Compute and update user interaction profiles
- Detect mode transitions
- Track engagement metrics

**Key Methods**:
```python
class InteractionStore:
    async def analyze_interaction(
        self,
        user_id: str,
        messages: List[Dict],  # [{role, content, timestamp}]
        session_id: Optional[str] = None
    ) -> InteractionSession
    
    async def update_user_pattern(
        self,
        user_id: str,
        session: InteractionSession
    ) -> InteractionPattern
    
    async def get_user_pattern(
        self,
        user_id: str
    ) -> Optional[InteractionPattern]
    
    async def detect_mode_transition(
        self,
        user_id: str,
        current_session: InteractionSession,
        previous_pattern: Optional[InteractionPattern] = None
    ) -> Optional[Dict]  # {from_mode, to_mode, trigger, confidence}
    
    async def get_sessions(
        self,
        user_id: str,
        since: Optional[datetime] = None,
        limit: int = 10
    ) -> List[InteractionSession]
```

### 4.3 Intention Extraction Engine

**New File**: `episemic/intention/extractor.py`

**Uses LLM to extract intentions from text**

```python
class IntentionExtractor:
    def __init__(self, llm_provider: str = "openai", model: str = "gpt-4o-mini"):
        self.llm_provider = llm_provider
        self.model = model
    
    async def extract(
        self,
        text: str,
        context: Optional[str] = None
    ) -> List[Dict]:  # [{type, content, confidence}]
        """
        Extract intentions using LLM with structured output.
        
        Prompt strategy:
        1. Identify explicit goals ("I want to...", "My goal is...")
        2. Identify implicit goals (infer from context)
        3. Identify motivations ("because...", "the reason is...")
        4. Identify constraints ("must by...", "can't if...")
        5. Identify plans ("I will...", "planning to...")
        6. Assign confidence scores
        """
```

**Prompt Template** (in `episemic/intention/prompts.py`):
```python
INTENTION_EXTRACTION_PROMPT = """
Analyze the following text and extract any intentions (goals, motivations, constraints, plans, preferences).

Text: {text}
{context_section}

For each intention found, identify:
1. Type: goal, motivation, constraint, plan, or preference
2. Content: natural language description
3. Confidence: your certainty (0.0-1.0)

Return as JSON array:
[
  {{"type": "goal", "content": "learn Rust programming", "confidence": 0.9}},
  {{"type": "motivation", "content": "wants memory-safe systems programming", "confidence": 0.7}}
]

Important:
- Be conservative with confidence scores
- Only extract clear intentions, not vague statements
- Distinguish between goals (desired outcomes) and plans (actions to achieve goals)
- Mark anything uncertain with confidence < 0.7
"""
```

### 4.4 Interaction Analyzer

**New File**: `episemic/interaction/analyzer.py`

**Analyzes message sequences to extract patterns**

```python
class InteractionAnalyzer:
    def compute_style_vector(
        self,
        messages: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute communication style from messages.
        
        Metrics:
        - detail_preference: avg message length relative to median
        - technicality: technical term density
        - formality: formal language markers
        """
    
    def compute_engagement_metrics(
        self,
        messages: List[Dict],
        topics: List[str]
    ) -> Dict[str, float]:
        """
        Compute engagement signals.
        
        Metrics:
        - follow_up_rate: (follow-up questions) / (total messages)
        - topic_persistence: avg consecutive messages on same topic
        - question_frequency: questions / total messages
        """
    
    def detect_mode(
        self,
        messages: List[Dict],
        previous_mode: Optional[str] = None
    ) -> str:
        """
        Detect interaction mode: exploration, implementation, evaluation.
        
        Signals:
        - exploration: questions, "what", "how", "why"
        - implementation: "create", "build", "implement", code requests
        - evaluation: "compare", "analyze", "assess", critique requests
        """
```

---

## 5. Integration with Existing Components

### 5.1 Enhanced EpistemicAPI

**File**: `episemic/api.py` (extend existing)

**Add new methods**:
```python
class EpistemicAPI:
    def __init__(self, config: EpistemicConfig):
        # ... existing init ...
        
        # New components (only if enabled in config)
        if config.enable_intentions:
            self.intention_store = IntentionStore(config)
            self.intention_extractor = IntentionExtractor(config)
        
        if config.enable_interactions:
            self.interaction_store = InteractionStore(config)
            self.interaction_analyzer = InteractionAnalyzer(config)
    
    # New intention methods
    async def extract_intentions(
        self,
        memory_id: str,
        text: Optional[str] = None
    ) -> List[Intention]:
        """Extract intentions from a memory"""
    
    async def get_user_intentions(
        self,
        user_id: str,
        status: Optional[IntentionStatus] = None
    ) -> List[Intention]:
        """Get all intentions for a user"""
    
    async def add_intention(
        self,
        intention: Intention
    ) -> str:
        """Manually add an intention"""
    
    async def update_intention_status(
        self,
        intention_id: str,
        new_status: IntentionStatus
    ) -> bool:
        """Update intention status"""
    
    # New interaction methods
    async def analyze_session(
        self,
        user_id: str,
        messages: List[Dict]
    ) -> InteractionSession:
        """Analyze an interaction session"""
    
    async def get_user_interaction_pattern(
        self,
        user_id: str
    ) -> Optional[InteractionPattern]:
        """Get user's interaction profile"""
    
    # Enhanced retrieval (modify existing)
    async def recall(
        self,
        query: str,
        limit: int = 10,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        use_intention_context: bool = True,  # NEW
        use_interaction_adaptation: bool = True  # NEW
    ) -> List[SearchResult]:
        """Enhanced recall with intention/interaction awareness"""
```

### 5.2 Enhanced Retrieval Logic

**File**: `episemic/retrieval/retrieval.py` (extend existing)

**Add intention-aware ranking**:
```python
async def _rank_with_intentions(
    self,
    results: List[SearchResult],
    user_id: str,
    query: str
) -> List[SearchResult]:
    """
    Re-rank results based on user's active intentions.
    
    Strategy:
    1. Get user's active intentions
    2. For each result, compute intention-relevance score
    3. Boost results that relate to active goals
    4. Penalize results that conflict with constraints
    5. Combine with original semantic score
    """
    
    # Get active intentions
    intentions = await self.intention_store.get_intentions(
        user_id=user_id,
        status=IntentionStatus.ACTIVE
    )
    
    if not intentions:
        return results  # No re-ranking needed
    
    # Re-rank
    for result in results:
        intention_score = 0.0
        
        for intention in intentions:
            # Check if memory relates to intention
            relatedness = await self._compute_intention_relatedness(
                result.memory,
                intention
            )
            intention_score += relatedness
        
        # Combine scores (60% semantic, 40% intention)
        result.score = 0.6 * result.score + 0.4 * intention_score
    
    # Re-sort
    results.sort(key=lambda r: r.score, reverse=True)
    return results
```

**Add interaction-adapted formatting**:
```python
async def _adapt_to_interaction_style(
    self,
    results: List[SearchResult],
    user_id: str
) -> List[SearchResult]:
    """
    Adapt result presentation to user's interaction preferences.
    
    Note: This might just add metadata to results for the LLM to use,
    rather than modifying the results themselves.
    """
    
    pattern = await self.interaction_store.get_user_pattern(user_id)
    if not pattern:
        return results
    
    # Add interaction hints to metadata
    for result in results:
        result.memory.metadata['interaction_hints'] = {
            'detail_level': 'detailed' if pattern.detail_preference > 0.5 else 'concise',
            'technicality': 'technical' if pattern.technicality > 0.7 else 'accessible',
            'include_examples': pattern.metadata.get('prefers_examples', False)
        }
    
    return results
```

---

## 6. Configuration

### 6.1 Extended EpistemicConfig

**File**: `episemic/config.py` (extend existing)

```python
class EpistemicConfig(BaseModel):
    # ... existing config fields ...
    
    # New: Enable/disable research features
    enable_intentions: bool = False
    enable_interactions: bool = False
    
    # Intention extraction config
    intention_llm_provider: str = "openai"
    intention_llm_model: str = "gpt-4o-mini"
    intention_extraction_threshold: float = 0.7  # Min confidence
    intention_auto_extract: bool = True  # Extract on remember()
    
    # Interaction tracking config
    interaction_min_messages: int = 3  # Min messages to compute pattern
    interaction_mode_detection: bool = True
    
    # Research/evaluation config
    enable_research_metrics: bool = False  # Extra instrumentation
    log_extractions: bool = False  # Log all extractions for analysis
```

### 6.2 Environment Variables

Add support for:
```bash
EPISEMIC_ENABLE_INTENTIONS=true
EPISEMIC_ENABLE_INTERACTIONS=true
EPISEMIC_INTENTION_LLM_MODEL=gpt-4o-mini
EPISEMIC_ENABLE_RESEARCH_METRICS=true
```

---

## 7. Database Schema

### 7.1 DuckDB Schema

**File**: `episemic/hippocampus/duckdb_schema.py` (extend)

```sql
-- Intentions table
CREATE TABLE IF NOT EXISTS intentions (
    id VARCHAR PRIMARY KEY,
    type VARCHAR NOT NULL,
    content TEXT NOT NULL,
    confidence REAL NOT NULL,
    status VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL,
    last_accessed TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    extracted_from_memory_id VARCHAR,
    user_id VARCHAR,
    tags JSON,
    metadata JSON
);

-- Intention relationships table
CREATE TABLE IF NOT EXISTS intention_relationships (
    id VARCHAR PRIMARY KEY,
    source_intention_id VARCHAR NOT NULL,
    target_intention_id VARCHAR NOT NULL,
    relationship_type VARCHAR NOT NULL,
    weight REAL NOT NULL,
    created_at TIMESTAMP NOT NULL,
    metadata JSON,
    FOREIGN KEY (source_intention_id) REFERENCES intentions(id),
    FOREIGN KEY (target_intention_id) REFERENCES intentions(id)
);

-- Interaction patterns table
CREATE TABLE IF NOT EXISTS interaction_patterns (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL UNIQUE,
    detail_preference REAL,
    technicality REAL,
    formality REAL,
    avg_message_length REAL,
    follow_up_rate REAL,
    topic_persistence REAL,
    question_frequency REAL,
    current_mode VARCHAR,
    mode_history JSON,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    sample_count INTEGER,
    metadata JSON
);

-- Interaction sessions table
CREATE TABLE IF NOT EXISTS interaction_sessions (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    message_count INTEGER,
    topics_discussed JSON,
    avg_response_time REAL,
    engagement_score REAL,
    detected_mode VARCHAR,
    style_vector JSON,
    memory_ids_accessed JSON,
    intentions_referenced JSON
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_intentions_user_status ON intentions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_intentions_type ON intentions(type);
CREATE INDEX IF NOT EXISTS idx_intention_rels_source ON intention_relationships(source_intention_id);
CREATE INDEX IF NOT EXISTS idx_intention_rels_target ON intention_relationships(target_intention_id);
CREATE INDEX IF NOT EXISTS idx_interaction_patterns_user ON interaction_patterns(user_id);
CREATE INDEX IF NOT EXISTS idx_interaction_sessions_user ON interaction_sessions(user_id);
```

### 7.2 PostgreSQL Schema

**File**: `episemic/cortex/postgres_schema.sql` (extend)

Similar to DuckDB but with proper foreign key constraints and additional indexes for graph queries.

---

## 8. CLI Extensions

### 8.1 New Commands

**File**: `episemic/cli/main.py` (extend)

```python
@app.command()
def intentions(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    show_graph: bool = False
):
    """List user intentions"""
    
@app.command()
def add_intention(
    content: str,
    type: str = "goal",
    user_id: Optional[str] = None
):
    """Manually add an intention"""

@app.command()
def interaction_profile(
    user_id: str
):
    """Show user's interaction profile"""

@app.command()
def analyze_session(
    user_id: str,
    messages_file: str
):
    """Analyze interaction session from JSON file"""

@app.command()
def extract_intentions_batch(
    memory_ids_file: str,
    output_file: str
):
    """Batch extract intentions for research annotation"""
```

---

## 9. Testing & Evaluation

### 9.1 Unit Tests

**New Files**:
- `tests/test_intention_store.py`
- `tests/test_interaction_store.py`
- `tests/test_intention_extraction.py`
- `tests/test_interaction_analysis.py`
- `tests/test_enhanced_retrieval.py`

**Test Coverage**:
- Intention CRUD operations
- Relationship graph operations
- Conflict detection
- Interaction pattern computation
- Mode transition detection
- Intention-aware retrieval ranking
- Integration with existing APIs

### 9.2 Evaluation Framework

**New File**: `episemic/evaluation/framework.py`

```python
class EvaluationFramework:
    """Framework for A/B testing and metrics collection"""
    
    async def run_baseline_vs_enhanced(
        self,
        test_queries: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, float]:
        """
        Run A/B test comparing baseline vs intention-enhanced.
        
        Returns metrics:
        - retrieval_precision
        - retrieval_recall
        - goal_alignment_score
        - user_satisfaction (simulated)
        """
    
    async def collect_extraction_metrics(
        self,
        memory_ids: List[str],
        human_annotations: Dict[str, List[Intention]]
    ) -> Dict[str, float]:
        """
        Evaluate intention extraction quality.
        
        Returns:
        - precision, recall, f1
        - confidence_calibration
        - false_positive_rate
        """
```

### 9.3 Annotation Tools

**New File**: `episemic/evaluation/annotation.py`

```python
class AnnotationTool:
    """Tool for human annotation of intentions"""
    
    def export_for_annotation(
        self,
        memory_ids: List[str],
        output_path: str
    ):
        """Export memories to CSV/JSON for human annotation"""
    
    def import_annotations(
        self,
        annotation_file: str
    ) -> Dict[str, List[Intention]]:
        """Import human-annotated intentions"""
    
    def compute_inter_rater_agreement(
        self,
        annotations: Dict[str, Dict[str, List[Intention]]]
    ) -> float:
        """Compute Krippendorff's alpha"""
```

---

## 10. Implementation Roadmap

### Phase 1: Core Components (Week 1)
- [ ] Implement Intention and Interaction models in `models.py`
- [ ] Create database schemas for DuckDB and PostgreSQL
- [ ] Implement IntentionStore with basic CRUD
- [ ] Implement InteractionStore with basic CRUD
- [ ] Add unit tests for stores

### Phase 2: Extraction & Analysis (Week 2)
- [ ] Implement IntentionExtractor with LLM integration
- [ ] Implement InteractionAnalyzer with pattern detection
- [ ] Add extraction on `remember()` (if enabled)
- [ ] Add unit tests for extraction/analysis

### Phase 3: Enhanced Retrieval (Week 3)
- [ ] Implement intention-aware ranking in retrieval.py
- [ ] Implement interaction-adapted formatting
- [ ] Add intention context to search results
- [ ] Add integration tests

### Phase 4: API & CLI Extensions (Week 4)
- [ ] Add intention methods to EpistemicAPI
- [ ] Add interaction methods to EpistemicAPI
- [ ] Extend CLI with new commands
- [ ] Update documentation

### Phase 5: Evaluation Framework (Week 5-6)
- [ ] Implement annotation tools
- [ ] Implement evaluation framework
- [ ] Create test datasets
- [ ] Run baseline experiments

### Phase 6: Research Validation (Week 7-8)
- [ ] Collect human annotations
- [ ] Run A/B tests
- [ ] Analyze results
- [ ] Document findings

---

## 11. Backward Compatibility

**Critical**: All existing APIs must continue to work without changes.

**Strategy**:
- New features are opt-in via config flags
- Default config has `enable_intentions=False`, `enable_interactions=False`
- All new methods are additions, not modifications
- Existing tests must continue to pass

**Migration Path**:
```python
# Existing code works unchanged
from episemic_core import Episemic

async with Episemic() as episemic:
    memory = await episemic.remember("Hello")
    results = await episemic.recall("Hello")

# New features opt-in
from episemic_core import Episemic, EpistemicConfig

config = EpistemicConfig(
    enable_intentions=True,
    enable_interactions=True
)

async with Episemic(config=config) as episemic:
    # Now with enhanced capabilities
    memory = await episemic.remember("I want to learn Rust")
    
    # Get extracted intentions
    intentions = await episemic.extract_intentions(memory.id)
    
    # Intention-aware recall
    results = await episemic.recall("Rust", user_id="user123")
```

---

## 12. Example Usage

### 12.1 Basic Intention Tracking

```python
from episemic_core import Episemic, EpistemicConfig

config = EpistemicConfig(enable_intentions=True)

async with Episemic(config=config) as episemic:
    # Store memory with automatic intention extraction
    memory = await episemic.remember(
        "I want to build a memory system for AI agents because "
        "current systems don't handle long-term context well. "
        "Must finish prototype by end of quarter.",
        user_id="researcher_001"
    )
    
    # Get extracted intentions
    intentions = await episemic.get_user_intentions("researcher_001")
    
    for intention in intentions:
        print(f"{intention.type}: {intention.content} ({intention.confidence})")
    
    # Output:
    # goal: build a memory system for AI agents (0.95)
    # motivation: current systems don't handle long-term context (0.85)
    # constraint: finish prototype by end of quarter (0.90)
```

### 12.2 Interaction-Adapted Retrieval

```python
from episemic_core import Episemic, EpistemicConfig

config = EpistemicConfig(
    enable_intentions=True,
    enable_interactions=True
)

async with Episemic(config=config) as episemic:
    # Simulate interaction session
    messages = [
        {"role": "user", "content": "How does vector search work?"},
        {"role": "assistant", "content": "...detailed explanation..."},
        {"role": "user", "content": "Can you give me a code example?"},
        {"role": "assistant", "content": "...code example..."},
    ]
    
    session = await episemic.analyze_session("user123", messages)
    
    # System learns: user prefers code examples
    # Future retrievals will prioritize memories with code
    results = await episemic.recall(
        "machine learning",
        user_id="user123",
        use_interaction_adaptation=True
    )
```

### 12.3 Research Evaluation

```python
from episemic.evaluation import EvaluationFramework

framework = EvaluationFramework()

# Run A/B test
test_queries = [
    {"query": "help me learn Rust", "user_id": "user123"},
    {"query": "what should I focus on today", "user_id": "user123"},
]

ground_truth = [
    {"expected_intentions": ["goal:learn Rust"], "relevant_memory_ids": [...]},
    {"expected_intentions": ["goal:complete project"], "relevant_memory_ids": [...]},
]

metrics = await framework.run_baseline_vs_enhanced(
    test_queries,
    ground_truth
)

print(f"Precision: {metrics['precision']}")
print(f"Goal Alignment: {metrics['goal_alignment_score']}")
```

---

## 13. Performance Considerations

**Expected Overhead**:
- Intention extraction: +0.5-1s per memory (LLM call)
- Interaction analysis: +0.1-0.2s per session
- Enhanced retrieval: +0.05-0.1s (graph queries)

**Mitigation Strategies**:
- Async extraction (don't block `remember()`)
- Cache LLM results
- Batch extraction when possible
- Make features optional

**Target Metrics** (from Mem0 paper):
- Keep token usage increase < 20%
- Keep latency increase < 25%
- Achieve goal alignment improvement > 15%

---

## 14. Documentation Requirements

**Files to Update**:
- [ ] `README.md` - Add section on research extensions
- [ ] `USAGE.md` - Add examples of intention/interaction APIs
- [ ] `CLAUDE.md` - Update with new components and patterns
- [ ] `BLUEPRINT.md` - Add intention/interaction architecture
- [ ] Create `RESEARCH.md` - Document evaluation methodology

**API Documentation**:
- [ ] Document all new models (Intention, InteractionPattern, etc.)
- [ ] Document all new API methods
- [ ] Document configuration options
- [ ] Add code examples

---

## 15. Questions for Implementation

1. **LLM Provider**: Default to OpenAI? Support others (Anthropic, local)?
2. **Extraction Timing**: Always async, or block if confidence needed immediately?
3. **User IDs**: Require explicit user_id or infer from context?
4. **Graph Query Depth**: What's reasonable max depth for intention graph traversal?
5. **Interaction Window**: How many sessions to keep for pattern computation?

---

## 16. Success Criteria

**Functional**:
- [ ] All existing tests pass
- [ ] New components have >80% test coverage
- [ ] Both DuckDB and PostgreSQL backends work
- [ ] CLI commands functional
- [ ] Documentation complete

**Research**:
- [ ] Intention extraction achieves >0.7 F1 vs human annotations
- [ ] Enhanced retrieval shows >10% improvement in goal alignment
- [ ] System maintains <25% latency increase
- [ ] A/B test results are statistically significant (p < 0.05)

---

## 17. File Structure Summary

```
episemic/
├── models.py (EXTEND)
│   └── Add: Intention, IntentionRelationship, InteractionPattern, InteractionSession
├── config.py (EXTEND)
│   └── Add: intention/interaction config fields
├── api.py (EXTEND)
│   └── Add: intention/interaction methods
├── intention/ (NEW)
│   ├── __init__.py
│   ├── intention_store.py
│   ├── extractor.py
│   └── prompts.py
├── interaction/ (NEW)
│   ├── __init__.py
│   ├── interaction_store.py
│   └── analyzer.py
├── retrieval/
│   └── retrieval.py (EXTEND)
│       └── Add: intention-aware ranking, interaction adaptation
├── evaluation/ (NEW)
│   ├── __init__.py
│   ├── framework.py
│   └── annotation.py
├── hippocampus/
│   └── duckdb_schema.py (EXTEND)
├── cortex/
│   └── postgres_schema.sql (EXTEND)
└── cli/
    └── main.py (EXTEND)

tests/
├── test_intention_store.py (NEW)
├── test_interaction_store.py (NEW)
├── test_intention_extraction.py (NEW)
├── test_interaction_analysis.py (NEW)
└── test_enhanced_retrieval.py (NEW)
```

---

**End of Specification**

This spec provides a complete blueprint for implementing EPISEMIC-II extensions. All components integrate with the existing architecture while maintaining backward compatibility and enabling rigorous research evaluation.