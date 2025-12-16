# Prompt & Context Engineering Improvement Plan

## Executive Summary
This plan outlines strategies to improve prompt engineering and context engineering to address response duplication, enhance response quality, and optimize information retrieval patterns.

---

## Current State Analysis

### Current Strengths
1. ✅ Basic context engineering strategy (vector DB first, then web search)
2. ✅ Task-specific prompts for different use cases
3. ✅ Tool descriptions with priority indicators
4. ✅ ReAct pattern for reasoning and actions

### Current Issues
1. ❌ Response duplication when agent provides multiple answers
2. ❌ No explicit instruction to consolidate/synthesize information
3. ❌ Limited guidance on handling multi-step queries
4. ❌ No deduplication logic in prompts
5. ❌ Context retrieval strategy is binary (all-or-nothing)

---

## Phase 1: Enhanced Prompt Engineering

### 1.1 Add Response Formatting Instructions

**Objective**: Prevent duplication by explicitly instructing the agent to consolidate information.

**Implementation**:
```python
RESPONSE_FORMATTING_GUIDELINES = """
RESPONSE FORMATTING RULES:
1. When providing multiple answers or updates, CONSOLIDATE them into a single, comprehensive response
2. DO NOT repeat information that was already provided in a previous answer section
3. When adding new information, indicate it's an ADDITION or UPDATE to previous content
4. Use clear section headers if providing structured information (e.g., "Part 1: From Documents", "Part 2: From Web Search")
5. If asked for multiple sets of facts (e.g., "5 facts from documents, 5 from web"), clearly separate them but avoid repeating overlap
"""
```

**Location**: Add to all TASK_PROMPTS

---

### 1.2 Implement Chain-of-Thought (CoT) with Consolidation

**Objective**: Guide the agent to explicitly think about information synthesis before answering.

**Implementation**:
```python
CONSOLIDATION_INSTRUCTION = """
BEFORE providing your final Answer:
1. Review all information you've gathered from your tools
2. Identify any overlapping or duplicate information
3. Synthesize the information into a coherent, non-redundant response
4. If you need to reference previous information, use phrases like "As mentioned earlier" or "Building on the previous points"
5. Only include NEW information in subsequent answer sections
"""
```

**Location**: Add to system prompts, especially for "research" and "general" task types

---

### 1.3 Add Few-Shot Examples

**Objective**: Show the agent desired response patterns with examples.

**Implementation**:
```python
FEW_SHOT_EXAMPLES = {
    "research": """
Example of GOOD response pattern:
User: "Give me 3 facts from documents and 3 from web"
Agent: 
Thought: I should first search documents, then web search for additional facts.
Action: search_documents("solar system facts")
Observation: [3 facts returned]
Thought: Now I'll search the web for 3 additional, different facts.
Action: web_search("solar system interesting facts")
Observation: [different facts returned]
Answer: Here are 6 unique facts about the solar system:

From Documents:
1. [fact 1]
2. [fact 2]
3. [fact 3]

From Web Search (Additional):
4. [new fact 4]
5. [new fact 5]
6. [new fact 6]

Example of BAD response (duplication):
[Shows example of agent repeating same facts in two sections]
""",
    # Add more examples for other task types
}
```

**Location**: Create new prompt component, inject into prompts based on task type

---

### 1.4 Negative Prompting for Duplication

**Objective**: Explicitly tell the agent what NOT to do.

**Implementation**:
```python
NEGATIVE_PROMPTING = """
WHAT NOT TO DO:
- DO NOT repeat the same information in multiple answer sections
- DO NOT include duplicate facts when asked for separate sets (e.g., "5 from docs, 5 from web")
- DO NOT restate your entire previous answer when providing an update
- DO NOT include redundant explanations that were already provided
- DO NOT list the same information twice, even if it appears in multiple sources
"""
```

**Location**: Append to all system prompts

---

### 1.5 State-Aware Prompting

**Objective**: Provide different guidance based on conversation state.

**Implementation**:
```python
def get_state_aware_prompt(base_prompt: str, has_previous_answer: bool, answer_count: int) -> str:
    """
    Enhance base prompt with state-aware instructions.
    
    Args:
        base_prompt: Base system prompt
        has_previous_answer: Whether agent already provided an answer
        answer_count: Number of answer sections provided so far
    """
    state_instructions = ""
    
    if has_previous_answer and answer_count > 0:
        state_instructions = f"""
IMPORTANT: You have already provided {answer_count} answer section(s). 
- If you need to provide additional information, clearly indicate it's an ADDITION or UPDATE
- Do NOT repeat information from your previous answer
- Synthesize and consolidate with previous information
- Use phrases like "Additionally," "To add to the previous points," or "Building on this,"
"""
    
    return f"{base_prompt}\n\n{state_instructions}"
```

**Location**: New method in AgentDocument class, called during agent initialization or context updates

**Note**: This requires tracking conversation state, which may need context/chat history integration

---

## Phase 2: Advanced Context Engineering

### 2.1 Multi-Stage Context Retrieval Strategy

**Objective**: Improve context retrieval to be more nuanced than binary (docs vs web).

**Implementation**:
```python
class ContextRetrievalStrategy:
    """
    Sophisticated context retrieval strategy with multiple stages.
    """
    
    def __init__(self, agent_instance):
        self.agent = agent_instance
    
    async def retrieve_context(self, query: str, stage: str = "initial") -> dict:
        """
        Multi-stage context retrieval:
        - Stage 1: High-confidence vector search (similarity_top_k=3, high threshold)
        - Stage 2: Broader vector search (similarity_top_k=5, lower threshold)
        - Stage 3: Web search (if vector search insufficient)
        """
        context_results = {
            "vector_db_results": [],
            "web_results": [],
            "confidence_score": 0.0,
            "recommendation": ""
        }
        
        # Stage 1: Tight vector search
        vector_results = await self._search_vector_db(query, top_k=3, threshold=0.7)
        
        if len(vector_results) >= 3:
            context_results["vector_db_results"] = vector_results
            context_results["confidence_score"] = 0.9
            context_results["recommendation"] = "sufficient_from_docs"
        else:
            # Stage 2: Broader search
            vector_results = await self._search_vector_db(query, top_k=5, threshold=0.5)
            context_results["vector_db_results"] = vector_results
            
            if len(vector_results) >= 3:
                context_results["confidence_score"] = 0.7
                context_results["recommendation"] = "partially_sufficient_from_docs"
            else:
                # Stage 3: Web search recommended
                context_results["recommendation"] = "needs_web_search"
        
        return context_results
```

**Location**: New module `context_retrieval_strategy.py`

---

### 2.2 Context Deduplication Pre-Processing

**Objective**: Remove duplicate information before passing context to agent.

**Implementation**:
```python
class ContextDeduplicator:
    """
    Deduplicate context from multiple sources before passing to agent.
    """
    
    def __init__(self):
        self.llm = OpenAI(model="gpt-4o-mini")
    
    async def deduplicate_context(self, contexts: List[str]) -> str:
        """
        Use LLM to identify and remove duplicate information from multiple contexts.
        """
        if len(contexts) <= 1:
            return "\n".join(contexts)
        
        prompt = f"""You are a context consolidation assistant. 
        Review the following information chunks and create a single, non-redundant summary.
        Remove duplicate facts but preserve unique information from each source.
        
        Contexts:
        {chr(10).join(f"--- Source {i+1} ---{chr(10)}{ctx}" for i, ctx in enumerate(contexts))}
        
        Provide a consolidated, non-redundant summary:"""
        
        response = await self.llm.acomplete(prompt)
        return str(response).strip()
```

**Location**: New module `context_deduplicator.py`, integrate into QueryEngineTool wrapper

**Integration Point**: Modify `document_query_engine` to deduplicate results before returning

---

### 2.3 Query Decomposition and Planning

**Objective**: Break complex queries into sub-queries and plan retrieval strategy.

**Implementation**:
```python
class QueryPlanner:
    """
    Analyze query to determine optimal context retrieval strategy.
    """
    
    async def plan_retrieval(self, query: str) -> dict:
        """
        Analyze query to determine:
        - Number of information sets needed (e.g., "5 from docs, 5 from web")
        - Optimal retrieval order
        - Whether to deduplicate or separate sources
        """
        prompt = f"""Analyze this query and determine the retrieval strategy:
        
        Query: {query}
        
        Identify:
        1. How many distinct information sets are requested?
        2. What sources should be used (documents, web, both)?
        3. Should information be consolidated or kept separate?
        4. Are there any constraints (e.g., "5 from X, 5 from Y")?
        
        Respond in JSON:
        {{
            "information_sets": <number>,
            "sources": ["documents", "web"],
            "consolidation_strategy": "consolidate" | "separate",
            "constraints": {{"documents": 5, "web": 5}},
            "retrieval_order": ["documents", "web"]
        }}"""
        
        # Use LLM to analyze query
        # Return structured plan
```

**Location**: New module `query_planner.py`

**Integration Point**: Use plan in `stream_response` to guide agent behavior

---

### 2.4 Context Relevance Filtering

**Objective**: Filter retrieved context based on relevance scores before passing to agent.

**Implementation**:
```python
class RelevanceFilter:
    """
    Filter context chunks based on relevance scores.
    """
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
    
    def filter_by_relevance(self, context_chunks: List[dict]) -> List[dict]:
        """
        Filter context chunks where relevance score < threshold.
        context_chunks format: [{"content": "...", "score": 0.8, "source": "vector_db"}, ...]
        """
        return [
            chunk for chunk in context_chunks 
            if chunk.get("score", 0.0) >= self.threshold
        ]
```

**Location**: Add to context retrieval strategy

**Integration Point**: Filter results from `search_documents` before returning to agent

---

## Phase 3: Response Post-Processing

### 3.1 Response Consolidation Check

**Objective**: Add a final check to consolidate duplicate content in responses.

**Implementation**:
```python
class ResponseConsolidator:
    """
    Post-process agent response to remove duplicates.
    """
    
    async def consolidate_response(self, response: str, query: str) -> str:
        """
        Use LLM to identify and remove duplicate content in response.
        """
        prompt = f"""Review this response and remove any duplicate or redundant information.
        Consolidate repeated facts or explanations into a single, clear presentation.
        
        Original Query: {query}
        
        Response to consolidate:
        {response}
        
        Provide the consolidated, non-redundant version:"""
        
        consolidated = await self.llm.acomplete(prompt)
        return str(consolidated).strip()
```

**Location**: New module `response_consolidator.py`

**Integration Point**: 
- Option A: Apply in `stream_response` after final answer is collected
- Option B: Apply in frontend before displaying (less ideal, as it requires LLM call in frontend)

---

### 3.2 Semantic Similarity Deduplication

**Objective**: Use embeddings to detect and remove semantically similar content.

**Implementation**:
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticDeduplicator:
    """
    Detect and remove semantically similar sentences/phrases.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = similarity_threshold
    
    def deduplicate_text(self, text: str) -> str:
        """
        Remove sentences that are semantically too similar.
        """
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return text
        
        embeddings = self.model.encode(sentences)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Keep track of sentences to remove
        to_remove = set()
        for i in range(len(sentences)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(sentences)):
                if similarity_matrix[i][j] > self.threshold:
                    # Keep the longer/more detailed sentence
                    if len(sentences[i]) >= len(sentences[j]):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        
        # Reconstruct text without duplicates
        unique_sentences = [sentences[i] for i in range(len(sentences)) if i not in to_remove]
        return " ".join(unique_sentences)
```

**Location**: New module `semantic_deduplicator.py`

**Note**: Requires additional dependency `sentence-transformers`

---

## Phase 4: Tool Description Enhancement

### 4.1 Enhanced Tool Descriptions with Consolidation Hints

**Objective**: Update tool descriptions to guide agent on information synthesis.

**Current**:
```python
description="PRIORITY 1: Search your vector database..."
```

**Enhanced**:
```python
description=(
    "PRIORITY 1: Search your vector database for information from uploaded documents. "
    "ALWAYS use this tool FIRST before using web_search. "
    "When you later use web_search, DO NOT repeat information already found here. "
    "Instead, use web_search only for ADDITIONAL, complementary information. "
    "If you've already provided an answer using search_documents, any subsequent answer "
    "should BUILD UPON it, not repeat it."
)
```

**Location**: Update `QueryEngineTool` descriptions in `agent.py`

---

### 4.2 Dynamic Tool Descriptions Based on Query

**Objective**: Adapt tool descriptions based on query analysis.

**Implementation**:
```python
def generate_dynamic_tool_description(base_description: str, query_analysis: dict) -> str:
    """
    Enhance tool description based on query characteristics.
    """
    hints = []
    
    if query_analysis.get("multiple_sources_requested"):
        hints.append("When using multiple sources, ensure each provides UNIQUE information.")
    
    if query_analysis.get("has_constraints"):
        hints.append("Respect the specified constraints (e.g., '5 from X, 5 from Y') and avoid overlap.")
    
    if hints:
        return f"{base_description}\n\nIMPORTANT: {', '.join(hints)}"
    
    return base_description
```

**Location**: New method, integrate into tool initialization

---

## Phase 5: Integration Architecture

### 5.1 Proposed Component Structure

```
example_seven/backend/
├── agent.py (modified)
├── prompt_engineer.py (NEW)
│   ├── ResponseFormattingGuidelines
│   ├── FewShotExamples
│   ├── NegativePrompting
│   └── StateAwarePrompting
├── context_engineering.py (NEW)
│   ├── ContextRetrievalStrategy
│   ├── QueryPlanner
│   ├── RelevanceFilter
│   └── ContextDeduplicator
├── response_processing.py (NEW)
│   ├── ResponseConsolidator
│   └── SemanticDeduplicator
└── mcp_tools.py (modified tool descriptions)
```

---

### 5.2 Integration Flow

```
User Query
    ↓
QueryPlanner.plan_retrieval() → Determine strategy
    ↓
StateAwarePrompting.get_prompt() → Enhance prompt based on state
    ↓
Agent runs with enhanced prompt
    ↓
ContextRetrievalStrategy.retrieve_context() → Multi-stage retrieval
    ↓
ContextDeduplicator.deduplicate_context() → Remove duplicates from context
    ↓
Agent generates response (with consolidation instructions)
    ↓
ResponseConsolidator.consolidate_response() → Post-process if needed
    ↓
Return to frontend
```

---

## Phase 6: Testing & Evaluation Strategy

### 6.1 Test Cases

1. **Simple Duplication Test**
   - Query: "Tell me 5 fun facts about the solar system, then 5 more from web search"
   - Expected: 10 unique facts, no duplicates

2. **Overlapping Sources Test**
   - Query: "What do documents say about X? What does the web say about X?"
   - Expected: Distinct sections, overlap clearly marked

3. **Sequential Answer Test**
   - Query: "Give me an initial answer, then search web and provide an update"
   - Expected: Update builds on initial, doesn't repeat it

4. **Complex Multi-Step Test**
   - Query with multiple sub-questions requiring different sources
   - Expected: Clear organization, no redundancy

### 6.2 Metrics

- **Duplication Rate**: Percentage of repeated content
- **Information Coverage**: Percentage of unique information retained
- **Response Coherence**: LLM-based evaluation
- **User Satisfaction**: Manual review scores

---

## Implementation Priority

### Phase 1 (Quick Wins - 1-2 days)
1. ✅ Add Response Formatting Guidelines
2. ✅ Add Negative Prompting
3. ✅ Enhanced Tool Descriptions

### Phase 2 (Medium Complexity - 3-5 days)
1. ✅ Query Planner
2. ✅ Context Deduplication
3. ✅ State-Aware Prompting (basic)

### Phase 3 (Advanced - 5-7 days)
1. ✅ Multi-Stage Context Retrieval
2. ✅ Response Consolidation
3. ✅ Semantic Similarity Deduplication

### Phase 4 (Optimization - Ongoing)
1. ✅ Few-Shot Examples refinement
2. ✅ A/B testing different prompt variations
3. ✅ Performance optimization

---

## Considerations & Trade-offs

### Pros
- ✅ Addresses root cause (prompt engineering) rather than symptoms
- ✅ Improves overall response quality
- ✅ Reduces token usage (no duplicate content)
- ✅ Better user experience

### Cons
- ⚠️ Additional LLM calls for consolidation (latency, cost)
- ⚠️ More complex codebase
- ⚠️ Need to fine-tune thresholds and parameters
- ⚠️ May over-consolidate and lose nuance

### Mitigation Strategies
- Cache consolidation results for similar queries
- Make post-processing optional (flag-based)
- Use faster models (gpt-4o-mini) for consolidation
- Monitor and log consolidation actions for analysis

---

## Conclusion

This plan provides a comprehensive approach to addressing duplication through:
1. **Prompt Engineering**: Explicit instructions to prevent duplication
2. **Context Engineering**: Better retrieval and deduplication
3. **Post-Processing**: Safety net for edge cases

The phased approach allows for incremental implementation and testing, ensuring each component is validated before moving to the next.

