from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.agent.workflow import ReActAgent, ToolCallResult, AgentStream, AgentInput
from llama_index.core.workflow import Context
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore
import os
import re
from pathlib import Path
from sqlalchemy import make_url
from models import create_document_session, table_exists_in_db, delete_document_metadata_by_doc_id
from mcp_tools import mcp_registry, create_llamaindex_tool_from_mcp
from context_evaluator import ContextEvaluator
from typing import List, Dict, Any, Optional

DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/vectordoc")

# Connect to PGVector
url = make_url(DB_URL)

# Common prompt components shared across all task types
COMMON_LANGUAGE_REQUIREMENT = """CRITICAL LANGUAGE REQUIREMENT - THIS IS MANDATORY:
- You MUST detect the language of the user's question and respond EXACTLY in that same language
- NEVER mix languages or respond in a different language than the user's question
- This rule applies to ALL your responses: reasoning (Thought), actions, and final Answer"""

COMMON_CONTEXT_STRATEGY = """CONTEXT ENGINEERING STRATEGY:
1. ALWAYS use search_documents tool FIRST to check your vector database for relevant information
2. ONLY use web_search tool if search_documents does not return relevant or sufficient information
3. This ensures you prioritize local knowledge before searching the web"""

# Fine-tuned system prompts for different task types
TASK_PROMPTS = {
    "default": f"""You are a helpful assistant that can perform calculations and search through uploaded documents to answer questions.

{COMMON_LANGUAGE_REQUIREMENT}

{COMMON_CONTEXT_STRATEGY}

Use available MCP tools when appropriate to enhance your capabilities.""",
    
    "document_analysis": f"""You are a specialized document analysis assistant. Your primary role is to:
1. Thoroughly analyze uploaded documents
2. Extract key information, themes, and insights
3. Provide detailed summaries and comparisons
4. Answer questions with specific references to document content

{COMMON_LANGUAGE_REQUIREMENT}

{COMMON_CONTEXT_STRATEGY}
- Use MCP tools to access file contents when needed

Always prioritize accuracy and cite specific sections when possible.""",
    
    "research": f"""You are a research assistant with access to multiple information sources. Your capabilities include:
1. Searching through your document knowledge base
2. Using web search tools for current information
3. Synthesizing information from multiple sources
4. Providing well-structured, cited responses

{COMMON_LANGUAGE_REQUIREMENT}

{COMMON_CONTEXT_STRATEGY}

Always verify information and indicate your confidence level.""",
    
    "calculation": f"""You are a calculation assistant. Your role is to:
1. Perform accurate mathematical calculations
2. Use the calculator tool for complex expressions
3. Explain your calculation steps
4. Verify results when appropriate

{COMMON_LANGUAGE_REQUIREMENT}

Always show your work and double-check calculations.""",
    
    "general": f"""You are an intelligent assistant with access to:
- A document knowledge base (vector store) via search_documents tool
- File system operations (via MCP tools)
- Web search capabilities via web_search tool
- Mathematical calculation tools

{COMMON_LANGUAGE_REQUIREMENT}

{COMMON_CONTEXT_STRATEGY}

Use the most appropriate tools for each task. Always provide clear, accurate, and helpful responses."""
}


class ReasoningActionTracker:
    """Tracks reasoning and action steps for ReAct agent."""
    
    def __init__(self):
        self.reasoning_steps: List[str] = []
        self.action_steps: List[Dict[str, Any]] = []
        self.reset()
    
    def reset(self):
        """Reset tracking for new query."""
        self.reasoning_steps = []
        self.action_steps = []
    
    def add_reasoning(self, reasoning: str):
        """Add a reasoning step."""
        self.reasoning_steps.append(reasoning)
    
    def add_action(self, action_name: str, action_input: Dict[str, Any], action_output: str):
        """Add an action step."""
        self.action_steps.append({
            "tool_name": action_name,
            "input": action_input,
            "output": action_output
        })
    
    def get_steps(self) -> Dict[str, Any]:
        """Get all reasoning and action steps."""
        return {
            "reasoning_steps": self.reasoning_steps,
            "action_steps": self.action_steps
        }


class AgentDocument:
    # State constants
    STATE_REASONING = "reasoning"
    STATE_ANSWER = "answer"
    STATE_ACTION = "action"
    STATE_NONE = None
    
    # Marker patterns
    MARKER_THOUGHT = "Thought:"
    MARKER_ACTION = "Action:"
    MARKER_ANSWER = "Answer:"
    
    def __init__(self, task_type: str = "default"):
        """
        Initialize the agent with MCP tools and fine-tuned prompts.
        Uses ReActAgent to enable reasoning and action tracking.
        
        Args:
            task_type: Type of task to optimize for. Options:
                - "default": General purpose
                - "document_analysis": Focus on document analysis
                - "research": Focus on research tasks
                - "calculation": Focus on calculations
                - "general": General purpose with all tools
        """
        # Initialize Postgres Vector Store
        self.vector_store = PGVectorStore.from_params(
            database=url.database,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name="agent_vectors",
            embed_dim=1536,  # OpenAI embedding dimension
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )
        
        self.index = self.load_or_create_index()
        
        # Initialize context evaluator
        self.context_evaluator = ContextEvaluator()
        
        # Initialize reasoning/action tracker
        self.tracker = ReasoningActionTracker()
        
        # Store context for maintaining session state
        self.ctx = None
        
        # Get fine-tuned system prompt
        system_prompt = TASK_PROMPTS.get(task_type, TASK_PROMPTS["default"])
        
        # Build tools list: QueryEngineTool for document search + base tools + MCP tools
        tools = [self.multiply]
        
        # Create QueryEngineTool for the main document index
        # This provides flexibility to add more query engines with descriptions
        document_query_engine = self.index.as_query_engine(similarity_top_k=3)
        document_tool = QueryEngineTool.from_defaults(
            query_engine=document_query_engine,
            name="search_documents",
            description=(
                "PRIORITY 1: Search your vector database for information from uploaded documents. "
                "ALWAYS use this tool FIRST before using web_search. "
                "Provides information from files, essays, or any content that has been uploaded and indexed. "
                "Use a detailed plain text question as input. "
                "Only use web_search if this tool doesn't return relevant information."
            ),
        )
        tools.append(document_tool)
        
        # Add MCP tools
        for mcp_tool_name in mcp_registry.tools.keys():
            mcp_tool = mcp_registry.get_tool(mcp_tool_name)
            llamaindex_tool = create_llamaindex_tool_from_mcp(mcp_tool)
            tools.append(llamaindex_tool)
        
        # Initialize ReAct Agent
        # ReActAgent uses the ReAct pattern: it reasons about what to do, then takes actions
        self.agent = ReActAgent(
            tools=tools,
            llm=OpenAI(model="gpt-4o-mini"),
            system_prompt=system_prompt
        )
        
        # Initialize context for maintaining session state
        # Context is created with the agent as a positional argument
        self.ctx = Context(self.agent)
        
        self.task_type = task_type
        print(f"✅ ReAct Agent initialized with task type: {task_type}")
        print(f"✅ MCP tools registered: {list(mcp_registry.tools.keys())}")
        print(f"✅ QueryEngineTool created for document search")
    
    def get_agent(self):
        return self.agent
    
    def load_or_create_index(self) -> VectorStoreIndex:
        """
        Loads the index from Postgres if it exists, otherwise creates it 
        from the data directory.
        """
        try:
            # 0. check if table exists, if not create one
            # Note that the real table create will have a data_ prefix.
            if table_exists_in_db("data_agent_vectors"):
                # 1. Try to load from the existing Vector Store
                index = VectorStoreIndex.from_vector_store(self.vector_store)
                print("✅ Index loaded from Postgres/pgvector.")
                return index
            else:
                # 2. If valid index not found (or first run), load from files
                print(f"⚠️ Could not load from DB. Creating new index from 'data' folder...")
                if not os.path.exists("data"):
                    print("Data folder not found...")
                    raise FileNotFoundError("Data folder not found.")
                documents = SimpleDirectoryReader("data").load_data()
                # store document_id on a separate metadata table
                create_document_session(documents)
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
                print("✅ New index created and persisted to Postgres.")
                return index
            
        except Exception as e:
            print(f"Error saving metadata: {e}")

    async def get_response_with_evaluation(self, query: str) -> dict:
        """
        Get agent response and evaluate the context quality.
        Uses streaming to capture reasoning and actions in real-time.
        Returns both the response and evaluation metrics.
        """
        # Reset tracker for new query
        self.tracker.reset()
        
        # Run agent with context and stream events
        handler = self.agent.run(query, ctx=self.ctx)
        
        # Stream events to capture reasoning, actions, and response
        full_response = ""
        async for ev in handler.stream_events():
            if isinstance(ev, ToolCallResult):
                # Capture tool calls (actions)
                self.tracker.add_action(
                    action_name=ev.tool_name,
                    action_input=ev.tool_kwargs,
                    action_output=str(ev.tool_output)[:500]  # Limit output length
                )
            elif isinstance(ev, AgentStream):
                # Accumulate response
                full_response += ev.delta
        
        # Get final response
        response = await handler
        response_str = str(response) if full_response == "" else full_response
        
        # Extract any remaining reasoning from full response
        self._extract_reasoning_from_response(response_str)
        
        # Evaluate context quality
        evaluation = await self.context_evaluator.evaluate_quality(query, response_str)
        
        return {
            "response": response_str,
            "evaluation": evaluation,
            "reasoning_steps": self.tracker.reasoning_steps,
            "action_steps": self.tracker.action_steps
        }
    
    async def get_response_with_reasoning(self, query: str) -> dict:
        """
        Get agent response with reasoning and action steps using streaming.
        Returns response, reasoning steps, and action steps.
        """
        # Reset tracker for new query
        self.tracker.reset()
        
        # Run agent with context and stream events
        handler = self.agent.run(query, ctx=self.ctx)
        
        # Stream events to capture reasoning, actions, and response
        full_response = ""
        async for ev in handler.stream_events():
            if isinstance(ev, ToolCallResult):
                # Capture tool calls (actions)
                self.tracker.add_action(
                    action_name=ev.tool_name,
                    action_input=ev.tool_kwargs,
                    action_output=str(ev.tool_output)[:500]  # Limit output length
                )
            elif isinstance(ev, AgentStream):
                # Accumulate response
                full_response += ev.delta
        
        # Get final response
        response = await handler
        response_str = str(response) if full_response == "" else full_response
        
        # Extract any remaining reasoning from full response
        self._extract_reasoning_from_response(response_str)
        
        return {
            "response": response_str,
            "reasoning_steps": self.tracker.reasoning_steps,
            "action_steps": self.tracker.action_steps
        }
    
    async def stream_response(self, query: str, evaluate_context: bool = False):
        """
        Stream agent response with reasoning, action, and answer detection.
        Detects "Thought:", "Action:", and "Answer:" markers in the stream.
        Handles flexible state transitions (e.g., Thought -> Action -> Answer -> Thought -> Action -> Answer).
        Yields chunks immediately as they come in for real-time streaming.
        
        Args:
            query: The user's query
            evaluate_context: If True, evaluates context quality after streaming completes
        """
        # Reset tracker for new query
        self.tracker.reset()
        
        # State machine: can transition between reasoning, action, and answer states
        # Start with reasoning state (agent typically starts with Thought:)
        current_state = self.STATE_REASONING
        
        # Buffer to accumulate deltas for marker detection
        # Markers may be split across multiple deltas
        content_buffer = ""
        
        # Run agent with context and stream events
        handler = self.agent.run(query, ctx=self.ctx)
        
        print(f"\n🔵 Starting ReAct Agent stream for query: {query}")
        print(f"📍 Initial state: {current_state}")
        
        # Process streaming events
        async for ev in handler.stream_events():
            # Handle ToolCallResult events (actual tool executions)
            if isinstance(ev, ToolCallResult):
                # ToolCallResult provides structured Action data - this is just an event, don't change state
                # State changes should only come from parsing "Thought:", "Action:", "Answer:" markers in the stream
                action_data = {
                    "type": "action",
                    "tool_name": ev.tool_name,
                    "tool_input": ev.tool_kwargs,
                    "tool_output": str(ev.tool_output)[:500],
                    "state_changed": False  # ToolCallResult doesn't change state - state is determined by text markers
                }
                self.tracker.add_action(
                    action_name=ev.tool_name,
                    action_input=ev.tool_kwargs,
                    action_output=str(ev.tool_output)[:500]
                )
                print(f"🔧 Action executed: {ev.tool_name} (current state: {current_state})")
                yield action_data
                continue
            
            # Handle AgentInput events (skip these, they're just input events)
            elif isinstance(ev, AgentInput):
                continue
            
            # Handle AgentStream events (text deltas)
            elif isinstance(ev, AgentStream):
                delta = ev.delta
                
                # Accumulate delta to buffer for marker detection
                content_buffer += delta
                
                # Check buffer for "Thought:", "Action:", and "Answer:" markers
                # We need to detect which marker appears first (if any)
                marker_detected = False
                buffer_lower = content_buffer.lower()
                
                # Find all marker positions
                thought_pattern = r'(^|\s)(thought:)(\s|$)'
                action_pattern = r'(^|\s)(action:)(\s|$)'
                answer_pattern = r'(^|\s)(answer:)(\s|$)'
                
                thought_match = re.search(thought_pattern, buffer_lower)
                action_match = re.search(action_pattern, buffer_lower)
                answer_match = re.search(answer_pattern, buffer_lower)
                
                # Determine which marker appears first (if any)
                matches = []
                if thought_match:
                    matches.append((thought_match.start(), "thought", thought_match))
                if action_match:
                    matches.append((action_match.start(), "action", action_match))
                if answer_match:
                    matches.append((answer_match.start(), "answer", answer_match))
                
                # Sort by position to find the first marker
                if matches:
                    matches.sort(key=lambda x: x[0])
                    first_match_pos, marker_type, match = matches[0]
                    marker_detected = True
                    
                    # Determine new state based on marker type
                    new_state = {
                        "thought": self.STATE_REASONING,
                        "action": self.STATE_ACTION,
                        "answer": self.STATE_ANSWER
                    }[marker_type]
                    
                    # Log state transition when marker is detected (always log, even if state doesn't change)
                    old_state = current_state
                    state_changed = current_state != new_state
                    marker_label = {
                        "thought": "Thought",
                        "action": "Action",
                        "answer": "Answer"
                    }[marker_type]
                    if state_changed:
                        print(f"📍 Marker: '{marker_label}' -> State: {old_state} → {new_state}")
                    else:
                        # Still log marker detection even if state doesn't change (for debugging)
                        print(f"📍 Marker: '{marker_label}' detected (state unchanged: {current_state})")
                    
                    # Split buffer: content before marker and after marker
                    marker_start = match.start()
                    marker_end = match.end()
                    content_before_marker = content_buffer[:marker_start]
                    content_after_marker = content_buffer[marker_end:].lstrip()  # Remove marker and leading whitespace
                    
                    # Stream content before marker based on OLD state (before marker was detected)
                    if content_before_marker.strip():
                        if old_state == self.STATE_REASONING:
                            # Content before marker in reasoning state - append to last reasoning step
                            reasoning_data = {
                                "type": "reasoning",
                                "content": content_before_marker,
                                "state_changed": False,  # No state change, continuing same reasoning
                                "new_reasoning": False  # Not a new reasoning step
                            }
                            # Append to last reasoning step if exists, otherwise create new one
                            if self.tracker.reasoning_steps:
                                self.tracker.reasoning_steps[-1] += " " + content_before_marker
                            else:
                                self.tracker.add_reasoning(content_before_marker)
                            yield reasoning_data
                        elif old_state == self.STATE_ANSWER:
                            answer_data = {
                                "type": "answer",
                                "content": content_before_marker
                            }
                            yield answer_data
                        elif old_state == self.STATE_ACTION:
                            # Action text (before actual tool execution)
                            # We can skip this or handle it differently
                            pass
                    
                    # Now update state to new_state for content after marker
                    current_state = new_state
                    
                    # Handle content after marker (using new state)
                    if marker_type == "thought":
                        # If there's content after the marker, stream it as reasoning
                        if content_after_marker.strip():
                            # ALWAYS create a new reasoning step when we detect a "Thought:" marker
                            # Each "Thought:" marker represents a distinct reasoning step, even if we're already in reasoning state
                            reasoning_data = {
                                "type": "reasoning",
                                "content": content_after_marker,
                                "new_reasoning": True,  # Always create new step when Thought marker detected
                                "state_changed": state_changed  # Signal state change (if transitioning from another state)
                            }
                            # New reasoning step - add it
                            self.tracker.add_reasoning(content_after_marker)
                            print(f"   → Creating NEW reasoning step #{len(self.tracker.reasoning_steps)} (Thought marker detected)")
                            yield reasoning_data
                            content_buffer = ""  # Clear buffer since we streamed it
                        else:
                            content_buffer = content_after_marker
                    elif marker_type == "answer":
                        if content_after_marker.strip():
                            answer_data = {
                                "type": "answer",
                                "content": content_after_marker,
                                "state_changed": state_changed  # Signal state change
                            }
                            yield answer_data
                            # Clear buffer since we streamed the content after the marker
                            content_buffer = ""
                        else:
                            # No content after marker, keep buffer empty
                            content_buffer = ""
                    elif marker_type == "action":
                        # Action text will be followed by ToolCallResult event
                        # Just update buffer, don't stream action text yet
                        content_buffer = content_after_marker
                
                # Stream buffer content based on current state (only if marker wasn't just detected)
                # If marker was detected, we already streamed the before-marker content
                # IMPORTANT: Keep a rolling buffer (last 200 chars) to detect markers that span deltas
                if not marker_detected and content_buffer.strip():
                    # Stream if buffer is substantial, but keep last 200 chars for marker detection
                    MIN_BUFFER_FOR_MARKER_DETECTION = 200
                    should_stream = len(content_buffer) > MIN_BUFFER_FOR_MARKER_DETECTION
                    if should_stream:
                        # Keep last N chars in buffer, stream the rest
                        chars_to_keep = min(MIN_BUFFER_FOR_MARKER_DETECTION, len(content_buffer))
                        stream_content = content_buffer[:-chars_to_keep]
                        content_buffer = content_buffer[-chars_to_keep:]  # Keep last chars for marker detection
                        
                        if current_state == self.STATE_REASONING:
                            reasoning_data = {
                                "type": "reasoning",
                                "content": stream_content,
                                "state_changed": False  # No state change during streaming within same state
                            }
                            self.tracker.add_reasoning(stream_content)
                            yield reasoning_data
                        elif current_state == self.STATE_ANSWER:
                            answer_data = {
                                "type": "answer",
                                "content": stream_content,
                                "state_changed": False  # No state change during streaming within same state
                            }
                            yield answer_data
                        # Skip streaming for ACTION state (wait for ToolCallResult)
        
        # Stream any remaining buffer content before finalizing
        # IMPORTANT: Flush ALL remaining buffer content, even if it's the rolling buffer
        if content_buffer.strip():
            print(f"\n⚠️ Flushing remaining buffer content (length: {len(content_buffer)}, state: {current_state})")
            print(f"📝 Buffer content: {content_buffer[:200]}...")
            if current_state == self.STATE_REASONING:
                reasoning_data = {
                    "type": "reasoning",
                    "content": content_buffer,
                    "state_changed": False
                }
                self.tracker.add_reasoning(content_buffer)
                yield reasoning_data
            elif current_state == self.STATE_ANSWER:
                answer_data = {
                    "type": "answer",
                    "content": content_buffer,
                    "state_changed": False
                }
                yield answer_data
            elif current_state == self.STATE_ACTION:
                # Action state buffer content is typically action metadata, not user-facing content
                # Skip streaming it as it's not useful for the user
                print(f"⚠️ Skipping buffer flush - content is in ACTION state (likely action metadata)")
            # Clear buffer after flushing
            content_buffer = ""
            print(f"✅ Buffer flushed and cleared")
        
        # Get final response - await handler ensures all events have been processed
        # By this point, all streaming events should have been handled and streamed
        print(f"\n⏳ Waiting for final response from handler...")
        try:
            import asyncio
            # Add timeout to prevent hanging indefinitely (5 minutes max)
            response = await asyncio.wait_for(handler, timeout=300.0)
            print(f"✅ Handler completed successfully")
        except asyncio.TimeoutError:
            print(f"❌ ERROR: Handler timed out after 300 seconds")
            # Try to get partial response if available
            try:
                response = handler  # handler might have partial result
                print(f"⚠️ Attempting to use partial response")
            except:
                response = "Error: Request timed out after 300 seconds"
                print(f"❌ Could not retrieve partial response")
        except Exception as e:
            print(f"❌ ERROR getting final response: {e}")
            response = f"Error: {str(e)}"
        
        response_str = str(response)
        print(f"\n✅ Final response received (length: {len(response_str)})")
        print(f"📊 Strategy summary: {len(self.tracker.reasoning_steps)} reasoning steps, {len(self.tracker.action_steps)} actions")
        print(f"📍 Final state: {current_state}, Buffer length: {len(content_buffer)}")
        print(f"\n{'='*80}")
        print(f"📝 FINAL RESPONSE CONTENT (for debugging):")
        print(f"{'='*80}")
        print(response_str)
        print(f"{'='*80}\n")
        
        # If evaluation is requested, perform it after streaming completes
        # Use the final response string which contains the complete answer
        if evaluate_context:
            print(f"🔍 Evaluating context quality...")
            evaluation_result = await self.context_evaluator.evaluate_quality(query, response_str)
            evaluation_event = {
                "type": "evaluation",
                "evaluation": evaluation_result
            }
            yield evaluation_event
        
        # Final event: Completion signal
        final_response = {
            "type": "final"
        }
        yield final_response
        
    
    def _extract_reasoning_from_response(self, response_str: str):
        """
        Extract reasoning steps from response string.
        Looks for Thought: patterns in the response.
        """
        try:
            # Look for Thought: patterns
            if "Thought:" in response_str:
                parts = response_str.split("Thought:")
                for part in parts[1:]:  # Skip first part (before first Thought)
                    # Extract until next marker or end
                    reasoning = part.split("Action:")[0].split("Answer:")[0].strip()
                    if reasoning:
                        self.tracker.add_reasoning(reasoning)
        except Exception as e:
            print(f"Error extracting reasoning from response: {e}")
    
    def _extract_reasoning_and_actions(self, response):
        """
        Extract reasoning and action steps from agent response.
        ReActAgent uses ReAct pattern which interleaves reasoning and actions.
        We extract this from the agent's source_nodes and message content.
        """
        try:
            # Get the chat history from the agent
            chat_history = self.agent.chat_history if hasattr(self.agent, 'chat_history') else []
            
            # Also check if response has source_nodes (tool calls)
            source_nodes = []
            if hasattr(response, 'source_nodes'):
                source_nodes = response.source_nodes
            
            # Process the full response text to extract ReAct pattern
            response_text = str(response)
            
            # Look for reasoning patterns in the response
            # ReAct agents typically show: Thought -> Action -> Observation -> Thought -> ...
            lines = response_text.split('\n')
            current_reasoning = []
            current_action = None
            
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                
                # Detect reasoning indicators
                if any(indicator in line_lower for indicator in [
                    'thought:', 'think:', 'reasoning:', 'i need', 'i should', 
                    'i will', 'let me', 'first', 'then', 'next', 'so'
                ]):
                    if line.strip():
                        current_reasoning.append(line.strip())
                
                # Detect action indicators (tool calls)
                elif any(indicator in line_lower for indicator in [
                    'action:', 'tool:', 'function:', 'calling', 'using tool',
                    'executing', 'running'
                ]):
                    if current_reasoning:
                        self.tracker.add_reasoning('\n'.join(current_reasoning))
                        current_reasoning = []
                    
                    # Extract tool name and input
                    action_name = "unknown"
                    action_input = {}
                    
                    # Try to extract tool name
                    for tool_word in ['multiply', 'search_content', 'read_file', 'web_search', 'calculate']:
                        if tool_word in line_lower:
                            action_name = tool_word
                            break
                    
                    current_action = {
                        "tool_name": action_name,
                        "input": action_input,
                        "output": ""
                    }
                    self.tracker.add_action(action_name, action_input, "")
                
                # Detect observation/result
                elif current_action and any(indicator in line_lower for indicator in [
                    'observation:', 'result:', 'output:', 'answer:'
                ]):
                    if line.strip():
                        # Update last action's output
                        if self.tracker.action_steps:
                            self.tracker.action_steps[-1]["output"] += line.strip() + "\n"
                    current_action = None
                elif current_action and line.strip():
                    # Continue collecting output
                    if self.tracker.action_steps:
                        self.tracker.action_steps[-1]["output"] += line.strip() + "\n"
            
            # Add any remaining reasoning
            if current_reasoning:
                self.tracker.add_reasoning('\n'.join(current_reasoning))
            
            # Process source nodes (actual tool calls from llama_index)
            for node in source_nodes:
                node_text = str(node.node.text) if hasattr(node, 'node') else str(node)
                # Try to identify which tool was used
                for tool_word in ['multiply', 'search_content', 'read_file', 'web_search', 'calculate']:
                    if tool_word in node_text.lower():
                        # Check if we already added this action
                        if not any(step["tool_name"] == tool_word for step in self.tracker.action_steps):
                            self.tracker.add_action(tool_word, {}, node_text[:500])  # Limit output length
                            break
            
            # If no reasoning/actions found, add a default one
            if not self.tracker.reasoning_steps and not self.tracker.action_steps:
                # Try to infer from response
                if len(response_text) > 100:
                    # Extract first few sentences as reasoning
                    sentences = response_text.split('.')[:3]
                    self.tracker.add_reasoning('. '.join(sentences) + '.')
                    
        except Exception as e:
            print(f"Error extracting reasoning and actions: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: add basic reasoning
            self.tracker.add_reasoning("Processing request...")

    def add_file_context_to_agent(self, filepath: Path) -> bool:
        """Process a new file and insert it into the Postgres vector store."""
        if filepath.exists():
            print(f"Processing file: {str(filepath)}")
            
            documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
            # store document_id on a separate metadata table
            create_document_session(documents)
            # Insert into existing index (automatically updates PGVector)
            for doc in documents:
                self.index.insert(doc)
            print(f"✅ {str(filepath)} added to vector store.")
            return True
        else:
            print(f"{str(filepath)} does not exist. Adding new file failed.")
            return False

    def update_file_context_in_agent(self, filepath: Path, doc_id: str) -> bool:
        """Update an existing file in the Postgres vector store."""
        # check if file exists
        if filepath.exists():
            print(f"Updating file: {filepath}")
            updated_documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
            # Re-insert into existing doc_id (LlamaIndex handles some deduplication)
            for i in range(len(updated_documents)):
                updated_documents[i].doc_id = doc_id  # ensure we use the same doc_id
                print(f"setting document id to : {doc_id} for updating same data.")
                # refresh the index, since we are not directly inserting text
                self.index.update_ref_doc(updated_documents[i])
            print(f"✅ {str(filepath)} updated in vector store.")
            # update DocumentMetadata table
            create_document_session(updated_documents)
            print(f"✅ {str(filepath)} updated in document metadata.")
            return True
        else:
            print(f"{str(filepath)} does not exist. Updating file failed.")
            return False
    
    def delete_file_context_in_agent(self, doc_id: str) -> bool:
        """Delete a file from the Postgres vector store using its doc_id."""
        try:
            self.index.delete_ref_doc(doc_id)
            print(f"✅ Document with doc_id: {doc_id} deleted from vector store.")
            delete_document_metadata_by_doc_id(doc_id)
            print(f"✅ Document metadata with doc_id: {doc_id} deleted from database.")
            return True
        except Exception as e:
            print(f"Error deleting document with doc_id: {doc_id}; {e}")
            return False

    def multiply(self, a: float, b: float) -> float:
        """Useful for multiplying two numbers."""
        return a * b

    async def search_content(self, query: str) -> str:
        """Useful for answering natural language questions about essays/files."""
        query_engine = self.index.as_query_engine()
        response = await query_engine.aquery(query)
        return str(response)