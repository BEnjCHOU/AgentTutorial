# Example Seven

In this example we implement a ReAct (Reasoning and Acting) agent that shows its reasoning process and actions in the frontend. This provides transparency into how the agent thinks and what tools it uses to answer questions.

## Features

This example extends [example six](../example_six/README.md) with the following enhancements:

### 1. ReAct Agent Pattern
- **ReAct Framework**: Uses `ReActAgent` from `llama_index.core.agent.workflow` which implements the ReAct (Reasoning and Acting) pattern
- **Iterative Reasoning**: The agent alternates between reasoning about the task and taking actions
- **Dynamic Decision Making**: The agent can adapt its approach based on intermediate results

### 2. Reasoning and Action Visibility
- **Reasoning Steps**: Captures and displays the agent's internal thought process
- **Action Steps**: Shows which tools were used, their inputs, and outputs
- **Frontend Display**: Visual representation of reasoning and actions in the chat interface
- **Real-time Transparency**: Users can see exactly how the agent arrives at its answers
- **Streaming Support**: Real-time streaming of reasoning, actions, and answers as they occur

### 3. QueryEngineTool Integration
- **Flexible Query Engines**: Uses `QueryEngineTool` to wrap query engines with descriptions
- **Multiple Query Engines**: Easy to add multiple query engines for different purposes
- **Descriptive Tools**: Each query engine has a clear description of what it searches
- **Better Tool Selection**: Agent can choose the right query engine based on descriptions

### 4. All Features from Example Six
- MCP (Model Context Protocol) server integration
- Fine-tuned prompts for dedicated tasks
- Context evaluation capabilities
- Document management (upload, update, delete)

## ReAct Pattern Explained

The ReAct (Reasoning and Acting) pattern is a framework where agents:
1. **Reason**: Analyze the current situation and plan next steps
2. **Act**: Execute actions using available tools
3. **Observe**: Process the results of actions
4. **Iterate**: Refine approach based on observations

This iterative process continues until the task is complete, making the agent's decision-making process transparent and interpretable.

## Notice
This example extends [example six](../example_six/README.md) with:
- ReAct agent implementation using `ReActAgent` from `llama_index.core.agent.workflow` instead of `FunctionAgent`
- `QueryEngineTool` integration for flexible query engine management
- Reasoning and action step tracking using streaming events
- Frontend UI components to display reasoning and actions in real-time
- Enhanced API responses that include reasoning/action metadata
- Streaming endpoint (`/ask-stream/`) for real-time updates

### Working Environment
- MacOS : Important : using a chip with Apple Silicon we need to explicitly turn the
environment variable ON.
```yml 
PYTHONUNBUFFERED: "1"
```

### Prerequisites Backend
1. Install python3 for the backend
2. Create an [OpenAI API KEY](https://platform.openai.com/api-keys)
3. Export the openai api key
4. Install postgresql and define the DATABASE_URL environment variable.
```
export OPENAI_API_KEY=XXXXX
```
5. Create a python environment
```
python3 -m venv env
```
6. source the environment
```
source env/bin/activate
```
7. install libraries
```bash
python3 -m pip install -r requirements.txt
```

**Note:** The web search feature uses DuckDuckGo via the `ddgs` package (included in requirements.txt). No API keys are required - web search works out of the box!

### Prerequisites Frontend
1. Install dependencies
```bash
cd frontend/my-app/
npm install
```

### Running the backend
```bash
cd backend/
uvicorn main:app --reload
```

### Running the frontend
```
cd frontend/my-app/
pnpm run dev
```

### Running with Docker Compose
```bash
docker-compose up --build
```
Open your browser to `http://localhost:3000`.

## API Endpoints

### Standard Endpoints (from example_six)
- `POST /ask/` - Ask a question to the agent (now includes reasoning and action steps)
- `POST /upload/` - Upload a file
- `PUT /update/` - Update an existing file
- `DELETE /delete/{filename}` - Delete a file
- `GET /files/` - List all uploaded files

### Enhanced Endpoints

#### `POST /ask/`
Ask a question to the ReAct agent. Returns response with reasoning and action steps.

**Request:**
```json
{
  "message": "What is the solar system?",
  "task_type": "document_analysis",
  "evaluate_context": false
}
```

**Response:**
```json
{
  "response": "The solar system is...",
  "reasoning_steps": [
    "I need to search through the document knowledge base to find information about the solar system.",
    "Let me check if there are any uploaded documents that might contain this information."
  ],
  "action_steps": [
    {
      "tool_name": "search_documents",
      "input": {"input": "solar system"},
      "output": "The solar system consists of the Sun and all objects..."
    }
  ],
  "status": "success"
}
```

#### `POST /ask-stream/`
Stream agent response with reasoning and action steps in real-time using Server-Sent Events (SSE).

**Request:**
```json
{
  "message": "What is the solar system?",
  "task_type": "document_analysis",
  "evaluate_context": false
}
```

**Response (Server-Sent Events stream):**
```
data: {"type": "reasoning", "content": "Thought: I need to search..."}
data: {"type": "action", "tool_name": "search_documents", "tool_input": {...}, "tool_output": "..."}
data: {"type": "answer", "content": "The solar system is..."}
data: {"type": "final", "response": "...", "reasoning_steps": [...], "action_steps": [...]}
```

#### `POST /ask-with-evaluation/`
Ask a question and get context evaluation metrics along with reasoning and action steps.

**Request:**
```json
{
  "message": "What are the main characteristics of dwarf planets?",
  "task_type": "document_analysis"
}
```

**Response:**
```json
{
  "response": "...",
  "evaluation": {
    "overall_quality_score": 0.85,
    "relevance": {...},
    "completeness": {...},
    "recommendation": "..."
  },
  "reasoning_steps": [...],
  "action_steps": [...],
  "status": "success"
}
```

#### `POST /set-task-type/`
Change the agent's task type for fine-tuned prompts.

**Request:**
```json
{
  "task_type": "document_analysis"
}
```

**Valid task types:**
- `default`
- `document_analysis`
- `research`
- `calculation`
- `general`

#### `GET /mcp-tools/`
List all available MCP tools.

**Response:**
```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "Read the contents of a file from the data directory",
      "inputSchema": {...}
    },
    ...
  ],
  "status": "success"
}
```

## Usage Examples

### Using ReAct Agent with Reasoning Display

When you ask a question, the frontend will display:
1. **Reasoning Steps**: Purple boxes showing the agent's thought process
2. **Action Steps**: Orange boxes showing which tools were used, their inputs, and outputs
3. **Final Response**: The agent's answer

Example interaction:
```
User: "What is 25 * 37?"

Agent Reasoning:
- Step 1: I need to calculate 25 * 37. I can use the multiply tool for this.

Agent Actions:
- Action 1:
  Tool: multiply
  Input: {"a": 25, "b": 37}
  Output: 925

Response: The result of 25 * 37 is 925.
```

### Using Task-Specific Prompts

```python
# Set task type for document analysis
POST /set-task-type/
{
  "task_type": "document_analysis"
}

# Ask a question - agent will use document_analysis prompt
POST /ask/
{
  "message": "Summarize the key points from the uploaded documents"
}
```

### Evaluating Context Quality

```python
# Ask with evaluation
POST /ask-with-evaluation/
{
  "message": "What are the main characteristics of dwarf planets?",
  "task_type": "document_analysis"
}
```

## Frontend Features

### Reasoning Display
- Purple-highlighted boxes showing each reasoning step
- Numbered steps for easy tracking
- Clear formatting for readability

### Action Display
- Orange-highlighted boxes for each action
- Tool name prominently displayed
- Input parameters shown in JSON format
- Output results displayed with scrollable overflow for long outputs

### Combined Display
The frontend shows reasoning and actions in chronological order, making it easy to understand the agent's decision-making process.

## Architecture

### ReAct Agent (`agent.py`)
- `AgentDocument`: Main agent class using `ReActAgent` from `llama_index.core.agent.workflow` (ReAct pattern)
- `ReasoningActionTracker`: Tracks reasoning and action steps
- `QueryEngineTool`: Wraps document query engine with description for better tool selection
- `get_response_with_reasoning()`: Returns response with reasoning/action steps (uses streaming)
- `get_response_with_evaluation()`: Returns response with evaluation and reasoning/action steps (uses streaming)
- `stream_response()`: Streams reasoning, actions, and answers in real-time
- `add_query_engine_tool()`: Method to add additional query engines with descriptions

### Enhanced API (`main.py`)
- All endpoints return reasoning and action steps when available
- Backward compatible with example_six (optional fields)

### Frontend (`page.tsx`)
- Updated `Message` interface to include `reasoning_steps` and `action_steps`
- New UI components for displaying reasoning and actions
- Color-coded sections (purple for reasoning, orange for actions)
- Streaming support: Real-time display of reasoning, actions, and answers as they occur
- Toggle option to enable/disable streaming mode

## Differences from Example Six

| Feature | Example Six | Example Seven |
|---------|-------------|---------------|
| Agent Type | `FunctionAgent` | `ReActAgent` |
| Reasoning Display | ❌ | ✅ |
| Action Display | ❌ | ✅ |
| ReAct Pattern | ❌ | ✅ |
| Step Tracking | ❌ | ✅ |

## Questions to ask the agent
1. How many groups can we separate the solar system's planets into?
2. What is 42 * 58? (Watch the reasoning and action steps!)
3. What is the earliest widely recognized writing system?
4. Search the web for recent AI developments. (See the web_search action)
5. What is the key distinction from a full planet compared to a dwarf planet?

## Extending the Agent

### Adding Multiple Query Engines

You can add multiple query engines with specific descriptions:

```python
# Create a specialized query engine
financial_query_engine = financial_index.as_query_engine(similarity_top_k=3)

# Add it to the agent
agent_instance.add_query_engine_tool(
    query_engine=financial_query_engine,
    name="search_financial_docs",
    description="Provides information from financial documents and reports. Use this for questions about revenue, expenses, or financial metrics."
)
```

### Adding Other Tools

To add new MCP tools or function tools, follow the same pattern as example_six. The ReAct agent will automatically:
- Include the tool in its reasoning process
- Show tool usage in action steps
- Display tool inputs and outputs in the frontend

## Benefits of ReAct Pattern

1. **Transparency**: Users can see how the agent thinks
2. **Debugging**: Easy to identify where the agent might have made mistakes
3. **Trust**: Users understand the reasoning behind answers
4. **Learning**: Users can learn from the agent's problem-solving approach
5. **Verification**: Users can verify that the correct tools were used

