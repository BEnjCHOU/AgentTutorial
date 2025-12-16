from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import AgentDocument
from models import init_db, SessionLocal, DocumentMetadata, check_document_exists_with_filename, get_doc_id_from_filename
from sqlalchemy import select
from utils import check_data_folder_exists, create_data_folder, save_file_to_data_folder, delete_file
from typing import Optional
import json

# Define request/response models
class Request(BaseModel):
    message: str
    task_type: Optional[str] = "default"  # Task type for fine-tuned prompts
    evaluate_context: Optional[bool] = False  # Whether to evaluate context quality

class TaskTypeRequest(BaseModel):
    task_type: str

# 1. Global Variable for Agent
agent_instance: AgentDocument = None

# 2. Startup Event Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    print("🤖 Initializing Agent with MCP tools and connecting to Postgres...")
    # init DB (creates document_metadata first if not exist)
    init_db()
    # check data folder exists
    if not check_data_folder_exists():
        created = create_data_folder()
        if created:
            print("📁 Data folder created.")
        else:
            print("⚠️ Failed to create data folder.")
    # init agent with default task type
    agent_instance = AgentDocument(task_type="default")
    yield
    print("🛑 Shutting down...")

app = FastAPI(lifespan=lifespan)

# 3. CORS Middleware (Essential for Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask/")
async def ask_question(request: Request):
    """Ask a question to the agent using ReAct pattern. Returns response with reasoning and action steps."""
    try:
        global agent_instance
        if agent_instance is None:
            return {"error": "Agent not initialized", "status": "failed"}
        # If task type changed, reinitialize agent
        if request.task_type and request.task_type != agent_instance.task_type:
            print(f"🔄 Switching task type from {agent_instance.task_type} to {request.task_type}")
            agent_instance = AgentDocument(task_type=request.task_type)
        # If evaluation requested, use evaluation method (which includes reasoning/actions)
        if request.evaluate_context:
            result = await agent_instance.get_response_with_evaluation(request.message)
            return {
                "response": result["response"],
                "evaluation": result["evaluation"],
                "reasoning_steps": result.get("reasoning_steps", []),
                "action_steps": result.get("action_steps", []),
                "status": "success"
            }
        else:
            # Use get_response_with_reasoning to get reasoning and action steps
            result = await agent_instance.get_response_with_reasoning(request.message)
            return {
                "response": result["response"],
                "reasoning_steps": result.get("reasoning_steps", []),
                "action_steps": result.get("action_steps", []),
                "status": "success"
            }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.post("/ask-stream/")
async def ask_question_stream(request: Request):
    """Stream agent response with reasoning and action steps in real-time."""
    async def generate():
        try:
            global agent_instance
            if agent_instance is None:
                yield f"data: {json.dumps({'error': 'Agent not initialized', 'status': 'failed'})}\n\n"
                return
            
            # If task type changed, reinitialize agent
            if request.task_type and request.task_type != agent_instance.task_type:
                print(f"🔄 Switching task type from {agent_instance.task_type} to {request.task_type}")
                agent_instance = AgentDocument(task_type=request.task_type)
            
            # Stream response events (pass evaluate_context flag)
            async for event in agent_instance.stream_response(request.message, evaluate_context=request.evaluate_context):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'status': 'failed', 'type': 'error'})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/ask-with-evaluation/")
async def ask_with_evaluation(request: Request):
    """Ask a question and get context evaluation metrics with reasoning and action steps"""
    try:
        if not agent_instance:
            return {"error": "Agent not initialized", "status": "failed"}
        
        result = await agent_instance.get_response_with_evaluation(request.message)
        return {
            "response": result["response"],
            "evaluation": result["evaluation"],
            "reasoning_steps": result.get("reasoning_steps", []),
            "action_steps": result.get("action_steps", []),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.post("/set-task-type/")
async def set_task_type(request: TaskTypeRequest):
    """Change the agent's task type for fine-tuned prompts"""
    global agent_instance
    try:
        valid_task_types = ["default", "document_analysis", "research", "calculation", "general"]
        if request.task_type not in valid_task_types:
            return {
                "error": f"Invalid task type. Must be one of: {valid_task_types}",
                "status": "failed"
            }
        
        agent_instance = AgentDocument(task_type=request.task_type)
        return {
            "message": f"Task type set to: {request.task_type}",
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.get("/mcp-tools/")
async def list_mcp_tools():
    """List all available MCP tools"""
    try:
        from mcp_tools import mcp_registry
        tools = mcp_registry.list_tools()
        return {"tools": tools, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

# will use the same function for update file
# The io operation should update the file in the data folder
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # check file exists in DB first
        exists_in_db = check_document_exists_with_filename(file.filename)
        if exists_in_db:
            return {
                "filename": file.filename,
                "error": "File already exist in database. Please use update.",
                "status": "failed"
            }
        file_saved, file_path, error_msg = save_file_to_data_folder(file)
        if not file_saved:
            return {"error": error_msg, "status": f"writing file into {str(file_path)} failed"}

        # Update global agent with new file
        if agent_instance:
            store_doc_in_agent = agent_instance.add_file_context_to_agent(file_path)
            if store_doc_in_agent:
                return {
                    "filename": file.filename,
                    "message": "File saved and indexed successfully",
                    "file_path": str(file_path)
                }
            else:
                return {
                    "filename": file.filename,
                    "message": "File saved but indexing failed",
                    "file_path": str(file_path)
                }
        else:
            return {
                "filename": file.filename,
                "message": "File saved but agent not initialized",
                "file_path": str(file_path)
            }
    except Exception as e:
        return {"error": str(e), "status": "upload file failed"}

# update existing file
@app.put("/update/")
async def update_file(file: UploadFile = File(...)):
    try:
        # check file exists in DB first
        exists_in_db = check_document_exists_with_filename(file.filename)
        if not exists_in_db:
            return {
                "filename": file.filename,
                "message": "File does not exist in database. Please upload first.",
            }
        # get document_id from filename
        doc_id = get_doc_id_from_filename(file.filename)
        if not doc_id:
            return {
                "filename": file.filename,
                "message": "Could not retrieve document ID from filename.",
            }
        # save file to data folder (overwrite)
        file_saved, file_path, error_msg = save_file_to_data_folder(file)
        if not file_saved:
            return {"error": error_msg, "status": f"writing file into {str(file_path)} failed"}

        # Update vector index and session data with updated file
        if agent_instance:
            update_doc_in_agent = agent_instance.update_file_context_in_agent(file_path, doc_id)
            if update_doc_in_agent:
                return {
                    "filename": file.filename,
                    "message": "File updated and re-indexed successfully",
                    "file_path": str(file_path)
                }
            else:
                return {
                    "filename": file.filename,
                    "message": "File updated but re-indexing failed",
                    "file_path": str(file_path)
                }
        else:
            return {
                "filename": file.filename,
                "message": "File updated but agent not initialized",
                "file_path": str(file_path)
            }
    except Exception as e:
        return {"error": str(e), "status": "update file failed"}

# delete file endpoint
@app.delete("/delete/{filename}")
async def delete_file_endpoint(filename: str):
    try:
        # get document_id from filename
        doc_id = get_doc_id_from_filename(filename)
        if not doc_id:
            return {
                "filename": filename,
                "message": "Could not retrieve document ID from filename.",
            }
        # delete from vector store and metadata
        if agent_instance:
            delete_in_agent = agent_instance.delete_file_context_in_agent(doc_id)
            if not delete_in_agent:
                return {
                    "filename": filename,
                    "message": "Failed to delete from vector store.",
                }
        # delete file from disk
        delete_file(filename)
        return {
            "filename": filename,
            "message": "File deleted successfully from disk and vector store.",
        }
    except Exception as e:
        return {"error": str(e), "status": "delete file failed"}

@app.get("/files/")
async def list_files():
    try:
        # let's get the files list from the database.
        db = SessionLocal()
        # key file_name
        result = db.execute(select(DocumentMetadata.filename))
        # print("Fetched files from DB:", result)
        files = result.scalars().all()
        return {"files": files}
        # read from directory if needed
        # uploadpath = Path("../data")
        # uploadpath.mkdir(parents=True, exist_ok=True)
        # files = [f.name for f in uploadpath.iterdir() if f.is_file()]
        # return {"files": files}
    except Exception as e:
        return {"error": str(e), "status": "failed"}