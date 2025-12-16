"""
MCP (Model Context Protocol) Tools Integration
Provides MCP-compatible tools that can be used by the agent.
"""
from typing import Any, Dict, List
import json
import os
from pathlib import Path
import re


class MCPTool:
    """Base class for MCP tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def get_tool_spec(self) -> Dict[str, Any]:
        """Get the tool specification in MCP format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.get_input_schema()
        }
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get the input schema for the tool."""
        raise NotImplementedError


class FileSystemMCPTool(MCPTool):
    """MCP tool for file system operations and URL reading."""
    
    def __init__(self):
        super().__init__(
            name="read_file",
            description=(
                "Read the contents of a file from the local data directory or from a URL/website. "
                "Supports both local file paths (relative to data directory) and HTTP/HTTPS URLs."
            )
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": (
                        "Path to the file relative to the data directory, or a URL (http:// or https://) "
                        "to read content from a website. Examples: 'document.txt' or 'https://example.com/page.html'"
                    )
                }
            },
            "required": ["filepath"]
        }
    
    def _is_url(self, path: str) -> bool:
        """Check if the given path is a URL."""
        url_pattern = re.compile(r'^https?://', re.IGNORECASE)
        return bool(url_pattern.match(path))
    
    async def _fetch_url_content(self, url: str) -> str:
        """Fetch content from a URL."""
        # Use httpx for async HTTP requests (preferred)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Try to decode as text, with fallback for encoding issues
                try:
                    content = response.text
                except UnicodeDecodeError:
                    # If UTF-8 fails, try to decode with detected encoding
                    content = response.content.decode('utf-8', errors='replace')
                
                return content
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_text = e.response.text[:200] if e.response.text else "No error details"
            return f"Error: HTTP {status_code} error fetching URL {url}: {error_text}"
        except httpx.TimeoutException:
            return f"Error: Timeout while fetching URL {url} (timeout: 30 seconds)"
        except ImportError:
            # Fallback to requests if httpx is not available (synchronous)
            try:
                import requests
                response = requests.get(url, timeout=30, allow_redirects=True)
                response.raise_for_status()
                return response.text
            except requests.HTTPError as e:
                return f"Error: HTTP {response.status_code} error fetching URL {url}: {str(e)[:200]}"
            except requests.Timeout:
                return f"Error: Timeout while fetching URL {url} (timeout: 30 seconds)"
            except ImportError:
                return f"Error: Neither httpx nor requests library is installed. Please install one with: pip install httpx (recommended) or pip install requests"
        except Exception as e:
            return f"Error fetching URL {url}: {str(e)}"
    
    async def execute(self, filepath: str) -> str:
        """Execute the file read operation (local file or URL)."""
        try:
            # Check if it's a URL
            if self._is_url(filepath):
                content = await self._fetch_url_content(filepath)
                return f"Content from URL {filepath}:\n{content}"
            
            # Otherwise, treat it as a local file path
            data_path = Path("./data") / filepath
            if not data_path.exists():
                return f"Error: File {filepath} not found in data directory"
            with open(data_path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"File contents of {filepath}:\n{content}"
        except Exception as e:
            return f"Error reading file/URL {filepath}: {str(e)}"


class WebSearchMCPTool(MCPTool):
    """MCP tool for web search using DuckDuckGo (no API key required)."""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description=(
                "PRIORITY 2: Search the web for current information using DuckDuckGo. "
                "ONLY use this tool AFTER trying search_documents first. "
                "Use when search_documents doesn't provide relevant or sufficient information. "
                "Returns up to 5 relevant search results with titles, URLs, and snippets."
            )
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to look up on the web"
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, query: str) -> str:
        """Execute web search using DuckDuckGo."""
        try:
            from ddgs import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                
                if not results:
                    return f"No search results found for '{query}'"
                
                formatted_results = f"Web search results for '{query}':\n\n"
                for i, result in enumerate(results, 1):
                    title = result.get('title', 'No title')
                    url = result.get('href', 'No URL')
                    body = result.get('body', 'No description')
                    
                    formatted_results += f"{i}. {title}\n"
                    formatted_results += f"   URL: {url}\n"
                    formatted_results += f"   {body}\n\n"
                
                return formatted_results.strip()
                
        except ImportError:
            return f"Error: ddgs library not installed. Please install it with: pip install ddgs"
        except Exception as e:
            return f"Error searching the web for '{query}': {str(e)}"


class CalculatorMCPTool(MCPTool):
    """MCP tool for mathematical calculations."""
    
    def __init__(self):
        super().__init__(
            name="calculate",
            description="Perform mathematical calculations"
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        }
    
    async def execute(self, expression: str) -> str:
        """Execute calculation."""
        try:
            # Safe evaluation of mathematical expressions
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"


class MCPToolRegistry:
    """Registry for managing MCP tools."""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default MCP tools."""
        self.register_tool(FileSystemMCPTool())
        self.register_tool(WebSearchMCPTool())
        self.register_tool(CalculatorMCPTool())
    
    def register_tool(self, tool: MCPTool):
        """Register a new MCP tool."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> MCPTool:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [tool.get_tool_spec() for tool in self.tools.values()]


# Global registry instance
mcp_registry = MCPToolRegistry()


def create_llamaindex_tool_from_mcp(mcp_tool: MCPTool):
    """Convert an MCP tool to a LlamaIndex-compatible tool."""
    from llama_index.core.tools import FunctionTool
    import inspect
    
    # Get the execute method signature
    sig = inspect.signature(mcp_tool.execute)
    param_names = list(sig.parameters.keys())
    
    # Create a wrapper function with the correct signature
    if len(param_names) == 1:
        # Single parameter case
        param_name = param_names[0]
        async def tool_wrapper(**kwargs) -> str:
            """Wrapper function for MCP tool execution."""
            if param_name in kwargs:
                return await mcp_tool.execute(kwargs[param_name])
            else:
                return f"Error: Missing required parameter '{param_name}'"
    else:
        # Multiple parameters or no parameters
        async def tool_wrapper(**kwargs) -> str:
            """Wrapper function for MCP tool execution."""
            # Extract parameters in order
            args = [kwargs.get(name) for name in param_names]
            # Filter out None values if they're optional
            filtered_args = [arg for arg in args if arg is not None]
            return await mcp_tool.execute(*filtered_args)
    
    tool_wrapper.__signature__ = sig
    tool_wrapper.__name__ = mcp_tool.name
    tool_wrapper.__doc__ = mcp_tool.description
    
    return FunctionTool.from_defaults(
        fn=tool_wrapper,
        name=mcp_tool.name,
        description=mcp_tool.description
    )