import asyncio
import shlex
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

from app.models.tools import McpTool

class McpClient:
    def __init__(self, tool: McpTool):
        self.tool = tool
        self.config = tool.mcp_server_config
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()

    async def connect(self):
        """Connect to the MCP server depending on the transport (stdio or sse)"""
        server_url = self.config.server_url

        # If the URL is http or https, assume SSE transport
        if server_url.startswith("http://") or server_url.startswith("https://"):
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                sse_client(url=server_url)
            )
        else:
            # For stdio, we assume the server_url contains the command to run.
            # Example: "python server.py" or "npx -y @modelcontextprotocol/server-everything"
            parts = shlex.split(server_url)
            command = parts[0]
            args = parts[1:] if len(parts) > 1 else []
            server_params = StdioServerParameters(command=command, args=args)

            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server=server_params)
            )

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

    async def get_tools(self) -> List[Dict[str, Any]]:
        """Call tools/list to dynamically push the tool manifest to the LLM"""
        if not self._session:
            raise RuntimeError("MCP Client is not connected. Call connect() first.")

        tools_response = await self._session.list_tools()
        result = []
        for t in tools_response.tools:
            result.append({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema
            })
        return result

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server and return the result"""
        if not self._session:
            raise RuntimeError("MCP Client is not connected. Call connect() first.")

        result = await self._session.call_tool(tool_name, arguments)
        return result

    async def disconnect(self):
        """Disconnect from the MCP server"""
        await self._exit_stack.aclose()
        self._session = None
