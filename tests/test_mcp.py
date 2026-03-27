import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from app.models.tools import McpTool, McpServerConfig, ToolType
from app.services.mcp import McpClient

@pytest.fixture
def mcp_tool_stdio():
    return McpTool(
        name="test_mcp_tool_stdio",
        description="A test tool using stdio",
        type=ToolType.MCP_SERVER,
        mcp_server_config=McpServerConfig(
            server_url="python mock_server.py arg1"
        )
    )

@pytest.fixture
def mcp_tool_sse():
    return McpTool(
        name="test_mcp_tool_sse",
        description="A test tool using sse",
        type=ToolType.MCP_SERVER,
        mcp_server_config=McpServerConfig(
            server_url="http://mock_server.com/sse"
        )
    )

class MockTool:
    def __init__(self, name, description, inputSchema):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema

class MockToolsResponse:
    def __init__(self, tools):
        self.tools = tools

@pytest.mark.asyncio
async def test_mcp_client_connect_stdio(mcp_tool_stdio):
    client = McpClient(tool=mcp_tool_stdio)

    mock_stdio_client = MagicMock()
    mock_stdio_client.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    mock_stdio_client.__aexit__ = AsyncMock()

    mock_client_session = MagicMock()
    mock_session_instance = MagicMock()
    mock_session_instance.initialize = AsyncMock()
    mock_client_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
    mock_client_session.__aexit__ = AsyncMock()

    with patch('app.services.mcp.stdio_client', return_value=mock_stdio_client) as mock_stdio, \
         patch('app.services.mcp.ClientSession', return_value=mock_client_session) as mock_session:

        await client.connect()

        mock_stdio.assert_called_once()
        server_params = mock_stdio.call_args[1]['server']
        assert server_params.command == "python"
        assert server_params.args == ["mock_server.py", "arg1"]

        mock_session.assert_called_once()
        mock_session_instance.initialize.assert_awaited_once()

    await client.disconnect()

@pytest.mark.asyncio
async def test_mcp_client_connect_sse(mcp_tool_sse):
    client = McpClient(tool=mcp_tool_sse)

    mock_sse_client = MagicMock()
    mock_sse_client.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    mock_sse_client.__aexit__ = AsyncMock()

    mock_client_session = MagicMock()
    mock_session_instance = MagicMock()
    mock_session_instance.initialize = AsyncMock()
    mock_client_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
    mock_client_session.__aexit__ = AsyncMock()

    with patch('app.services.mcp.sse_client', return_value=mock_sse_client) as mock_sse, \
         patch('app.services.mcp.ClientSession', return_value=mock_client_session) as mock_session:

        await client.connect()

        mock_sse.assert_called_once_with(url="http://mock_server.com/sse")

        mock_session.assert_called_once()
        mock_session_instance.initialize.assert_awaited_once()

    await client.disconnect()

@pytest.mark.asyncio
async def test_mcp_client_get_tools(mcp_tool_sse):
    client = McpClient(tool=mcp_tool_sse)

    # Mock connection first
    client._session = MagicMock()
    mock_tool = MockTool(name="test_tool", description="test desc", inputSchema={"type": "object"})
    client._session.list_tools = AsyncMock(return_value=MockToolsResponse(tools=[mock_tool]))

    tools = await client.get_tools()

    assert len(tools) == 1
    assert tools[0]["name"] == "test_tool"
    assert tools[0]["description"] == "test desc"
    assert tools[0]["inputSchema"] == {"type": "object"}

    client._session.list_tools.assert_awaited_once()

@pytest.mark.asyncio
async def test_mcp_client_get_tools_unconnected(mcp_tool_sse):
    client = McpClient(tool=mcp_tool_sse)
    with pytest.raises(RuntimeError):
        await client.get_tools()

@pytest.mark.asyncio
async def test_mcp_client_call_tool(mcp_tool_sse):
    client = McpClient(tool=mcp_tool_sse)

    # Mock connection first
    client._session = MagicMock()
    client._session.call_tool = AsyncMock(return_value={"result": "success"})

    result = await client.call_tool("test_tool", {"arg1": "value1"})

    assert result == {"result": "success"}
    client._session.call_tool.assert_awaited_once_with("test_tool", {"arg1": "value1"})

@pytest.mark.asyncio
async def test_mcp_client_call_tool_unconnected(mcp_tool_sse):
    client = McpClient(tool=mcp_tool_sse)
    with pytest.raises(RuntimeError):
        await client.call_tool("test_tool", {"arg1": "value1"})
