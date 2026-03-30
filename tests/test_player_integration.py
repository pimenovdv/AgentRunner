import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.api import ExecutionStatus
from app.config import settings
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import AIMessage

client = TestClient(app)

@pytest.fixture
def base_manifest():
    return {
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        "output_schema": {
            "title": "FinalResult",
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"]
        },
        "prompts": {"system_instructions": "You are a test agent."},
        "tools": [],
        "graph": {
            "nodes": [{"id": "node_1", "type": "reasoning"}],
            "edges": []
        },
        "execution_limits": {
            "max_tokens": 1000,
            "timeout_ms": 60000,
            "max_steps": 10
        }
    }

def test_execute_player_with_retry(base_manifest, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "mock_key")

    call_count = 0

    class MockChatOpenAIWithTool:
        def __init__(self, *args, **kwargs):
            self.call_count = 0

        async def ainvoke(self, messages):
            # This is called by _invoke_llm_with_retry
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate a transient error during LLM call which should trigger tenacity retry
                import httpx
                response = MagicMock(); response.status_code = 503; raise httpx.HTTPStatusError("503 Error", request=MagicMock(), response=response)

            # On second call, return success
            class Resp:
                content = "Done!"
                tool_calls = []
                response_metadata = {"token_usage": {"total_tokens": 10}}
            return Resp()

        def bind_tools(self, tools):
            return self

        def with_structured_output(self, schema):
            class StructuredLLM:
                async def ainvoke(self, messages):
                    return {"result": f"Done! call count: {call_count}"}
            return StructuredLLM()

    monkeypatch.setattr("app.api.routes.player.ChatOpenAI", MockChatOpenAIWithTool)
    monkeypatch.setattr("app.services.graph_builder.ChatOpenAI", MockChatOpenAIWithTool)

    request_data = {
        "execution_id": "test_retry_001",
        "agent_manifest": base_manifest,
        "input_context": {"query": "Retry this"},
        "execution_limits": {"timeout_ms": 10000}
    }

    # Should succeed after retrying
    response = client.post("/api/v1/player/execute", json=request_data)

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == ExecutionStatus.SUCCESS
    assert call_count == 2
    assert "Done! call count: 2" in data["output_data"]["result"]

def test_execute_player_self_correction(base_manifest, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "mock_key")

    base_manifest["tools"] = [{
        "name": "mock_builtin_tool",
        "description": "Mock tool",
        "type": "builtin",
        "builtin_config": {"function_name": "calculator"}
    }]

    class MockChatOpenAIForSelfCorrection:
        def __init__(self, *args, **kwargs):
            self.call_count = 0

        async def ainvoke(self, messages):
            self.call_count += 1

            class ToolCallMessage:
                def __init__(self, tool_calls, content=""):
                    self.tool_calls = tool_calls
                    self.content = content
                    self.response_metadata = {"token_usage": {"total_tokens": 10}}

            if self.call_count == 1:
                return ToolCallMessage([{
                    "id": "call_bad_schema",
                    "name": "mock_builtin_tool",
                    "args": {"invalid_arg": "value"}
                }])
            else:
                return ToolCallMessage([], content="Final answer after correction")

        def bind_tools(self, tools):
            return self

        def with_structured_output(self, schema):
            class StructuredLLM:
                async def ainvoke(self, messages):
                    return {"result": "Corrected Result"}
            return StructuredLLM()

    monkeypatch.setattr("app.api.routes.player.ChatOpenAI", MockChatOpenAIForSelfCorrection)
    monkeypatch.setattr("app.services.graph_builder.ChatOpenAI", MockChatOpenAIForSelfCorrection)

    request_data = {
        "execution_id": "test_self_correct_002",
        "agent_manifest": base_manifest,
        "input_context": {"query": "Use the tool incorrectly"},
        "execution_limits": {"timeout_ms": 10000}
    }

    response = client.post("/api/v1/player/execute", json=request_data)

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == ExecutionStatus.SUCCESS
    assert data["output_data"]["result"] == "Corrected Result"
