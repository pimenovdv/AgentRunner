import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.api import ExecutionStatus
from app.config import settings

client = TestClient(app)

@pytest.fixture
def manifest_with_schema():
    return {
        "input_schema": {"type": "object", "properties": {"data": {"type": "string"}}},
        "output_schema": {
            "title": "FinalResult",
            "type": "object",
            "properties": {
                "extracted_value": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["extracted_value", "confidence"]
        },
        "prompts": {"system_instructions": "You are a test bot"},
        "tools": [],
        "graph": {
            "nodes": [{"id": "node_1", "type": "reasoning"}],
            "edges": []
        },
        "execution_limits": {
            "max_tokens": 100,
            "timeout_ms": 60000,
            "max_steps": 10
        }
    }

def test_structured_output_enforcement(manifest_with_schema, monkeypatch):
    # Set settings so that it tries to use OpenAI
    monkeypatch.setattr(settings, "openai_api_key", "mock_key")

    # Mock ChatOpenAI
    class MockStructuredLLM:
        async def ainvoke(self, messages):
            # This simulates the structured output return
            return {"extracted_value": "mocked_data", "confidence": 0.99}

    class MockChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass
        def with_structured_output(self, schema):
            self.schema = schema
            return MockStructuredLLM()
        async def ainvoke(self, messages):
            class Resp:
                content = "dummy"
                tool_calls = []
            return Resp()
        def bind_tools(self, tools):
            return self

    monkeypatch.setattr("app.api.routes.player.ChatOpenAI", MockChatOpenAI)
    monkeypatch.setattr("app.services.graph_builder.ChatOpenAI", MockChatOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "mock_key")
    monkeypatch.setattr("app.services.graph_builder.ChatOpenAI", MockChatOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "mock_key")
    monkeypatch.setattr("app.services.graph_builder.ChatOpenAI", MockChatOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "mock_key")
    monkeypatch.setattr("app.services.graph_builder.ChatOpenAI", MockChatOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "mock_key")

    request_data = {
        "execution_id": "test_struct_001",
        "agent_manifest": manifest_with_schema,
        "input_context": {"data": "some context"},
        "execution_limits": {"timeout_ms": 5000}
    }

    response = client.post("/api/v1/player/execute", json=request_data)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == ExecutionStatus.SUCCESS
    assert data["output_data"] == {"extracted_value": "mocked_data", "confidence": 0.99}
