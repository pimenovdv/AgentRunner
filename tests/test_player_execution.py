import pytest
import asyncio
from fastapi.testclient import TestClient

from app.main import app
from app.models.api import ExecutionStatus

client = TestClient(app)

# Fixture representing a valid simple manifest
@pytest.fixture
def simple_manifest():
    return {
        "input_schema": {"type": "object", "properties": {"msg": {"type": "string"}}},
        "output_schema": {"type": "object"},
        "prompts": {"system_instructions": "You are a bot"},
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

def test_execute_player_success(simple_manifest):
    request_data = {
        "execution_id": "test_exec_001",
        "agent_manifest": simple_manifest,
        "input_context": {"msg": "hello"},
        "execution_limits": {"timeout_ms": 5000}
    }

    response = client.post("/api/v1/player/execute", json=request_data)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == ExecutionStatus.SUCCESS
    assert "execution_time_ms" in data["telemetry"]
    assert "used_tokens" in data["telemetry"]
    assert "called_tools" in data["telemetry"]
    # Due to missing llm_config in tests, it will use the fallback node which returns context + step_count

def test_execute_player_timeout(simple_manifest, monkeypatch):
    # To test timeout, we'll patch graph builder to create a graph that sleeps
    from app.services.graph_builder import GraphBuilder

    original_build = GraphBuilder.build

    def slow_build(self):
        graph = original_build(self)

        # Patch ainvoke of the compiled graph to simulate sleep
        original_ainvoke = graph.ainvoke

        async def slow_ainvoke(*args, **kwargs):
            await asyncio.sleep(0.5)
            return await original_ainvoke(*args, **kwargs)

        graph.ainvoke = slow_ainvoke
        return graph

    monkeypatch.setattr(GraphBuilder, "build", slow_build)

    request_data = {
        "execution_id": "test_exec_002",
        "agent_manifest": simple_manifest,
        "input_context": {"msg": "hello"},
        "execution_limits": {
            "timeout_ms": 100, # 100 ms timeout, should trigger since sleep is 0.5s
            "max_steps": 10
        }
    }

    response = client.post("/api/v1/player/execute", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == ExecutionStatus.TIMEOUT
    assert "Execution exceeded 100ms" in data["telemetry"]["error"]

def test_execute_player_telemetry(simple_manifest, monkeypatch):
    from app.services.graph_builder import GraphBuilder

    original_build = GraphBuilder.build

    def mock_build(self):
        graph = original_build(self)
        original_ainvoke = graph.ainvoke

        async def mock_ainvoke(state_dict, **kwargs):
            # Simulate state with telemetry data
            return {
                "input_context": state_dict.get("input_context", {}),
                "messages": [],
                "step_count": 3,
                "used_tokens": 150,
                "called_tools": ["test_tool_1", "test_tool_2"]
            }

        graph.ainvoke = mock_ainvoke
        return graph

    monkeypatch.setattr(GraphBuilder, "build", mock_build)

    request_data = {
        "execution_id": "test_exec_telemetry",
        "agent_manifest": simple_manifest,
        "input_context": {"msg": "telemetry_test"},
        "execution_limits": {"timeout_ms": 5000}
    }

    response = client.post("/api/v1/player/execute", json=request_data)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == ExecutionStatus.SUCCESS

    telemetry = data["telemetry"]
    assert "execution_time_ms" in telemetry
    assert telemetry["step_count"] == 3
    assert telemetry["used_tokens"] == 150
    assert telemetry["called_tools"] == ["test_tool_1", "test_tool_2"]
