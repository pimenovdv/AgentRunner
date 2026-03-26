import pytest
from pydantic import ValidationError
from app.models.api import ExecuteRequest, ExecuteResponse
from app.models.manifest import AgentManifest, ExecutionLimits, Prompts
from app.models.graph import MicroGraph, Node

def get_valid_manifest_dict():
    return {
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        "output_schema": {"type": "object", "properties": {"result": {"type": "string"}}},
        "prompts": {
            "system_instructions": "You are a test agent."
        },
        "tools": [],
        "graph": {
            "nodes": [
                {
                    "id": "node_1",
                    "type": "reasoning"
                }
            ],
            "edges": []
        },
        "execution_limits": {
            "max_tokens": 1000,
            "timeout_ms": 30000
        }
    }

def test_execute_request_valid():
    data = {
        "execution_id": "test-id-123",
        "agent_manifest": get_valid_manifest_dict(),
        "input_context": {"query": "Hello"},
        "execution_limits": {
            "max_tokens": 500,
            "timeout_ms": 10000
        }
    }

    request = ExecuteRequest(**data)
    assert request.execution_id == "test-id-123"
    assert request.input_context == {"query": "Hello"}
    assert request.execution_limits.max_tokens == 500
    assert request.execution_limits.timeout_ms == 10000
    assert request.agent_manifest.prompts.system_instructions == "You are a test agent."
    assert len(request.agent_manifest.graph.nodes) == 1

def test_execute_request_missing_required():
    data = {
        # missing execution_id
        "agent_manifest": get_valid_manifest_dict(),
        "input_context": {"query": "Hello"},
    }

    with pytest.raises(ValidationError) as exc_info:
        ExecuteRequest(**data)

    assert "execution_id" in str(exc_info.value)

def test_execute_response_valid():
    data = {
        "status": "success",
        "output_data": {"result": "Done"},
        "telemetry": {"tokens_used": 150, "time_ms": 1200}
    }

    response = ExecuteResponse(**data)
    assert response.status == "success"
    assert response.output_data == {"result": "Done"}
    assert response.telemetry == {"tokens_used": 150, "time_ms": 1200}

def test_execute_response_invalid_status():
    data = {
        "status": "pending", # Invalid literal
        "output_data": {"result": "Done"},
        "telemetry": {}
    }

    with pytest.raises(ValidationError) as exc_info:
        ExecuteResponse(**data)

    assert "Input should be 'success', 'failed', 'timeout' or 'error'" in str(exc_info.value)

def test_execute_request_default_limits():
    data = {
        "execution_id": "test-id-123",
        "agent_manifest": get_valid_manifest_dict(),
        "input_context": {"query": "Hello"}
    }

    request = ExecuteRequest(**data)
    # Checks if default execution_limits are applied
    assert request.execution_limits.max_tokens == 8000
    assert request.execution_limits.timeout_ms == 60000
