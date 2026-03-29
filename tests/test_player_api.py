import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_execute_player_success():
    payload = {
        "execution_id": "test-exec-1",
        "agent_manifest": {
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"]
            },
            "output_schema": {
                "type": "object"
            },
            "prompts": {
                "system_instructions": "You are a test agent."
            },
            "tools": [],
            "graph": {
                "nodes": [],
                "edges": []
            },
            "execution_limits": {
                "max_tokens": 1000,
                "timeout_ms": 10000,
                "max_steps": 10
            }
        },
        "input_context": {
            "question": "What is the capital of France?"
        },
        "execution_limits": {
            "max_tokens": 1000,
            "timeout_ms": 10000,
            "max_steps": 10
        }
    }

    response = client.post("/api/v1/player/execute", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["output_data"] == {}
    assert "telemetry" in data


def test_execute_player_validation_error():
    payload = {
        "execution_id": "test-exec-2",
        "agent_manifest": {
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"]
            },
            "output_schema": {
                "type": "object"
            },
            "prompts": {
                "system_instructions": "You are a test agent."
            },
            "tools": [],
            "graph": {
                "nodes": [],
                "edges": []
            },
            "execution_limits": {
                "max_tokens": 1000,
                "timeout_ms": 10000,
                "max_steps": 10
            }
        },
        "input_context": {
            "answer": 42  # Fails schema validation because 'question' is required
        },
        "execution_limits": {
            "max_tokens": 1000,
            "timeout_ms": 10000,
            "max_steps": 10
        }
    }

    response = client.post("/api/v1/player/execute", json=payload)
    assert response.status_code == 400
    assert "Input validation failed" in response.json()["detail"]
