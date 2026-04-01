import pytest
from pydantic import ValidationError
from app.models.manifest import AgentManifest, ExecutionLimits, Prompts
from app.models.tools import ToolType, HttpMethod, RestApiTool, McpTool, KafkaTool

def test_manifest_parsing_valid():
    manifest_data = {
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "prompts": {"system_instructions": "You are a test bot"},
        "tools": [
            {
                "name": "rest_tool",
                "description": "Calls a REST API",
                "type": ToolType.REST_API,
                "rest_api_config": {
                    "method": HttpMethod.GET,
                    "base_url": "https://api.example.com",
                    "parameters_schema": {"type": "object"}
                }
            },
            {
                "name": "mcp_tool",
                "type": ToolType.MCP_SERVER,
                "mcp_server_config": {
                    "server_url": "http://mcp:8080"
                }
            },
            {
                "name": "kafka_tool",
                "type": ToolType.KAFKA,
                "kafka_config": {
                    "topic": "test_topic",
                    "bootstrap_servers": "localhost:9092"
                }
            }
        ],
        "graph": {
            "nodes": [{"id": "1", "type": "reasoning"}],
            "edges": []
        },
        "execution_limits": {
            "max_tokens": 1000,
            "timeout_ms": 10000
        }
    }

    manifest = AgentManifest(**manifest_data)
    assert manifest.prompts.system_instructions == "You are a test bot"
    assert len(manifest.tools) == 3

    assert isinstance(manifest.tools[0], RestApiTool)
    assert manifest.tools[0].rest_api_config.method == HttpMethod.GET

    assert isinstance(manifest.tools[1], McpTool)
    assert manifest.tools[1].mcp_server_config.server_url == "http://mcp:8080"

    assert isinstance(manifest.tools[2], KafkaTool)
    assert manifest.tools[2].kafka_config.topic == "test_topic"

def test_rest_api_tool_validation():
    # Missing required field in rest_api_config
    invalid_tool = {
        "name": "invalid_rest",
        "type": ToolType.REST_API,
        "rest_api_config": {
            "method": "GET",
            "base_url": "https://api.example.com"
            # Missing parameters_schema
        }
    }

    with pytest.raises(ValidationError):
        RestApiTool(**invalid_tool)

def test_kafka_tool_validation():
    # Missing required field in kafka_config
    invalid_tool = {
        "name": "invalid_kafka",
        "type": ToolType.KAFKA,
        "kafka_config": {
            "topic": "test_topic"
            # Missing bootstrap_servers
        }
    }

    with pytest.raises(ValidationError):
        KafkaTool(**invalid_tool)

def test_manifest_missing_required():
    # Missing prompts
    manifest_data = {
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "tools": [],
        "graph": {
            "nodes": [],
            "edges": []
        }
    }

    with pytest.raises(ValidationError):
        AgentManifest(**manifest_data)
