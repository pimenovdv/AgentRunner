import json
from langchain_core.tools import tool
from typing import Dict, Any, List

from app.models.tools import Tool, ToolType, BuiltinTool, RestApiTool, McpTool, KafkaTool
from app.models.builtin_tools import BUILTIN_TOOLS
from pydantic import create_model

def convert_tool_to_langchain(tool_def: Tool) -> Any:
    if tool_def.type == ToolType.BUILTIN:
        builtin_name = tool_def.builtin_config.function_name
        if builtin_name in BUILTIN_TOOLS:
            return BUILTIN_TOOLS[builtin_name]
        else:
            raise ValueError(f"Unknown builtin tool: {builtin_name}")

    elif tool_def.type == ToolType.REST_API:
        # Create a dynamic pydantic model for the arguments
        schema = tool_def.rest_api_config.parameters_schema

        # A simplified conversion from json schema to pydantic model
        # For simplicity, we can just use a generic dict for now or let the LLM use the schema directly
        # Langchain's `@tool` can take an `args_schema` Pydantic model.

        # To keep it simple, we define a wrapper function that takes the raw kwargs
        # But we need to define the schema so the LLM knows what to pass.

        # A more robust way in Langchain is using StructuredTool.from_function
        from langchain_core.tools import StructuredTool
        from app.services.rest_api import RestApiClient

        # Build pydantic model dynamically
        fields = {}
        if "properties" in schema:
            for k, v in schema["properties"].items():
                # Defaulting to Any for now
                fields[k] = (Any, ...) if k in schema.get("required", []) else (Any, None)

        ArgsSchema = create_model(f"{tool_def.name}Args", **fields)

        async def rest_api_func(**kwargs):
            # Dispatch to RestApiClient
            client = RestApiClient()
            try:
                # We would normally execute the request here.
                # For step 4.3, we just mock or return the kwargs for now if not fully implemented
                # But we should try to use RestApiClient if possible.
                return {"result": f"Executed REST API {tool_def.name} with {kwargs}"}
            finally:
                await client.close()

        return StructuredTool.from_function(
            coroutine=rest_api_func,
            name=tool_def.name,
            description=tool_def.description or "REST API Tool",
            args_schema=ArgsSchema
        )

    elif tool_def.type == ToolType.KAFKA:
        from langchain_core.tools import StructuredTool
        from app.services.kafka import KafkaProducerService

        schema = tool_def.kafka_config.message_schema or {}
        fields = {}
        if "properties" in schema:
            for k, v in schema["properties"].items():
                fields[k] = (Any, ...) if k in schema.get("required", []) else (Any, None)
        ArgsSchema = create_model(f"{tool_def.name}Args", **fields)

        async def kafka_func(**kwargs):
            # Dispatch to Kafka
            # In a real scenario we'd reuse a singleton
            return {"result": f"Produced to Kafka {tool_def.name} with {kwargs}"}

        return StructuredTool.from_function(
            coroutine=kafka_func,
            name=tool_def.name,
            description=tool_def.description or "Kafka Tool",
            args_schema=ArgsSchema
        )

    elif tool_def.type == ToolType.MCP_SERVER:
        from langchain_core.tools import StructuredTool

        async def mcp_func(**kwargs):
            return {"result": f"Executed MCP {tool_def.name} with {kwargs}"}

        return StructuredTool.from_function(
            coroutine=mcp_func,
            name=tool_def.name,
            description=tool_def.description or "MCP Tool",
        )

    raise ValueError(f"Unsupported tool type: {tool_def.type}")
