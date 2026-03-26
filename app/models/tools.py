from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal, Union
from enum import StrEnum

class HttpMethod(StrEnum):
    GET = "GET"
    POST = "POST"
    PATCH = "PATCH"
    PUT = "PUT"
    DELETE = "DELETE"

class ToolType(StrEnum):
    BUILTIN = "builtin"
    REST_API = "rest_api"
    MCP_SERVER = "mcp_server"
    KAFKA = "kafka"

class RestApiConfig(BaseModel):
    method: HttpMethod = Field(description="HTTP метод (GET, POST, PATCH и др.)")
    base_url: str = Field(description="Базовый URL")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Конфигурация заголовков")
    authentication: Optional[Dict[str, Any]] = Field(default=None, description="Правила аутентификации")
    parameters_schema: Dict[str, Any] = Field(description="JSON Schema параметров, которые должна сгенерировать LLM (query/body)")

class McpServerConfig(BaseModel):
    server_url: str = Field(description="URL сервера MCP")

class KafkaConfig(BaseModel):
    topic: str = Field(description="Топик Kafka для публикации/подписки")
    bootstrap_servers: str = Field(description="Адреса серверов брокера")
    message_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON Schema сообщения")

class BuiltinConfig(BaseModel):
    function_name: str = Field(description="Имя внутренней функции/инструмента системы")

class BaseTool(BaseModel):
    name: str = Field(description="Имя инструмента")
    description: Optional[str] = Field(default=None, description="Описание инструмента")

class BuiltinTool(BaseTool):
    type: Literal[ToolType.BUILTIN] = Field(description="Тип инструмента")
    builtin_config: BuiltinConfig = Field(description="Конфигурация для встроенных инструментов")

class RestApiTool(BaseTool):
    type: Literal[ToolType.REST_API] = Field(description="Тип инструмента")
    rest_api_config: RestApiConfig = Field(description="Конфигурация для REST API инструмента")

class McpTool(BaseTool):
    type: Literal[ToolType.MCP_SERVER] = Field(description="Тип инструмента")
    mcp_server_config: McpServerConfig = Field(description="Конфигурация для MCP сервера")

class KafkaTool(BaseTool):
    type: Literal[ToolType.KAFKA] = Field(description="Тип инструмента")
    kafka_config: KafkaConfig = Field(description="Конфигурация для Kafka")

Tool = Union[BuiltinTool, RestApiTool, McpTool, KafkaTool]
