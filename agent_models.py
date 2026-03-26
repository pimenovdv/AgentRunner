from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

class ExecutionLimits(BaseModel):
    max_tokens: int = Field(default=8000, description="Максимальное количество токенов")
    timeout_ms: int = Field(default=60000, description="Таймаут выполнения в миллисекундах")

class RestApiConfig(BaseModel):
    method: Literal["GET", "POST", "PATCH", "PUT", "DELETE"] = Field(description="HTTP метод (GET, POST, PATCH и др.)")
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

class Tool(BaseModel):
    name: str = Field(description="Имя инструмента")
    type: Literal["rest_api", "mcp_server", "kafka", "builtin"] = Field(description="Тип инструмента")
    description: Optional[str] = Field(default=None, description="Описание инструмента")
    rest_api_config: Optional[RestApiConfig] = Field(default=None, description="Конфигурация для REST API инструмента")
    mcp_server_config: Optional[McpServerConfig] = Field(default=None, description="Конфигурация для MCP сервера")
    kafka_config: Optional[KafkaConfig] = Field(default=None, description="Конфигурация для Kafka")
    builtin_config: Optional[BuiltinConfig] = Field(default=None, description="Конфигурация для встроенных инструментов")

class EdgeCondition(BaseModel):
    condition_expression: str = Field(description="Логическое выражение для перехода (skip logic)")

class Edge(BaseModel):
    source: str = Field(description="ID исходного узла")
    target: str = Field(description="ID целевого узла")
    condition: Optional[EdgeCondition] = Field(default=None, description="Условие перехода")

class Node(BaseModel):
    id: str = Field(description="Уникальный ID узла")
    type: Literal["reasoning", "tool_execution", "data_transformation"] = Field(description="Тип узла (логика выполнения)")
    tool_name: Optional[str] = Field(default=None, description="Имя инструмента (обязательно для типа tool_execution)")
    description: Optional[str] = Field(default=None, description="Описание шага микро-графа")

class MicroGraph(BaseModel):
    nodes: List[Node] = Field(description="Массив узлов (изолированных этапов выполнения)")
    edges: List[Edge] = Field(description="Массив ребер (направленных связей между узлами)")

class Prompts(BaseModel):
    system_instructions: str = Field(description="Системные инструкции (описание роли, формат рассуждений)")
    guardrails: Optional[str] = Field(default=None, description="Ограничения безопасности (опционально)")

class AgentManifest(BaseModel):
    input_schema: Dict[str, Any] = Field(description="Контракт входящих данных: JSON Schema (RFC 8259)")
    output_schema: Dict[str, Any] = Field(description="Контракт исходящих данных: JSON Schema (RFC 8259)")
    prompts: Prompts = Field(description="Системные промпты и ролевые модели")
    tools: List[Tool] = Field(default_factory=list, description="Набор доступных инструментов агенту")
    graph: MicroGraph = Field(description="Внутренняя машина состояний (микро-граф)")
    execution_limits: ExecutionLimits = Field(default_factory=ExecutionLimits, description="Ограничения по выполнению")

class AgentIndexDocument(BaseModel):
    agent_id: str = Field(description="Уникальный детерминированный идентификатор агента (например, finance_auditor_v1)")
    name: str = Field(description="Человекочитаемое имя агента для визуализации и лексического поиска")
    description: str = Field(description="Подробное описание компетенций, бизнес-логики и ограничений агента")
    capabilities_embedding: List[float] = Field(description="Плотный вектор (например, 1536 измерений), сгенерированный на основе поля description")
    dependencies: List[str] = Field(default_factory=list, description="Массив идентификаторов других агентов, от вывода которых зависит данный узел")
    manifest: AgentManifest = Field(description="Полная декларативная JSON-структура агента (manifest)")
