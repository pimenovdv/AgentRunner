from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.models.tools import Tool
from app.models.graph import MicroGraph

class ExecutionLimits(BaseModel):
    max_tokens: int = Field(default=8000, description="Максимальное количество токенов")
    timeout_ms: int = Field(default=60000, description="Таймаут выполнения в миллисекундах")

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
