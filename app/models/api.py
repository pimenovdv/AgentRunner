from pydantic import BaseModel, Field
from typing import Dict, Any, Literal
from enum import StrEnum

from app.models.manifest import AgentManifest, ExecutionLimits

class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"

class ExecuteRequest(BaseModel):
    execution_id: str = Field(description="Уникальный идентификатор запроса на выполнение")
    agent_manifest: AgentManifest = Field(description="Манифест агента для выполнения")
    input_context: Dict[str, Any] = Field(description="Контекст и входные данные для агента")
    execution_limits: ExecutionLimits = Field(default_factory=ExecutionLimits, description="Ограничения выполнения (переопределяют те, что в манифесте)")

class ExecuteResponse(BaseModel):
    status: ExecutionStatus = Field(description="Статус выполнения запроса")
    output_data: Dict[str, Any] = Field(description="Результат работы агента")
    telemetry: Dict[str, Any] = Field(description="Телеметрия (потребленные токены, время выполнения и т.д.)")
