from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import StrEnum

class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class ToolCall(BaseModel):
    id: str = Field(description="Уникальный идентификатор вызова инструмента")
    name: str = Field(description="Имя вызываемого инструмента")
    arguments: Dict[str, Any] = Field(description="Аргументы, переданные в инструмент")

class Message(BaseModel):
    role: MessageRole = Field(description="Роль отправителя сообщения (system, user, assistant, tool)")
    content: Optional[str] = Field(default=None, description="Текстовое содержимое сообщения")
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="Список вызовов инструментов (для роли assistant)")
    tool_call_id: Optional[str] = Field(default=None, description="ID вызова инструмента, к которому относится ответ (для роли tool)")

class State(BaseModel):
    messages: List[Message] = Field(default_factory=list, description="История сообщений в графе")
    input_context: Dict[str, Any] = Field(default_factory=dict, description="Входной контекст данных для агента")
    # Дополнительные поля состояния могут быть добавлены позже по мере необходимости
