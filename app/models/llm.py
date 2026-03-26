from pydantic import BaseModel, Field
from enum import StrEnum

class LlmProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class LlmConfig(BaseModel):
    provider: LlmProvider = Field(description="Провайдер LLM (например, openai, anthropic)")
    model_name: str = Field(description="Имя модели (например, gpt-4o, claude-3-opus)")
    temperature: float = Field(default=0.7, description="Температура генерации (от 0 до 1)")
