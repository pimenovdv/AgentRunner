from pydantic import BaseModel, Field

class RetryPolicy(BaseModel):
    max_retries: int = Field(default=3, description="Максимальное количество попыток повторения (retries)")
    backoff_factor: float = Field(default=2.0, description="Коэффициент экспоненциальной задержки (Exponential Backoff)")
