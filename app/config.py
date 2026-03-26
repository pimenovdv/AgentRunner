from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Kafka Settings
    kafka_brokers: str | None = None
    kafka_topic: str | None = None
    kafka_role: str | None = None

    # Execution Limits (Timeouts, tokens)
    timeout_ms: int = 30000
    max_tokens: int = 4096

    # Other settings
    environment: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
