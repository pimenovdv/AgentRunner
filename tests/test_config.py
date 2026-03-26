from app.config import Settings

def test_settings_default_values():
    settings = Settings()
    assert settings.timeout_ms == 30000
    assert settings.max_tokens == 4096
    assert settings.environment == "development"
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None
    assert settings.kafka_brokers is None
    assert settings.kafka_topic is None
    assert settings.kafka_role is None

def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    monkeypatch.setenv("TIMEOUT_MS", "5000")
    monkeypatch.setenv("ENVIRONMENT", "production")

    settings = Settings()

    assert settings.openai_api_key == "test-key"
    assert settings.kafka_brokers == "localhost:9092"
    assert settings.timeout_ms == 5000
    assert settings.environment == "production"
