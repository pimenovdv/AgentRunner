import structlog
from app.config import settings
from app.logger import setup_logging

def main():
    # Настройка логгера
    is_prod = settings.environment.lower() == "production"
    setup_logging(json_logs=is_prod)
    logger = structlog.get_logger()

    logger.info("app_starting", env=settings.environment)
    logger.debug("config_loaded",
                 timeout_ms=settings.timeout_ms,
                 max_tokens=settings.max_tokens)

    logger.info("Hello from app!")

if __name__ == "__main__":
    main()
