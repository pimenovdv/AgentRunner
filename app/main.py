import structlog
from fastapi import FastAPI
from app.config import settings
from app.logger import setup_logging
from app.api.routes import player

def create_app() -> FastAPI:
    # Настройка логгера
    is_prod = settings.environment.lower() == "production"
    setup_logging(json_logs=is_prod)
    logger = structlog.get_logger()

    logger.info("app_starting", env=settings.environment)
    logger.debug("config_loaded",
                 timeout_ms=settings.timeout_ms,
                 max_tokens=settings.max_tokens)

    app = FastAPI(
        title="Runner Service",
        description="Execution Engine / Player for Multi-Agent Platform",
        version="0.1.0"
    )

    # Подключение роутеров
    app.include_router(player.router, prefix="/api/v1/player", tags=["Player"])

    return app

app = create_app()

def main():
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
