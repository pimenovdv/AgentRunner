import logging
import sys

import structlog

def setup_logging(json_logs: bool = False, log_level: str = "INFO"):
    """
    Configures structured logging for the application.
    Uses JSON formatting if json_logs is True, otherwise uses human-readable console output.
    """
    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(),
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # Disable duplicate logging by uvincorn/fastapi if present
    # logging.getLogger("uvicorn.error").handlers.clear()
    # logging.getLogger("uvicorn.error").propagate = True
