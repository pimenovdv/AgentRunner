import pytest
import structlog
from app.logger import setup_logging
import logging

def test_setup_logging_console():
    setup_logging(json_logs=False)
    logger = structlog.get_logger()
    assert logger is not None

def test_setup_logging_json():
    setup_logging(json_logs=True)
    logger = structlog.get_logger()
    assert logger is not None
