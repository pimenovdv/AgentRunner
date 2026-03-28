import structlog
from typing import Any, Callable, TypeVar, cast
import httpx
from tenacity import retry, retry_if_exception, wait_exponential, stop_after_attempt
from mcp.shared.exceptions import McpError

# Try to import LLM errors if possible to handle them too
try:
    from openai import APIStatusError
except ImportError:
    class APIStatusError(Exception):
        pass

try:
    from anthropic import APIStatusError as AnthropicAPIStatusError
except ImportError:
    class AnthropicAPIStatusError(Exception):
        pass

logger = structlog.get_logger()

def _is_retryable_error(exception: BaseException) -> bool:
    """Check if the exception is a retryable network/API error (429 or 50x)."""

    # 1. HTTPX errors
    if isinstance(exception, httpx.HTTPStatusError):
        status = exception.response.status_code
        return status == 429 or 500 <= status < 600

    # 2. OpenAI / Anthropic errors
    if isinstance(exception, APIStatusError):
        status = getattr(exception, "status_code", None)
        if status is not None:
            return status == 429 or 500 <= status < 600

    if isinstance(exception, AnthropicAPIStatusError):
        status = getattr(exception, "status_code", None)
        if status is not None:
            return status == 429 or 500 <= status < 600

    # 3. MCP Errors (JSON-RPC)
    if isinstance(exception, McpError):
        # MCP/JSON-RPC error codes are often negative or standard HTTP-like codes inside the payload
        # Standard json-rpc codes: -32603 is Internal Error.
        # But we'll retry internal errors and standard HTTP 5xx codes if they are wrapped.
        code = exception.error.code
        return code == -32603 or code == 429 or 500 <= code < 600

    return False

def _log_retry(retry_state: Any) -> None:
    logger.warning(
        "retrying_operation",
        attempt=retry_state.attempt_number,
        error=str(retry_state.outcome.exception()),
        func=retry_state.fn.__name__
    )

T = TypeVar('T', bound=Callable[..., Any])

def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0
) -> Callable[[T], T]:
    """
    Decorator that applies exponential backoff retry logic for 429 and 50x errors.
    """
    def decorator(func: T) -> T:
        return cast(T, retry(
            retry=retry_if_exception(_is_retryable_error),
            wait=wait_exponential(multiplier=backoff_factor, min=1, max=10),
            stop=stop_after_attempt(max_retries),
            after=_log_retry,
            reraise=True
        )(func))

    return decorator
