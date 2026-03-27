import pytest
import httpx
from unittest.mock import AsyncMock, patch
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
import time

from app.services.retry_utils import with_retry

# Use a faster backoff for tests
@with_retry(max_retries=3, backoff_factor=0.1)
async def mock_network_call(mock_func):
    return await mock_func()

@pytest.mark.asyncio
async def test_retry_on_httpx_429():
    mock_func = AsyncMock()
    req = httpx.Request("GET", "http://test")
    resp = httpx.Response(429, request=req)
    err = httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)

    # Fail twice, succeed on third
    mock_func.side_effect = [err, err, "success"]

    start = time.time()
    result = await mock_network_call(mock_func)
    end = time.time()

    assert result == "success"
    assert mock_func.call_count == 3

@pytest.mark.asyncio
async def test_retry_on_httpx_500():
    mock_func = AsyncMock()
    req = httpx.Request("GET", "http://test")
    resp = httpx.Response(500, request=req)
    err = httpx.HTTPStatusError("500 Internal Error", request=req, response=resp)

    mock_func.side_effect = [err, "success"]

    result = await mock_network_call(mock_func)
    assert result == "success"
    assert mock_func.call_count == 2

@pytest.mark.asyncio
async def test_no_retry_on_httpx_400():
    mock_func = AsyncMock()
    req = httpx.Request("GET", "http://test")
    resp = httpx.Response(400, request=req)
    err = httpx.HTTPStatusError("400 Bad Request", request=req, response=resp)

    mock_func.side_effect = [err, "success"]

    with pytest.raises(httpx.HTTPStatusError):
        await mock_network_call(mock_func)

    assert mock_func.call_count == 1

@pytest.mark.asyncio
async def test_retry_on_mcp_error_500():
    mock_func = AsyncMock()
    error_data = ErrorData(code=500, message="Internal Server Error")
    err = McpError(error=error_data)

    mock_func.side_effect = [err, "success"]

    result = await mock_network_call(mock_func)

    assert result == "success"
    assert mock_func.call_count == 2

@pytest.mark.asyncio
async def test_retry_on_mcp_error_internal():
    mock_func = AsyncMock()
    error_data = ErrorData(code=-32603, message="Internal error")
    err = McpError(error=error_data)

    mock_func.side_effect = [err, "success"]

    result = await mock_network_call(mock_func)

    assert result == "success"
    assert mock_func.call_count == 2

@pytest.mark.asyncio
async def test_max_retries_exceeded():
    mock_func = AsyncMock()
    req = httpx.Request("GET", "http://test")
    resp = httpx.Response(429, request=req)
    err = httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)

    # Fail 4 times (max retries is 2 for this test decorator)
    mock_func.side_effect = [err, err, err, "success"]

    with pytest.raises(httpx.HTTPStatusError):
        await mock_network_call(mock_func)

    assert mock_func.call_count == 3
