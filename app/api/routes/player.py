from fastapi import APIRouter, HTTPException
from jsonschema import validate, ValidationError
import structlog
import asyncio
import time

from app.models.api import ExecuteRequest, ExecuteResponse, ExecutionStatus
from app.models.state import State
from app.services.graph_builder import GraphBuilder

logger = structlog.get_logger()
router = APIRouter()

@router.post("/execute", response_model=ExecuteResponse)
async def execute_player(request: ExecuteRequest) -> ExecuteResponse:
    """
    Эндпоинт для запуска графа агента с валидацией входных данных.
    """
    logger.info("execute_player_called", execution_id=request.execution_id)

    # 6.2 Входная валидация
    try:
        validate(instance=request.input_context, schema=request.agent_manifest.input_schema)
        logger.debug("input_validation_success", execution_id=request.execution_id)
    except ValidationError as e:
        logger.warning("input_validation_failed", execution_id=request.execution_id, error=str(e))
        raise HTTPException(status_code=400, detail=f"Input validation failed: {str(e)}")

    # 6.3 Запуск графа
    builder = GraphBuilder(manifest=request.agent_manifest)
    graph = builder.build()

    initial_state = State(input_context=request.input_context)

    # Use explicit request timeout if it differs from default, otherwise fallback to manifest
    timeout_ms = request.execution_limits.timeout_ms
    # default in pydantic model is 60000.
    # To properly handle defaults, we assume if request.execution_limits.timeout_ms == 60000
    # but manifest has a custom one, we prefer manifest (unless explicitly 60000 was set, but we can't tell easily).
    if timeout_ms == 60000 and request.agent_manifest.execution_limits.timeout_ms != 60000:
        timeout_ms = request.agent_manifest.execution_limits.timeout_ms

    timeout_sec = timeout_ms / 1000.0

    start_time = time.time()
    try:
        final_state_dict = await asyncio.wait_for(
            graph.ainvoke(initial_state.model_dump()),
            timeout=timeout_sec
        )
        final_state = State(**final_state_dict)

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Temporary output data until 6.4 Structured Output is implemented
        output_data = final_state.input_context

        logger.info("execution_completed", execution_id=request.execution_id, time_ms=execution_time_ms, steps=final_state.step_count)

        return ExecuteResponse(
            status=ExecutionStatus.SUCCESS,
            output_data=output_data,
            telemetry={
                "execution_time_ms": execution_time_ms,
                "step_count": final_state.step_count,
            }
        )

    except asyncio.TimeoutError:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.error("execution_timeout", execution_id=request.execution_id, timeout_ms=timeout_ms)
        return ExecuteResponse(
            status=ExecutionStatus.TIMEOUT,
            output_data={},
            telemetry={
                "execution_time_ms": execution_time_ms,
                "error": f"Execution exceeded {timeout_ms}ms"
            }
        )
    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.error("execution_error", execution_id=request.execution_id, error=str(e))
        return ExecuteResponse(
            status=ExecutionStatus.ERROR,
            output_data={},
            telemetry={
                "execution_time_ms": execution_time_ms,
                "error": str(e)
            }
        )
