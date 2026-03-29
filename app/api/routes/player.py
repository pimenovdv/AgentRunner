from fastapi import APIRouter, HTTPException
from jsonschema import validate, ValidationError
import structlog

from app.models.api import ExecuteRequest, ExecuteResponse, ExecutionStatus

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

    # Возвращаем заглушку пока (6.3-6.5 не реализованы)
    return ExecuteResponse(
        status=ExecutionStatus.SUCCESS,
        output_data={},
        telemetry={"message": "Stub response, execution graph not yet invoked"}
    )
