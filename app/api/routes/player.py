
from fastapi import APIRouter, HTTPException
from jsonschema import validate, ValidationError
import structlog
import asyncio
import time

from app.models.api import ExecuteRequest, ExecuteResponse, ExecutionStatus
from app.models.state import State
from app.services.graph_builder import GraphBuilder
from app.config import settings
from app.models.llm import LlmConfig, LlmProvider
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

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
    # In a real app, llm_config might be passed or built from settings. We build a default one here.
    llm_config = None
    if settings.openai_api_key:
        llm_config = LlmConfig(provider=LlmProvider.OPENAI, model_name="gpt-4o-mini", temperature=0.0)

    builder = GraphBuilder(manifest=request.agent_manifest, llm_config=llm_config)
    graph = builder.build()

    initial_state = State(input_context=request.input_context)

    # Use explicit request timeout if it differs from default, otherwise fallback to manifest
    timeout_ms = request.execution_limits.timeout_ms
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

        # 6.4 Structured Output
        output_data = final_state.input_context

        if llm_config and llm_config.provider == LlmProvider.OPENAI and ChatOpenAI and request.agent_manifest.output_schema:
            try:
                chat_model = ChatOpenAI(model=llm_config.model_name, temperature=0.0)
                # Ensure the schema is compatible by checking if it's a dict
                schema = request.agent_manifest.output_schema
                if isinstance(schema, dict) and schema:
                    structured_llm = chat_model.with_structured_output(schema)
                    # Prepare context string from messages or just input_context
                    from langchain_core.messages import SystemMessage, HumanMessage
                    context_str = str(final_state.input_context)

                    sys_msg = SystemMessage(content="Extract the structured output according to the provided schema based on the given context.")
                    usr_msg = HumanMessage(content=f"Context:\n{context_str}")

                    output_data = await structured_llm.ainvoke([sys_msg, usr_msg])
                    # Ensure output_data is a dict (if it returns a Pydantic object, dump it)
                    if hasattr(output_data, 'model_dump'):
                        output_data = output_data.model_dump()
                    elif not isinstance(output_data, dict):
                        # If the output is not a dict or a model, wrap it
                        output_data = {"result": output_data}
            except Exception as e:
                logger.error("structured_output_extraction_failed", error=str(e), execution_id=request.execution_id)
                # Fallback to the raw context or just empty if we can't extract
                pass

        # If still not matching and no llm was available (e.g. tests), validate and possibly return just input context.
        # It's better to just leave output_data as the dict we got.

        execution_time_ms = int((time.time() - start_time) * 1000)

        logger.info("execution_completed", execution_id=request.execution_id, time_ms=execution_time_ms, steps=final_state.step_count)

        return ExecuteResponse(
            status=ExecutionStatus.SUCCESS,
            output_data=output_data,
            telemetry={
                "execution_time_ms": execution_time_ms,
                "step_count": final_state.step_count,
                "used_tokens": final_state.used_tokens,
                "called_tools": final_state.called_tools,
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
