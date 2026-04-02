# AgentRunner Development Guidelines

## General Information
- The repository is dedicated to a multi-agent orchestration platform.
- The 'Runner' service (Execution Engine / Player) is developed using FastAPI and LangGraph.
- Graph execution is orchestrated within the `/execute` API endpoint in `app/api/routes/player.py`.
- It uses `GraphBuilder` to instantiate the LangGraph and invokes `graph.ainvoke` wrapped in `asyncio.wait_for` to strictly enforce configured timeouts (`timeout_ms`) at the orchestration process level.
- Pydantic domain models are structured within the `app/models/` directory, modularized by domain (e.g., `api.py`, `graph.py`, `manifest.py`, and polymorphic `tools.py`).
- Environment variables and application configuration are managed using `pydantic-settings`, and structured logging is handled by `structlog`.
- Network resilience (circuit breakers and retries) is implemented using the `tenacity` library. A centralized `@with_retry` decorator in `app/services/retry_utils.py` applies exponential backoff to class/static methods for LLM, REST API, and MCP service calls, specifically handling transient errors like HTTP 429 and 50x.
- The project uses `uv` for dependency management, package installation, and execution (e.g., `uv add`, `uv run`).
- Continuous Integration is implemented via GitHub Actions (`.github/workflows/ci.yml`) to automatically install dependencies and run tests using `uv` and `pytest`. Testing is performed using `pytest`, which is executed via `PYTHONPATH=. uv run pytest` to ensure the `app` module is correctly resolved.

## Security
- To prevent Arbitrary Code Execution (ACE) vulnerabilities, string-based conditional routing rules from JSON manifests MUST NOT be evaluated using `eval()`. Instead, they are parsed securely using an AST parser (like the custom evaluator in `app/services/graph_builder.py` with explicitly allowed operators).
- When parsing shell commands for external processes, such as MCP stdio transport commands, always use `shlex.split` instead of a simple `.split()` to properly handle string arguments containing spaces.

## LangGraph specifics
- The LangGraph `StateGraph` is dynamically constructed from JSON manifests using `GraphBuilder` in `app/services/graph_builder.py`, which maps manifest nodes to async executable Python functions and edges to routing logic.
- LangGraph agent state is managed using a Pydantic `BaseModel` (`State` in `app/models/state.py`). The `messages` list must be annotated with `typing.Annotated[List[Message], operator.add]` to instruct LangGraph's reducer to properly append rather than overwrite messages.
- Execution telemetry (such as invoked tools and LLM token usage safely extracted from LangChain's `AIMessage.response_metadata`) is continually accumulated within the LangGraph `State` utilizing `operator.add` reducers and exposed via the `ExecuteResponse` payload.

## Integrations and tools
- Built-in LangChain tools for local, secure operations (e.g., AST-based calculator, date formatter) are defined in `app/models/builtin_tools.py` using the `@tool` decorator.
- Declarative REST API calls are managed by `RestApiClient` in `app/services/rest_api.py`. It utilizes `httpx.AsyncClient` for networking, `jsonschema` for validating LLM-generated arguments, and `jsonpath-ng` to extract specific payload data to optimize token context.
- When implementing or modifying REST API template interpolation, ensure regex matching is constrained to alphanumeric/word characters (e.g., `r'\{([a-zA-Z0-9_]+)\}'`) to avoid accidentally greedily matching JSON object curly braces inside request bodies.
- The Model Context Protocol (MCP) client implementation uses the `mcp` Python SDK in `app/services/mcp.py`. Transports (`mcp.client.stdio.stdio_client` and `mcp.client.sse.sse_client`) must be entered as async context managers yielding streams, which are passed to `mcp.client.session.ClientSession`. Using `contextlib.AsyncExitStack` is an effective pattern to manage these nested async lifecycles.
- Dynamic tools retrieved from external sources (e.g., via MCP `tools/list`) must be converted into LangChain-compatible formats, such as `@tool` functions or `StructuredTool` objects, before they can be integrated into the LangGraph/LangChain agent logic.
- Kafka integration (`app/services/kafka.py`) uses `confluent_kafka.Producer` for messaging and `jsonschema` for validating LLM-generated payloads. Synchronous `produce` calls are offloaded to an executor thread pool via `asyncio.get_running_loop().run_in_executor` for asynchronous behavior. Unit tests mock the Producer to avoid cluster dependencies.

## Misc
- LLM self-correction (Reflection loop) is implemented in the `tool_executor` of `app/services/graph_builder.py` by catching Pydantic `ValidationError`s during tool argument parsing and returning the error text as a `MessageRole.TOOL` message to prompt the model to fix its JSON schema.
- String parameters with limited options should be defined using `enum.StrEnum` rather than `typing.Literal`.
