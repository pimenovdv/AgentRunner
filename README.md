# AgentRunner: Execution Engine for Multi-Agent Platform

AgentRunner is an execution engine (Player) designed for a multi-agent orchestration platform. It is developed using **FastAPI** and **LangGraph**, providing a scalable and reliable environment for executing complex multi-agent workflows defined via declarative JSON manifests.

## Features

*   **Dynamic Graph Execution:** Dynamically constructs and executes a LangGraph `StateGraph` from JSON manifests, mapping nodes to async Python functions and edges to secure conditional routing logic.
*   **Secure Routing:** Uses an AST parser to safely evaluate string-based conditional routing rules from JSON manifests, avoiding arbitrary code execution (ACE) vulnerabilities associated with `eval()`.
*   **Timeouts at Orchestration Level:** Strictly enforces execution timeouts using `asyncio.wait_for` at the API orchestration level.
*   **Extensive Tool Support:**
    *   **Built-in Tools:** Secure local operations like an AST-based calculator and date formatting.
    *   **REST API Integration:** Declarative calls using `httpx.AsyncClient` with `jsonschema` validation for LLM arguments, and `jsonpath-ng` to extract payload data, saving context tokens. Template interpolation strictly uses `r'\{([a-zA-Z0-9_]+)\}'` to avoid matching JSON payload braces.
    *   **MCP (Model Context Protocol):** Integrates tools dynamically retrieved from an MCP server via SSE or stdio transports using `contextlib.AsyncExitStack` for async lifecycle management. Converts external tools into LangChain-compatible objects.
    *   **Kafka Messaging:** Asynchronous message production using `confluent_kafka.Producer` offloaded to a thread pool, with strict payload validation via JSON Schema.
*   **Resilience & Retries:** Employs the `tenacity` library for network resilience. A centralized `@with_retry` decorator automatically applies exponential backoff for LLM, REST, and MCP transient errors (e.g., HTTP 429, 50x).
*   **LLM Self-Correction (Reflection Loop):** Catches Pydantic `ValidationError`s during tool execution and prompts the LLM to fix its JSON schema output dynamically.
*   **Strict State Management:** Uses Pydantic `BaseModel` for LangGraph state with properly annotated reducers (e.g., `operator.add` for message appends). Continually accumulates execution telemetry (tools invoked, token usage).

## Architecture & Code Structure

The project code is primarily located in the `app/` directory:

*   **`app/api/`**: FastAPI routes. The entry point for execution is `POST /api/v1/player/execute` in `routes/player.py`.
*   **`app/models/`**: Pydantic domain models, modularized (e.g., `api.py` for request/response contracts, `graph.py` for state, `manifest.py` for manifest definitions, `tools.py` for polymorphic tool schemas).
*   **`app/services/`**: Core logic including `graph_builder.py` (LangGraph instantiation), `rest_api.py` (API tool executor), `mcp.py` (MCP client), `kafka.py` (Kafka producer), and `retry_utils.py`.
*   **Configuration:** Managed by `pydantic-settings` via `app/config.py`.
*   **Logging:** Structured JSON logging configured with `structlog`.

## Usage & Development

### Prerequisites

The project relies on `uv` for dependency management.

### Installation

```bash
# Install dependencies using uv
uv sync
```

### Running Locally

```bash
# Start the FastAPI server using uv
uv run app/main.py
```

The application is containerized utilizing a lightweight `python:3.12-slim` image. Infrastructure such as Redis, Postgres, and Kafka can be managed via `docker-compose.yml`.

### Running Tests

Testing is implemented with `pytest` and integrated into CI pipelines via GitHub Actions.

```bash
# Run tests ensuring correct Python path
PYTHONPATH=. uv run pytest
```

## Contributing Guidelines

Please see `AGENTS.md` for specific development instructions, architectural constraints, and security principles required when contributing to this project.
