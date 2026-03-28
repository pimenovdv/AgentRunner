import structlog
import ast
import operator
from typing import Callable, Dict, Any, List, Optional
from pydantic import create_model

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage as LangchainToolMessage

from app.models.manifest import AgentManifest
from app.models.graph import Node, NodeType, Edge
from app.models.state import State, Message, MessageRole, ToolCall
from app.models.llm import LlmConfig, LlmProvider
from app.models.tools import Tool, ToolType
from app.models.builtin_tools import BUILTIN_TOOLS
from app.services.retry_utils import with_retry

# Dynamic import to avoid missing dependencies
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatOpenAI = None
    ChatAnthropic = None

logger = structlog.get_logger()

# Allowed operators for the safe AST evaluator
_allowed_ast_ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: operator.contains,
    ast.NotIn: lambda a, b: not operator.contains(a, b),
    ast.And: lambda a, b: a and b,
    ast.Or: lambda a, b: a or b,
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _safe_eval_condition(expression: str, context: dict[str, Any]) -> bool:
    """Safely evaluate a boolean condition expression against a dictionary context using AST parsing."""
    def _eval(node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            # Handle standard python literals mapped to names (mostly handled by Constant in newer Pythons)
            if node.id == 'True': return True
            if node.id == 'False': return False
            if node.id == 'None': return None
            # Allow fallback to gracefully return None for undefined variables
            return None

        elif isinstance(node, ast.Dict):
            return {_eval(k): _eval(v) for k, v in zip(node.keys, node.values) if k is not None}

        elif isinstance(node, ast.List):
            return [_eval(elt) for elt in node.elts]

        elif isinstance(node, ast.Tuple):
            return tuple(_eval(elt) for elt in node.elts)

        elif isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = _eval(right_node)
                op_type = type(op)
                if op_type not in _allowed_ast_ops:
                    raise ValueError(f"Unsupported comparison operator: {op_type}")
                # For 'In' and 'NotIn', the python operator module puts the collection on the right (b in a).
                if op_type in (ast.In, ast.NotIn):
                    res = _allowed_ast_ops[op_type](right, left)
                else:
                    res = _allowed_ast_ops[op_type](left, right)
                if not res:
                    return False
                left = right
            return True

        elif isinstance(node, ast.BoolOp):
            op_type = type(node.op)
            if op_type not in _allowed_ast_ops:
                raise ValueError(f"Unsupported boolean operator: {op_type}")

            if op_type == ast.And:
                for val_node in node.values:
                    if not _eval(val_node):
                        return False
                return True
            elif op_type == ast.Or:
                for val_node in node.values:
                    if _eval(val_node):
                        return True
                return False

        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in _allowed_ast_ops:
                raise ValueError(f"Unsupported unary operator: {op_type}")
            return _allowed_ast_ops[op_type](operand)

        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type not in _allowed_ast_ops:
                raise ValueError(f"Unsupported binary operator: {op_type}")
            return _allowed_ast_ops[op_type](left, right)

        elif isinstance(node, ast.Subscript):
            value = _eval(node.value)
            slice_val = _eval(node.slice)
            try:
                return value[slice_val]
            except (KeyError, IndexError, TypeError):
                return None # Graceful fallback for missing keys or indexes

        elif isinstance(node, ast.Call):
            # Whitelist safe dict methods like .get()
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                obj = _eval(node.func.value)
                if isinstance(obj, dict):
                    args = [_eval(arg) for arg in node.args]
                    return obj.get(*args)
            raise ValueError(f"Unsupported function call in expression: {ast.unparse(node)}")

        raise ValueError(f"Unsupported expression AST type: {type(node)}")

    parsed = ast.parse(expression, mode='eval')
    result = _eval(parsed.body)
    return bool(result)

def create_router(edges: List[Edge], max_steps: int) -> Callable[[State], str]:
    """Creates a routing function for conditional edges based on manifest conditions using a safe AST evaluator."""
    def router(state: State) -> str:
        if state.step_count >= max_steps:
            logger.warning("max_steps_reached_fallback_triggered", step_count=state.step_count, max_steps=max_steps)
            return "__fallback_node__"

        # We find fallback edges without a condition
        fallback_target = END
        for edge in edges:
            if edge.condition is None or not edge.condition.condition_expression:
                fallback_target = edge.target
                continue

            # Context for evaluation
            locals_dict = {
                "state": state,
                "input_context": state.input_context,
                "messages": state.messages
            }
            try:
                # Safely evaluate condition string expression
                # Example: "input_context.get('status') == 'ok'"
                res = _safe_eval_condition(edge.condition.condition_expression, locals_dict)
                if res:
                    return edge.target
            except Exception as ex:
                logger.error("error_evaluating_edge_condition", expression=edge.condition.condition_expression, error=str(ex))
                continue

        return fallback_target
    return router

def _convert_tool_to_langchain(tool_def: Tool) -> Any:
    """Converts our Tool manifest model to Langchain compatible tool (either @tool or StructuredTool)."""
    if tool_def.type == ToolType.BUILTIN:
        builtin_name = tool_def.builtin_config.function_name
        if builtin_name in BUILTIN_TOOLS:
            tool_copy = BUILTIN_TOOLS[builtin_name].copy() # type: ignore
            tool_copy.name = tool_def.name
            if tool_def.description:
                tool_copy.description = tool_def.description
            return tool_copy
        raise ValueError(f"Unknown builtin tool: {builtin_name}")

    elif tool_def.type == ToolType.REST_API:
        from langchain_core.tools import StructuredTool

        schema = tool_def.rest_api_config.parameters_schema
        fields = {}
        if "properties" in schema:
            for k, v in schema["properties"].items():
                fields[k] = (Any, ...) if k in schema.get("required", []) else (Any, None)

        ArgsSchema = create_model(f"{tool_def.name}Args", **fields)

        async def rest_api_func(**kwargs):
            # Dispatch to RestApiClient
            from app.services.rest_api import RestApiClient
            client = RestApiClient()
            try:
                # For step 4.3, we return a mock response or forward args
                # In full execution, this would parse URL, headers, and call HTTP
                return {"result": f"Executed REST API {tool_def.name} with {kwargs}"}
            finally:
                await client.close()

        return StructuredTool.from_function(
            coroutine=rest_api_func,
            name=tool_def.name,
            description=tool_def.description or f"REST API: {tool_def.name}",
            args_schema=ArgsSchema
        )

    elif tool_def.type == ToolType.KAFKA:
        from langchain_core.tools import StructuredTool

        schema = tool_def.kafka_config.message_schema or {}
        fields = {}
        if "properties" in schema:
            for k, v in schema["properties"].items():
                fields[k] = (Any, ...) if k in schema.get("required", []) else (Any, None)

        ArgsSchema = create_model(f"{tool_def.name}Args", **fields)

        async def kafka_func(**kwargs):
            # Dispatch to Kafka
            return {"result": f"Produced to Kafka {tool_def.name} with {kwargs}"}

        return StructuredTool.from_function(
            coroutine=kafka_func,
            name=tool_def.name,
            description=tool_def.description or f"Kafka Tool: {tool_def.name}",
            args_schema=ArgsSchema
        )

    elif tool_def.type == ToolType.MCP_SERVER:
        from langchain_core.tools import StructuredTool
        from typing import Any

        # We need a dynamic args schema, so we create an empty one that accepts any kwargs
        ArgsSchema = create_model(f"{tool_def.name}Args", __root__=(Any, None))

        async def mcp_func(**kwargs):
            return {"result": f"Executed MCP {tool_def.name} with {kwargs}"}

        return StructuredTool.from_function(
            coroutine=mcp_func,
            name=tool_def.name,
            description=tool_def.description or f"MCP Server: {tool_def.name}",
        )

    raise ValueError(f"Unsupported tool type: {tool_def.type}")

@with_retry(max_retries=3, backoff_factor=2.0)
async def _invoke_llm_with_retry(chat_model: Any, lc_messages: List[Any]) -> Any:
    return await chat_model.ainvoke(lc_messages)

class GraphBuilder:
    """Builds a LangGraph StateGraph dynamically from an AgentManifest."""
    def __init__(self, manifest: AgentManifest, llm_config: Optional[LlmConfig] = None):
        self.manifest = manifest
        self.llm_config = llm_config
        self.graph = StateGraph(State)
        self._tools_cache = None

    def _get_langchain_tools(self) -> List[Any]:
        if self._tools_cache is None:
            self._tools_cache = [_convert_tool_to_langchain(t) for t in self.manifest.tools]
        return self._tools_cache

    def _create_node_func(self, node: Node) -> Callable:
        """Translates a manifest Node into a python function suitable for LangGraph."""

        if node.type == NodeType.REASONING:
            async def reasoning_executor(state: State) -> Dict[str, Any]:
                logger.info("executing_reasoning_node", node_id=node.id)
                if not self.llm_config:
                    # Fallback to stub if no LLM configured (e.g., in some tests)
                    ctx = dict(state.input_context)
                    ctx["_last_node_executed"] = node.id
                    return {"input_context": ctx, "step_count": 1}

                # 1. Prepare chat model
                if self.llm_config.provider == LlmProvider.OPENAI:
                    if ChatOpenAI is None:
                        raise ImportError("langchain-openai not installed")
                    chat_model = ChatOpenAI(model=self.llm_config.model_name, temperature=self.llm_config.temperature)
                elif self.llm_config.provider == LlmProvider.ANTHROPIC:
                    if ChatAnthropic is None:
                        raise ImportError("langchain-anthropic not installed")
                    chat_model = ChatAnthropic(model=self.llm_config.model_name, temperature=self.llm_config.temperature)
                else:
                    raise ValueError(f"Unsupported LLM provider: {self.llm_config.provider}")

                # 2. Bind tools
                tools = self._get_langchain_tools()
                if tools:
                    chat_model = chat_model.bind_tools(tools)

                # 3. Format messages
                lc_messages = [SystemMessage(content=self.manifest.prompts.system_instructions)]
                for msg in state.messages:
                    if msg.role == MessageRole.USER:
                        lc_messages.append(HumanMessage(content=msg.content or ""))
                    elif msg.role == MessageRole.ASSISTANT:
                        tool_calls = []
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_calls.append({
                                    "id": tc.id,
                                    "name": tc.name,
                                    "args": tc.arguments,
                                })
                        lc_messages.append(AIMessage(content=msg.content or "", tool_calls=tool_calls))
                    elif msg.role == MessageRole.TOOL:
                        lc_messages.append(LangchainToolMessage(content=msg.content or "", tool_call_id=msg.tool_call_id or ""))

                # 4. Invoke model
                response = await _invoke_llm_with_retry(chat_model, lc_messages)

                # 5. Convert response to state Message
                tool_calls_out = []
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    for tc in response.tool_calls:
                        tool_calls_out.append(ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("args", {})))

                out_msg = Message(
                    role=MessageRole.ASSISTANT,
                    content=str(response.content) if response.content else None,
                    tool_calls=tool_calls_out if tool_calls_out else None
                )

                ctx = dict(state.input_context)
                ctx["_last_node_executed"] = node.id
                return {"messages": [out_msg], "input_context": ctx, "step_count": 1}

            reasoning_executor.__name__ = f"node_{node.id}"
            return reasoning_executor

        elif node.type == NodeType.TOOL_EXECUTION:
            async def tool_executor(state: State) -> Dict[str, Any]:
                logger.info("executing_tool_node", node_id=node.id, tool_name=node.tool_name)
                ctx = dict(state.input_context)
                ctx["_last_node_executed"] = node.id

                if not state.messages:
                    return {"input_context": ctx, "step_count": 1}

                # Find the last AIMessage with tool calls
                last_msg = None
                for msg in reversed(state.messages):
                    if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                        last_msg = msg
                        break

                if not last_msg:
                    logger.warning("no_tool_call_found_in_state", node_id=node.id)
                    return {"input_context": ctx, "step_count": 1}

                # Find the specific tool call matching this node's tool_name
                tool_call = None
                for tc in last_msg.tool_calls:
                    if tc.name == node.tool_name:
                        tool_call = tc
                        break

                if not tool_call:
                     logger.warning("tool_call_name_mismatch", node_id=node.id, expected=node.tool_name)
                     return {"input_context": ctx, "step_count": 1}

                # Execute the tool
                tools_dict = {t.name: t for t in self._get_langchain_tools()}
                if tool_call.name not in tools_dict:
                    out_msg = Message(role=MessageRole.TOOL, content=f"Error: Tool {tool_call.name} not found", tool_call_id=tool_call.id)
                    return {"messages": [out_msg], "input_context": ctx, "step_count": 1}

                tool = tools_dict[tool_call.name]
                try:
                    result = await tool.ainvoke(tool_call.arguments)
                except Exception as e:
                    logger.error("tool_execution_failed", error=str(e), tool_name=tool_call.name)
                    result = f"Error executing tool: {e}"

                out_msg = Message(
                    role=MessageRole.TOOL,
                    content=str(result),
                    tool_call_id=tool_call.id
                )

                return {"messages": [out_msg], "input_context": ctx, "step_count": 1}

            tool_executor.__name__ = f"node_{node.id}"
            return tool_executor

        else:
            # Fallback / Data Transformation stub
            async def node_executor(state: State) -> Dict[str, Any]:
                logger.info("executing_node", node_id=node.id, node_type=node.type)
                ctx = dict(state.input_context)
                ctx["_last_node_executed"] = node.id
                return {"input_context": ctx, "step_count": 1}

            node_executor.__name__ = f"node_{node.id}"
            return node_executor

    def build(self) -> StateGraph:
        """Constructs the LangGraph from the manifest's nodes and edges."""
        if not self.manifest.graph.nodes:
            logger.warning("empty_graph_nodes")
            # LangGraph requires at least one start point
            async def empty_node(state: State) -> Dict[str, Any]:
                return {"step_count": 1}
            empty_node.__name__ = "empty_node"
            self.graph.add_node("empty_node", empty_node)
            self.graph.add_edge(START, "empty_node")
            self.graph.add_edge("empty_node", END)
            return self.graph.compile()

        # 1. Register all nodes
        for node in self.manifest.graph.nodes:
            self.graph.add_node(node.id, self._create_node_func(node))

        # 2. Add edges to determine graph structure
        targets = {edge.target for edge in self.manifest.graph.edges}
        start_nodes = [node.id for node in self.manifest.graph.nodes if node.id not in targets]
        if not start_nodes and self.manifest.graph.nodes:
            start_nodes = [self.manifest.graph.nodes[0].id]

        # Connect START to initial node(s)
        for start_node in start_nodes:
            self.graph.add_edge(START, start_node)

        # Add fallback node
        async def fallback_node(state: State) -> Dict[str, Any]:
            logger.error("fallback_node_executed", reason="max_steps_reached")
            out_msg = Message(
                role=MessageRole.ASSISTANT,
                content="Error: Execution exceeded maximum allowed steps (loop detected)."
            )
            return {"messages": [out_msg]}
        fallback_node.__name__ = "fallback_node"
        self.graph.add_node("__fallback_node__", fallback_node)
        self.graph.add_edge("__fallback_node__", END)

        # Group outgoing edges by source
        source_edges: Dict[str, List[Edge]] = {}
        for edge in self.manifest.graph.edges:
            source_edges.setdefault(edge.source, []).append(edge)

        max_steps = self.manifest.execution_limits.max_steps

        # Add edges and routers
        for source, edges in source_edges.items():
            # Conditional routing (Skip Logic) to handle max steps bounds ALWAYS
            router = create_router(edges, max_steps=max_steps)

            # Path map explicitly maps all possible return values of the router function to target nodes
            path_map = {e.target: e.target for e in edges}
            path_map[END] = END
            path_map["__fallback_node__"] = "__fallback_node__"

            self.graph.add_conditional_edges(source, router, path_map)

        # Connect leaf nodes (nodes with no outgoing edges) to END
        sources = {edge.source for edge in self.manifest.graph.edges}
        end_nodes = [node.id for node in self.manifest.graph.nodes if node.id not in sources]

        for end_node in end_nodes:
            self.graph.add_edge(end_node, END)

        return self.graph.compile()
