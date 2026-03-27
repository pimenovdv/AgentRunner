import structlog
from typing import Dict, Any, List, Callable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage as LangchainToolMessage
from app.models.state import State, Message, MessageRole, ToolCall
from app.models.graph import Node, NodeType
from app.models.manifest import AgentManifest
from app.models.llm import LlmConfig, LlmProvider
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

logger = structlog.get_logger()

def _create_chat_model(config: LlmConfig):
    if config.provider == LlmProvider.OPENAI:
        return ChatOpenAI(model=config.model_name, temperature=config.temperature)
    elif config.provider == LlmProvider.ANTHROPIC:
        return ChatAnthropic(model=config.model_name, temperature=config.temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

def convert_state_messages_to_langchain(messages: List[Message]) -> List[Any]:
    lc_messages = []
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            lc_messages.append(SystemMessage(content=msg.content or ""))
        elif msg.role == MessageRole.USER:
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
            # content can be None but langchain expects a string or None, though some models prefer empty string
            lc_messages.append(AIMessage(content=msg.content or "", tool_calls=tool_calls))
        elif msg.role == MessageRole.TOOL:
            lc_messages.append(LangchainToolMessage(content=msg.content or "", tool_call_id=msg.tool_call_id or ""))
    return lc_messages

def convert_langchain_to_state_message(lc_msg: Any) -> Message:
    if isinstance(lc_msg, AIMessage):
        tool_calls = []
        if lc_msg.tool_calls:
            for tc in lc_msg.tool_calls:
                tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=tc["args"]))
        return Message(
            role=MessageRole.ASSISTANT,
            content=str(lc_msg.content) if lc_msg.content else None,
            tool_calls=tool_calls if tool_calls else None
        )
    elif isinstance(lc_msg, LangchainToolMessage):
        return Message(
            role=MessageRole.TOOL,
            content=str(lc_msg.content),
            tool_call_id=lc_msg.tool_call_id
        )
    raise ValueError(f"Cannot convert {type(lc_msg)} to State Message")

def create_reasoning_node_executor(node: Node, manifest: AgentManifest, llm_config: LlmConfig, get_bound_tools: Callable) -> Callable:
    async def node_executor(state: State) -> Dict[str, Any]:
        logger.info("executing_reasoning_node", node_id=node.id)

        # 1. Compile System Prompt with context
        sys_msg = Message(role=MessageRole.SYSTEM, content=manifest.prompts.system_instructions)
        # Note: We should probably prepend this to the state.messages, but let's do it dynamically here

        # 2. Get the LLM model
        chat_model = _create_chat_model(llm_config)

        # 3. Bind tools to the model
        tools = get_bound_tools()
        if tools:
            chat_model = chat_model.bind_tools(tools)

        # 4. Prepare messages for Langchain
        lc_messages = [convert_state_messages_to_langchain([sys_msg])[0]] + convert_state_messages_to_langchain(state.messages)

        # 5. Invoke LLM
        response = await chat_model.ainvoke(lc_messages)

        # 6. Convert response and append to state
        out_msg = convert_langchain_to_state_message(response)

        # Also track last node executed for tests
        ctx = dict(state.input_context)
        ctx["_last_node_executed"] = node.id

        return {"messages": [out_msg], "input_context": ctx}

    node_executor.__name__ = f"node_{node.id}"
    return node_executor

def create_tool_execution_node_executor(node: Node, manifest: AgentManifest, get_bound_tools: Callable) -> Callable:
    async def node_executor(state: State) -> Dict[str, Any]:
        logger.info("executing_tool_node", node_id=node.id, tool_name=node.tool_name)

        if not state.messages:
            raise ValueError("No messages in state to execute tool from")

        # Find the last AIMessage with tool calls
        last_msg = None
        for msg in reversed(state.messages):
            if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                last_msg = msg
                break

        if not last_msg:
            # We must be here by mistake if there's no tool call, or we just log and skip
            logger.warning("no_tool_call_found_in_state", node_id=node.id)
            return {"input_context": {"_last_node_executed": node.id}}

        # Find the specific tool call matching this node's tool_name
        tool_call = None
        for tc in last_msg.tool_calls:
            if tc.name == node.tool_name:
                tool_call = tc
                break

        if not tool_call:
             logger.warning("tool_call_name_mismatch", node_id=node.id, expected=node.tool_name)
             return {"input_context": {"_last_node_executed": node.id}}

        # Execute the tool
        tools = {t.name: t for t in get_bound_tools()}
        if tool_call.name not in tools:
            raise ValueError(f"Tool {tool_call.name} not found in bound tools")

        tool = tools[tool_call.name]
        try:
            # Most Langchain tools can be invoked directly
            result = await tool.ainvoke(tool_call.arguments)
        except Exception as e:
            logger.error("tool_execution_failed", error=str(e), tool_name=tool_call.name)
            result = f"Error executing tool: {e}"

        # Append Tool message
        out_msg = Message(
            role=MessageRole.TOOL,
            content=str(result),
            tool_call_id=tool_call.id
        )

        ctx = dict(state.input_context)
        ctx["_last_node_executed"] = node.id

        return {"messages": [out_msg], "input_context": ctx}

    node_executor.__name__ = f"node_{node.id}"
    return node_executor

def create_data_transformation_node_executor(node: Node) -> Callable:
    async def node_executor(state: State) -> Dict[str, Any]:
        logger.info("executing_data_transformation_node", node_id=node.id)
        ctx = dict(state.input_context)
        ctx["_last_node_executed"] = node.id
        return {"input_context": ctx}

    node_executor.__name__ = f"node_{node.id}"
    return node_executor
