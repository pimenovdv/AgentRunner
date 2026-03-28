import pytest
from app.models.manifest import AgentManifest, ExecutionLimits, Prompts
from app.models.graph import MicroGraph, Node, NodeType, Edge, EdgeCondition
from app.models.state import State
from app.services.graph_builder import GraphBuilder

@pytest.fixture
def sample_manifest():
    return AgentManifest(
        input_schema={},
        output_schema={},
        prompts=Prompts(system_instructions="Test prompt"),
        graph=MicroGraph(
            nodes=[
                Node(id="reasoning_node", type=NodeType.REASONING),
                Node(id="tool_node", type=NodeType.TOOL_EXECUTION, tool_name="calc"),
                Node(id="format_node", type=NodeType.DATA_TRANSFORMATION),
            ],
            edges=[
                Edge(source="reasoning_node", target="tool_node", condition=EdgeCondition(condition_expression="input_context.get('need_tool') == True")),
                Edge(source="reasoning_node", target="format_node", condition=EdgeCondition(condition_expression="input_context.get('need_tool') == False")),
                Edge(source="tool_node", target="format_node"),
            ]
        )
    )

@pytest.mark.asyncio
async def test_graph_builder_routing_tool(sample_manifest):
    builder = GraphBuilder(sample_manifest)
    app = builder.build()

    state = State(input_context={"need_tool": True})

    # Trace nodes executed by looking at _last_node_executed since our stub updates it
    res = await app.ainvoke(state)

    # Flow: START -> reasoning_node -> tool_node -> format_node -> END
    assert res["input_context"]["_last_node_executed"] == "format_node"

@pytest.mark.asyncio
async def test_graph_builder_routing_format(sample_manifest):
    builder = GraphBuilder(sample_manifest)
    app = builder.build()

    state = State(input_context={"need_tool": False})

    # Flow: START -> reasoning_node -> format_node -> END
    res = await app.ainvoke(state)

    assert res["input_context"]["_last_node_executed"] == "format_node"

@pytest.mark.asyncio
async def test_graph_builder_empty_graph():
    manifest = AgentManifest(
        input_schema={},
        output_schema={},
        prompts=Prompts(system_instructions="Test empty"),
        graph=MicroGraph(nodes=[], edges=[])
    )
    builder = GraphBuilder(manifest)
    app = builder.build()

    # Should safely handle an empty graph
    state = State(input_context={})
    res = await app.ainvoke(state)
    assert "_last_node_executed" not in res["input_context"]

from unittest.mock import AsyncMock, patch, MagicMock
from app.models.state import MessageRole, Message, ToolCall
from app.models.tools import ToolType, BuiltinTool, BuiltinConfig
from app.models.llm import LlmConfig, LlmProvider
from langchain_core.messages import AIMessage

@pytest.fixture
def manifest_with_tools(sample_manifest):
    # Add a tool to the manifest
    tool = BuiltinTool(
        name="calc",
        type=ToolType.BUILTIN,
        builtin_config=BuiltinConfig(function_name="calculator")
    )
    sample_manifest.tools = [tool]
    return sample_manifest

@pytest.mark.asyncio
async def test_reasoning_node_execution(manifest_with_tools):
    # Setup LLM config
    llm_config = LlmConfig(provider=LlmProvider.OPENAI, model_name="gpt-4o")

    # We want to mock ChatOpenAI so we don't make real API calls
    with patch("app.services.graph_builder.ChatOpenAI") as MockChat:
        mock_instance = MockChat.return_value
        # Mock bind_tools
        mock_instance.bind_tools.return_value = mock_instance

        # Mock ainvoke to return an AIMessage with a tool call
        mock_response = AIMessage(
            content="I need to calculate this.",
            tool_calls=[{"id": "call_123", "name": "calc", "args": {"expression": "2+2"}}]
        )
        mock_instance.ainvoke = AsyncMock(return_value=mock_response)

        builder = GraphBuilder(manifest_with_tools, llm_config=llm_config)
        app = builder.build()

        # We invoke the graph with a state that forces reasoning -> tool_node -> format_node
        state = State(input_context={"need_tool": True}, messages=[Message(role=MessageRole.USER, content="What is 2+2?")])

        res = await app.ainvoke(state)

        # Check that the reasoning node appended the Assistant message with tool calls
        assert len(res["messages"]) >= 3 # USER + ASSISTANT (from reasoning) + TOOL (from tool node)

        assistant_msg = next((m for m in res["messages"] if m.role == MessageRole.ASSISTANT), None)
        assert assistant_msg is not None
        assert assistant_msg.content == "I need to calculate this."
        assert len(assistant_msg.tool_calls) == 1
        assert assistant_msg.tool_calls[0].name == "calc"

        # Check that the tool execution node executed and appended a tool message
        tool_msg = next((m for m in res["messages"] if m.role == MessageRole.TOOL), None)
        assert tool_msg is not None
        assert tool_msg.tool_call_id == "call_123"
        assert "4.0" in tool_msg.content  # 2+2 = 4.0 (calc output)

@pytest.mark.asyncio
async def test_tool_execution_node_direct(manifest_with_tools):
    builder = GraphBuilder(manifest_with_tools)
    app = builder.build()

    # Manually create state with an assistant message requiring a tool call
    assistant_msg = Message(
        role=MessageRole.ASSISTANT,
        content="Calculating...",
        tool_calls=[ToolCall(id="call_456", name="calc", arguments={"expression": "10*5"})]
    )

    # Send it directly to the tool_node (since reasoning node will skip if no llm_config and just pass through)
    # Actually wait, let's just use the builder to invoke the specific node func
    node = [n for n in manifest_with_tools.graph.nodes if n.id == "tool_node"][0]
    func = builder._create_node_func(node)

    state = State(messages=[assistant_msg], input_context={})

    res = await func(state)

    # Verify the result of the tool execution
    assert "messages" in res
    assert len(res["messages"]) == 1
    tool_msg = res["messages"][0]

    assert tool_msg.role == MessageRole.TOOL
    assert tool_msg.tool_call_id == "call_456"
    assert "50.0" in tool_msg.content

@pytest.mark.asyncio
async def test_graph_builder_loop_bounds():
    manifest = AgentManifest(
        input_schema={},
        output_schema={},
        prompts=Prompts(system_instructions="Test loop bounds"),
        execution_limits=ExecutionLimits(max_steps=5),
        graph=MicroGraph(
            nodes=[
                Node(id="node_a", type=NodeType.DATA_TRANSFORMATION),
                Node(id="node_b", type=NodeType.DATA_TRANSFORMATION)
            ],
            edges=[
                Edge(source="node_a", target="node_b"),
                Edge(source="node_b", target="node_a")
            ]
        )
    )

    builder = GraphBuilder(manifest)
    app = builder.build()

    state = State(input_context={}, step_count=0)

    # Invoke the graph, it should hit the loop bound (max_steps=5) and return to fallback node.
    # Without loop protection, this would hang indefinitely.
    res = await app.ainvoke(state)

    # It should have executed 5 steps, hit the router which returns __fallback_node__,
    # and then the fallback node should add the error message.
    assert res["step_count"] >= 5
    assert len(res["messages"]) == 1
    assert "exceeded maximum allowed steps" in res["messages"][0].content
