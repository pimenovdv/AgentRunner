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
