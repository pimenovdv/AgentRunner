with open("app/services/graph_builder.py", "r") as f:
    content = f.read()

# Update create_router
content = content.replace(
    'def create_router(edges: List[Edge]) -> Callable[[State], str]:',
    'def create_router(edges: List[Edge], max_steps: int) -> Callable[[State], str]:'
)

# Add fallback condition
content = content.replace(
    'def router(state: State) -> str:\n        # We find fallback edges without a condition',
    'def router(state: State) -> str:\n        if state.step_count >= max_steps:\n            logger.warning("max_steps_reached_fallback_triggered", step_count=state.step_count, max_steps=max_steps)\n            return "__fallback_node__"\n\n        # We find fallback edges without a condition'
)

# Replace returns with step_count increment
content = content.replace(
    'return {"messages": [out_msg], "input_context": ctx}',
    'return {"messages": [out_msg], "input_context": ctx, "step_count": 1}'
)
content = content.replace(
    'return {"input_context": ctx}',
    'return {"input_context": ctx, "step_count": 1}'
)
content = content.replace(
    'return {}',
    'return {"step_count": 1}'
)

# Update build logic
build_mod = """        # 2. Add edges to determine graph structure
        targets = {edge.target for edge in self.manifest.graph.edges}
        start_nodes = [node.id for node in self.manifest.graph.nodes if node.id not in targets]

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

            self.graph.add_conditional_edges(source, router, path_map)"""

# Instead of blindly replacing build, let's use a regex to replace the specific block.
import re
pattern = re.compile(r'        # 2\. Add edges to determine graph structure.*?(?=        # Connect leaf nodes)', re.DOTALL)
content = pattern.sub(build_mod + '\n\n', content)

with open("app/services/graph_builder.py", "w") as f:
    f.write(content)
