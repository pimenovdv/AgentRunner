with open("app/services/graph_builder.py", "r") as f:
    content = f.read()

# If there's a loop with all nodes being targets (e.g., node_a -> node_b -> node_a),
# start_nodes will be empty. We need to ensure there is ALWAYS at least one start node.
# We can default to the first node in the list if start_nodes is empty.
new_code = """        start_nodes = [node.id for node in self.manifest.graph.nodes if node.id not in targets]
        if not start_nodes and self.manifest.graph.nodes:
            start_nodes = [self.manifest.graph.nodes[0].id]"""

content = content.replace(
    '        start_nodes = [node.id for node in self.manifest.graph.nodes if node.id not in targets]',
    new_code
)

with open("app/services/graph_builder.py", "w") as f:
    f.write(content)
