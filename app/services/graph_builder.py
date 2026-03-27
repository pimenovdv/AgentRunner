import structlog
import ast
import operator
from typing import Callable, Dict, Any, List

from langgraph.graph import StateGraph, START, END

from app.models.manifest import AgentManifest
from app.models.graph import Node, NodeType, Edge
from app.models.state import State

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

def create_router(edges: List[Edge]) -> Callable[[State], str]:
    """Creates a routing function for conditional edges based on manifest conditions using a safe AST evaluator."""
    def router(state: State) -> str:
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

class GraphBuilder:
    """Builds a LangGraph StateGraph dynamically from an AgentManifest."""
    def __init__(self, manifest: AgentManifest):
        self.manifest = manifest
        self.graph = StateGraph(State)

    def _create_node_func(self, node: Node) -> Callable:
        """Translates a manifest Node into a python function suitable for LangGraph."""
        # For now (step 4.2), we implement a stub that logs execution and updates state.
        # The full-fledged LLM and Tools execution logic will be done in step 4.3.
        async def node_executor(state: State) -> Dict[str, Any]:
            logger.info("executing_node", node_id=node.id, node_type=node.type)

            # Temporary stub: track node execution path in state for testing
            ctx = dict(state.input_context)
            ctx["_last_node_executed"] = node.id
            return {"input_context": ctx}

        node_executor.__name__ = f"node_{node.id}"
        return node_executor

    def build(self) -> StateGraph:
        """Constructs the LangGraph from the manifest's nodes and edges."""
        if not self.manifest.graph.nodes:
            logger.warning("empty_graph_nodes")
            # LangGraph requires at least one start point
            async def empty_node(state: State) -> Dict[str, Any]:
                return {}
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

        # Connect START to initial node(s)
        for start_node in start_nodes:
            self.graph.add_edge(START, start_node)

        # Group outgoing edges by source
        source_edges: Dict[str, List[Edge]] = {}
        for edge in self.manifest.graph.edges:
            source_edges.setdefault(edge.source, []).append(edge)

        # Add edges and routers
        for source, edges in source_edges.items():
            if len(edges) == 1 and (edges[0].condition is None or not edges[0].condition.condition_expression):
                # Simple sequential edge
                self.graph.add_edge(source, edges[0].target)
            else:
                # Conditional routing (Skip Logic)
                router = create_router(edges)

                # Path map explicitly maps all possible return values of the router function to target nodes
                path_map = {e.target: e.target for e in edges}
                path_map[END] = END

                self.graph.add_conditional_edges(source, router, path_map)

        # Connect leaf nodes (nodes with no outgoing edges) to END
        sources = {edge.source for edge in self.manifest.graph.edges}
        end_nodes = [node.id for node in self.manifest.graph.nodes if node.id not in sources]

        for end_node in end_nodes:
            self.graph.add_edge(end_node, END)

        return self.graph.compile()
