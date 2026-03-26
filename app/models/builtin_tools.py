import ast
import operator
import math
import re
import hashlib
from datetime import datetime
from typing import List, Union

from langchain_core.tools import tool

_allowed_ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

def _safe_eval(node: ast.expr):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_type = type(node.op)
        if op_type not in _allowed_ops:
            raise ValueError(f"Unsupported binary operator: {op_type}")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("Division by zero")
        return _allowed_ops[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_type = type(node.op)
        if op_type not in _allowed_ops:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        return _allowed_ops[op_type](operand)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == 'abs':
                return abs(_safe_eval(node.args[0]))
            elif func_name == 'round':
                if len(node.args) == 1:
                    return round(_safe_eval(node.args[0]))
                elif len(node.args) == 2:
                    return round(_safe_eval(node.args[0]), _safe_eval(node.args[1]))
            elif hasattr(math, func_name):
                func = getattr(math, func_name)
                args = [_safe_eval(arg) for arg in node.args]
                return func(*args)
            raise ValueError(f"Unsupported function call: {func_name}")
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'math':
            attr = node.func.attr
            if hasattr(math, attr):
                func = getattr(math, attr)
                args = [_safe_eval(arg) for arg in node.args]
                return func(*args)
            raise ValueError(f"Unsupported function call: math.{attr}")
        raise ValueError("Unsupported function call")
    elif isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == 'math':
            attr = node.attr
            if hasattr(math, attr):
                return getattr(math, attr)
            raise ValueError(f"Unsupported attribute: math.{attr}")
        raise ValueError("Unsupported attribute access")
    else:
        raise ValueError(f"Unsupported expression type: {type(node)}")

@tool
def calculator(expression: str) -> float | str:
    """
    Safely evaluate a mathematical expression.
    Supported operations: +, -, *, /, //, %, **, abs, round, and math functions (e.g., math.sin, math.pi).
    Example: calculator("2 + 2 * 3") -> 8.0
    """
    try:
        node = ast.parse(expression, mode='eval').body
        return float(_safe_eval(node))
    except Exception as e:
        return f"Error: {e}"

@tool
def datetime_formatter(date_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime string or get current datetime if date_str is 'now'.
    Examples:
    - datetime_formatter("now") -> "2023-10-27 15:30:00"
    - datetime_formatter("now", "%Y-%m-%d") -> "2023-10-27"
    """
    try:
        if date_str.lower() == "now":
            dt = datetime.now()
        else:
            try:
                dt = datetime.fromisoformat(date_str)
            except ValueError:
                return f"Error: Cannot parse datetime string '{date_str}'"
        return dt.strftime(format_str)
    except Exception as e:
        return f"Error: {e}"

@tool
def crypto_hasher(data: str, algorithm: str = "sha256") -> str:
    """
    Generate a cryptographic hash for the given data.
    Supported algorithms: md5, sha1, sha256, sha512.
    Example: crypto_hasher("hello", "sha256")
    """
    algo = algorithm.lower()
    if algo not in hashlib.algorithms_available:
         return f"Error: Unsupported algorithm '{algorithm}'"
    try:
        h = hashlib.new(algo)
        h.update(data.encode('utf-8'))
        return h.hexdigest()
    except Exception as e:
        return f"Error: {e}"

@tool
def regex_parser(pattern: str, text: str, group: int = 0) -> Union[List[str], str]:
    """
    Extract data from text using a regular expression.
    Returns all matches. If group > 0, returns that specific group for all matches.
    Example: regex_parser(r"\\d+", "hello 123 world 456") -> ["123", "456"]
    """
    try:
        matches = list(re.finditer(pattern, text))
        if not matches:
             return []
        if group == 0:
            return [m.group(0) for m in matches]
        else:
            return [m.group(group) for m in matches if group <= m.lastindex]
    except re.error as e:
        return f"Regex Error: {e}"
    except Exception as e:
        return f"Error: {e}"

BUILTIN_TOOLS = {
    "calculator": calculator,
    "datetime_formatter": datetime_formatter,
    "crypto_hasher": crypto_hasher,
    "regex_parser": regex_parser,
}
