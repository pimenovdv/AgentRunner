import re
with open("app/services/graph_builder.py", "r") as f:
    content = f.read()

# Fix tool name matching by overriding the name if necessary
old_block = """    if tool_def.type == ToolType.BUILTIN:
        builtin_name = tool_def.builtin_config.function_name
        if builtin_name in BUILTIN_TOOLS:
            return BUILTIN_TOOLS[builtin_name]
        raise ValueError(f"Unknown builtin tool: {builtin_name}")"""

new_block = """    if tool_def.type == ToolType.BUILTIN:
        builtin_name = tool_def.builtin_config.function_name
        if builtin_name in BUILTIN_TOOLS:
            tool_copy = BUILTIN_TOOLS[builtin_name].copy()
            tool_copy.name = tool_def.name
            if tool_def.description:
                tool_copy.description = tool_def.description
            return tool_copy
        raise ValueError(f"Unknown builtin tool: {builtin_name}")"""

content = content.replace(old_block, new_block)

with open("app/services/graph_builder.py", "w") as f:
    f.write(content)
