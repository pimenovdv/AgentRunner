import re

with open("app/models/state.py", "r") as f:
    content = f.read()

new_content = content.replace(
    'input_context: Dict[str, Any] = Field(default_factory=dict, description="Входной контекст данных для агента")',
    'input_context: Dict[str, Any] = Field(default_factory=dict, description="Входной контекст данных для агента")\n    step_count: Annotated[int, operator.add] = Field(default=0, description="Счетчик шагов (узлов) для защиты от зацикливания")'
)

with open("app/models/state.py", "w") as f:
    f.write(new_content)
