import re

with open("app/models/manifest.py", "r") as f:
    content = f.read()

new_content = content.replace(
    'timeout_ms: int = Field(default=60000, description="Таймаут выполнения в миллисекундах")',
    'timeout_ms: int = Field(default=60000, description="Таймаут выполнения в миллисекундах")\n    max_steps: int = Field(default=30, description="Максимальное количество шагов (узлов) для защиты от бесконечных циклов")'
)

with open("app/models/manifest.py", "w") as f:
    f.write(new_content)
