from app.models.builtin_tools import BUILTIN_TOOLS
import copy

t = BUILTIN_TOOLS["calculator"]
print(t.name)
t2 = t.copy(update={"name": "calc", "description": "calc desc"})
print(t2.name)
