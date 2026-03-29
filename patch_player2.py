import re

with open("app/api/routes/player.py", "r") as f:
    code = f.read()

# I want to ensure that `output_data` strictly falls back to dict validation.
# We also have an issue where test assumes we get input_context back, so we should ensure if LLM is mock or failed, we return final_state.input_context.
# But also we need `from langchain_core.messages import SystemMessage, HumanMessage` at the top.
# The previous patch added it dynamically inside the function, which is fine.
