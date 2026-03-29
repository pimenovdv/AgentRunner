import re

with open("tests/test_structured_output.py", "r") as f:
    content = f.read()

# We need to mock ChatOpenAI both in app.api.routes.player and app.services.graph_builder if they use it. Wait, app.services.graph_builder uses ChatOpenAI from langchain_openai.
# The error says "The api_key client option must be set". This means the real ChatOpenAI is still being called inside GraphBuilder because we only patched it in app.api.routes.player.
content = content.replace('monkeypatch.setattr("app.api.routes.player.ChatOpenAI", MockChatOpenAI)',
'''monkeypatch.setattr("app.api.routes.player.ChatOpenAI", MockChatOpenAI)
    monkeypatch.setattr("app.services.graph_builder.ChatOpenAI", MockChatOpenAI)
    import os
    monkeypatch.setenv("OPENAI_API_KEY", "mock_key")''')

with open("tests/test_structured_output.py", "w") as f:
    f.write(content)
