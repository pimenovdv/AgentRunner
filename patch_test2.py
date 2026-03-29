import re

with open("tests/test_structured_output.py", "r") as f:
    content = f.read()

content = content.replace(
'''    class MockChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass
        def with_structured_output(self, schema):
            self.schema = schema
            return MockStructuredLLM()''',
'''    class MockChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass
        def with_structured_output(self, schema):
            self.schema = schema
            return MockStructuredLLM()
        async def ainvoke(self, messages):
            class Resp:
                content = "dummy"
                tool_calls = []
            return Resp()
        def bind_tools(self, tools):
            return self''')

with open("tests/test_structured_output.py", "w") as f:
    f.write(content)
