import pytest
import operator
from typing import get_args, get_origin, Annotated

from app.models.state import State, Message, MessageRole, ToolCall

def test_message_role_enum():
    assert MessageRole.SYSTEM == "system"
    assert MessageRole.USER == "user"
    assert MessageRole.ASSISTANT == "assistant"
    assert MessageRole.TOOL == "tool"

def test_tool_call_model():
    tc = ToolCall(id="tc-123", name="my_tool", arguments={"key": "value"})
    assert tc.id == "tc-123"
    assert tc.name == "my_tool"
    assert tc.arguments == {"key": "value"}

def test_message_model():
    msg = Message(role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello"
    assert msg.tool_calls is None
    assert msg.tool_call_id is None

def test_state_model_instantiation():
    state = State(
        messages=[Message(role=MessageRole.SYSTEM, content="System prompt")],
        input_context={"foo": "bar"}
    )
    assert len(state.messages) == 1
    assert state.messages[0].role == MessageRole.SYSTEM
    assert state.input_context == {"foo": "bar"}

def test_state_model_default_values():
    state = State()
    assert state.messages == []
    assert state.input_context == {}

def test_state_model_messages_annotation():
    # Verify that the messages field has the correct annotation for LangGraph reducer
    messages_field = State.model_fields["messages"]

    # In Pydantic v2, .annotation can sometimes strip Annotated.
    # Use rebuild_annotation() to get the full type hint.
    annotation = messages_field.rebuild_annotation()

    assert get_origin(annotation) is Annotated
    args = get_args(annotation)
    assert len(args) == 2
    # args[0] is List[Message], args[1] is operator.add
    assert args[1] is operator.add
