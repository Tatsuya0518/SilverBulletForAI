"""Tests for A2A protocol compliance."""

from sbfa.a2a.agent_card import AgentCard, AgentCardSignature, Capabilities
from sbfa.a2a.protocol import JsonRpcRequest, JsonRpcResponse, Message, Part, Role, Task, TaskState
from sbfa.agents.base import Skill


def test_agent_card_camel_case_serialization():
    """A2A v0.3: All JSON fields MUST use camelCase."""
    card = AgentCard(
        name="test-agent",
        description="A test agent",
        url="http://localhost:8080",
        provider="mock",
        model="mock-v1",
        skills=[Skill(name="code", description="Coding", tags=["code"])],
        capabilities=Capabilities(streaming=True, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
    )

    json_data = card.to_json()

    assert json_data["name"] == "test-agent"
    assert json_data["protocolVersion"] == "0.3.0"
    assert json_data["defaultInputModes"] == ["text"]
    assert json_data["defaultOutputModes"] == ["text"]

    # Capabilities must also be camelCase
    caps = json_data["capabilities"]
    assert "pushNotifications" in caps
    assert "stateTransitionHistory" in caps


def test_agent_card_db_record_snake_case():
    """SurrealDB stores in snake_case."""
    card = AgentCard(
        name="test",
        description="Test",
        url="http://localhost",
        provider="mock",
        model="mock-v1",
        skills=[],
        capabilities=Capabilities(),
    )

    db_record = card.to_db_record()
    assert "protocol_version" in db_record
    assert "default_input_modes" in db_record


def test_agent_card_well_known_path():
    """A2A v0.3: Discovery at /.well-known/agent-card.json (not agent.json)."""
    # This is a documentation test - the actual endpoint is in main.py
    # The path should be /.well-known/agent-card.json per v0.3 spec
    pass


def test_task_states():
    task = Task()
    assert task.state == TaskState.SUBMITTED
    assert task.id is not None
    assert task.context_id is not None


def test_message_camel_case():
    msg = Message(
        role=Role.USER,
        parts=[Part(type="text", text="Hello")],
        context_id="ctx-123",
    )
    data = msg.model_dump(by_alias=True)
    assert "contextId" in data


def test_json_rpc_request():
    req = JsonRpcRequest(method="tasks/send", params={"task": "test"})
    assert req.jsonrpc == "2.0"
    assert req.method == "tasks/send"


def test_json_rpc_response():
    resp = JsonRpcResponse(id="1", result={"status": "ok"})
    assert resp.error is None
    assert resp.result["status"] == "ok"
