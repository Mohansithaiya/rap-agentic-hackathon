import json

from fastapi.testclient import TestClient

from app.agent import agent as agent_module
from app.agent.agent import InventoryAgent, RequestInterpretation
from app.agent.harness import Harness
from app.api.main import app


def test_llm_interprets_request_then_tools_run_through_harness(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeCompletions:
        def create(self, **kwargs):
            assert "tools" not in kwargs
            assert "Check inventory for Widget" in kwargs["messages"][1]["content"]

            class Message:
                content = json.dumps({"product": "Widget", "operation": "reorder_status"})

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs == {"api_key": "test-key"}
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAI)
    harness = Harness()
    response = InventoryAgent(harness).run(
        "Check inventory for Widget and tell me whether we need to reorder it."
    )

    assert response.product == "widget"
    assert response.reorder_needed is False  # 24 > reorder level 10
    assert [event.tool_name for event in harness.events] == [
        "get_inventory",
        "get_product_details",
    ]
    assert all(event.success for event in harness.events)


def test_missing_api_key_uses_deterministic_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    harness = Harness()
    response = InventoryAgent(harness).run("Do we need to reorder sprocket?")

    assert response.product == "sprocket"
    assert response.reorder_needed is True  # 3 <= reorder level 8
    assert len(harness.events) == 2


def test_llm_failure_uses_deterministic_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FailedOpenAI:
        def __init__(self, **kwargs):
            raise RuntimeError("unavailable")

    monkeypatch.setattr(agent_module, "OpenAI", FailedOpenAI)
    response = InventoryAgent().run("Do we need to reorder gadget?")
    assert response.product == "gadget"
    assert response.quantity == 7


def test_inventory_tool_failure_is_traced(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fail_inventory(product):
        raise RuntimeError("inventory down")

    monkeypatch.setattr(agent_module.tools, "get_inventory", fail_inventory)
    harness = Harness()
    response = InventoryAgent(harness).run("reorder widget")

    assert response.quantity is None
    assert len(harness.events) == 1
    assert harness.events[0].tool_name == "get_inventory"
    assert harness.events[0].success is False


def test_product_details_failure_preserves_inventory_answer_and_trace(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fail_details(product):
        raise RuntimeError("details down")

    monkeypatch.setattr(agent_module.tools, "get_product_details", fail_details)
    harness = Harness()
    response = InventoryAgent(harness).run("reorder sprocket")

    assert response.reorder_needed is True
    assert "Product details were unavailable" in response.explanation
    assert [event.tool_name for event in harness.events] == [
        "get_inventory",
        "get_product_details",
    ]
    assert [event.success for event in harness.events] == [True, False]


def test_health_and_ask_endpoints():
    from unittest.mock import patch

    with patch.dict("os.environ", {}, clear=True):
        client = TestClient(app)
        health_response = client.get("/health")
        ask_response = client.post("/ask", json={"message": "reorder widget"})

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "healthy"}
    assert ask_response.status_code == 200
    assert ask_response.json()["product"] == "widget"
    assert ask_response.json()["reorder_needed"] is False


def test_request_interpretation_schema():
    parsed = RequestInterpretation(product="widget", operation="inventory")
    assert parsed.product == "widget"
