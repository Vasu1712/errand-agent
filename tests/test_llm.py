import pytest
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from errand.config import Settings
from errand.llm import (
    ChatMessage,
    ModelClient,
    ModelOutputError,
    ModelUnavailable,
    ScriptedModel,
    ScriptExhausted,
    TokenUsage,
)
from errand.llm.ollama import OllamaModel


class SlotPatch(BaseModel):
    service: str | None = None
    area: str | None = None


MESSAGES = [
    ChatMessage(role="system", content="Extract the details the user gave."),
    ChatMessage(role="user", content="plumber in HSR Layout"),
]


def test_scripted_model_returns_validated_replies_in_order():
    model = ScriptedModel([{"service": "plumber"}, SlotPatch(area="HSR Layout")])
    first = model.complete(MESSAGES, SlotPatch, purpose="slot_patch")
    second = model.complete(MESSAGES, SlotPatch, purpose="slot_patch")

    assert first.output == SlotPatch(service="plumber")
    assert second.output == SlotPatch(area="HSR Layout")
    assert first.usage.input_tokens > 0 and first.model == "scripted"
    assert model.remaining == 0
    assert [request.purpose for request in model.requests] == ["slot_patch", "slot_patch"]
    assert model.requests[0].messages == tuple(MESSAGES)


def test_scripted_steps_can_compute_a_reply_or_raise():
    model = ScriptedModel(
        [
            lambda request: {"service": request.messages[-1].content.split()[0]},
            ModelUnavailable("Ollama is down"),
        ]
    )
    assert model.complete(MESSAGES, SlotPatch, purpose="slot_patch").output.service == "plumber"
    with pytest.raises(ModelUnavailable):
        model.complete(MESSAGES, SlotPatch, purpose="slot_patch")


def test_scripted_model_fails_loudly_when_the_script_runs_out_or_misfits():
    with pytest.raises(ScriptExhausted):
        ScriptedModel().complete(MESSAGES, SlotPatch, purpose="proposal")
    with pytest.raises(ModelOutputError) as error:
        ScriptedModel([{"service": 42}]).complete(MESSAGES, SlotPatch, purpose="slot_patch")
    assert error.value.raw == '{"service": 42}'


def test_both_models_satisfy_the_protocol_and_the_live_one_builds_offline():
    live = OllamaModel.from_settings(Settings(_env_file=None))
    assert isinstance(live, ModelClient) and live.name == "qwen2.5:7b"
    assert isinstance(ScriptedModel(), ModelClient)


def test_live_model_sends_the_schema_as_format_and_validates_the_reply(monkeypatch):
    seen = {}

    def fake_invoke(self, messages, **kwargs):
        seen.update(kwargs, messages=messages)
        return AIMessage(
            content='{"service": "plumber", "area": "HSR Layout"}',
            usage_metadata={"input_tokens": 42, "output_tokens": 9, "total_tokens": 51},
        )

    monkeypatch.setattr(ChatOllama, "invoke", fake_invoke)
    reply = OllamaModel.from_settings(Settings(_env_file=None)).complete(
        MESSAGES, SlotPatch, purpose="slot_patch"
    )

    assert seen["format"] == SlotPatch.model_json_schema()
    assert [type(message).__name__ for message in seen["messages"]] == [
        "SystemMessage",
        "HumanMessage",
    ]
    assert reply.output == SlotPatch(service="plumber", area="HSR Layout")
    assert reply.usage == TokenUsage(input_tokens=42, output_tokens=9)


def test_live_model_maps_failures_to_model_errors(monkeypatch):
    model = OllamaModel.from_settings(Settings(_env_file=None))

    def server_down(self, messages, **kwargs):
        raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr(ChatOllama, "invoke", server_down)
    with pytest.raises(ModelUnavailable):
        model.complete(MESSAGES, SlotPatch, purpose="slot_patch")

    monkeypatch.setattr(
        ChatOllama, "invoke", lambda self, messages, **kwargs: AIMessage(content="nope")
    )
    with pytest.raises(ModelOutputError) as error:
        model.complete(MESSAGES, SlotPatch, purpose="slot_patch")
    assert error.value.raw == "nope"
