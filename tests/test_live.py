"""Checks against the real model runtimes. Deselected by default: run with ``pytest -m live``.

They need ``ollama serve`` with qwen2.5:7b pulled, and download the Vega checkpoint on first run.
"""

from datetime import date

import pytest

from errand.config import Settings
from errand.decide import VegaDecider
from errand.decide.questions import APPROVAL, INJECTION, INTENT, QUESTIONS, REVIEW_QUALITY
from errand.gateway.untrusted import untrusted_block
from errand.graph.slots import ground_patch
from errand.llm import ChatMessage
from errand.llm.ollama import OllamaModel
from errand.llm.prompts import slot_patch_system
from errand.llm.schemas import SlotPatch
from errand.mocks.catalog import INJECTION_TEXTS

pytestmark = pytest.mark.live
TODAY = date(2026, 11, 3)


@pytest.fixture(scope="module")
def vega():
    decider = VegaDecider.from_settings(QUESTIONS, Settings(_env_file=None))
    decider.load()
    return decider


def test_qwen_extracts_the_spec_example_and_grounding_keeps_only_what_was_said():
    model = OllamaModel.from_settings(Settings(_env_file=None))
    message = "Find a plumber for my kitchen tap this week, mornings only, under ₹1,500."
    try:
        reply = model.complete(
            [
                ChatMessage(role="system", content=slot_patch_system(TODAY)),
                ChatMessage(role="user", content=message),
            ],
            SlotPatch,
            purpose="live_check",
        )
    finally:
        model.close()
    accepted, _ = ground_patch(reply.output, message, {}, TODAY)
    assert accepted["service"] == "plumber"
    assert (accepted["date_from"], accepted["date_to"]) == ("2026-11-03", "2026-11-08")
    assert accepted["time_window"] == "morning" and accepted["budget_max_inr"] == 1500
    assert "area" not in accepted  # not stated, so not invented
    assert reply.usage.input_tokens > 0 and reply.usage.output_tokens > 0


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Find a plumber for my kitchen tap this week", "book"),
        ("Move my AC servicing to Saturday.", "reschedule"),
        ("cancel my cleaning appointment", "cancel"),
        ("what do you remember about me?", "memory_command"),
    ],
)
def test_vega_routes_the_spec_examples(vega, text, intent):
    decision = vega.decide(text, INTENT)
    assert decision.choice == intent and decision.is_confident(0.85)


def test_vega_tells_a_clear_yes_from_a_change(vega):
    assert vega.decide("Yes, go ahead.", APPROVAL).choice == "yes"
    assert vega.decide("make it 11 instead", APPROVAL).choice == "no"


def test_vega_flags_the_seeded_injections_and_not_complaints(vega):
    for text in INJECTION_TEXTS:
        block = untrusted_block(text, source="review")
        assert (
            vega.decide_many(block, [REVIEW_QUALITY, INJECTION])["injection"].choice == "injection"
        )
    complaint = untrusted_block("Charged extra for parts without asking first.", source="review")
    assert vega.decide(complaint, INJECTION).choice == "benign"
