import threading
import time

import pytest
from pydantic import ValidationError

from errand.decide import (
    DecisionClient,
    DecisionError,
    Question,
    ScriptedDecider,
    VegaDecider,
)

INTENT = Question(
    id="intent",
    instructions="What does the user want to do?",
    criteria={"book": None, "reschedule": None, "cancel": None, "other": "anything else"},
)


def test_scripted_decider_follows_its_script_per_question():
    decider = ScriptedDecider({"intent": ["book", ("cancel", 0.6)]})
    first = decider.decide("Find a plumber this week", INTENT)
    second = decider.decide("Cancel it", INTENT)

    assert (first.choice, first.confidence, first.readout) == ("book", 0.99, "scripted")
    assert second.choice == "cancel"
    assert not second.is_confident(0.85)
    assert decider.calls == [("intent", "Find a plumber this week"), ("intent", "Cancel it")]


def test_a_rule_answers_every_call():
    decider = ScriptedDecider(
        {"intent": lambda text: "cancel" if "cancel" in text.lower() else "book"}
    )
    texts = ["Cancel my AC servicing", "Book a plumber", "please cancel"]
    assert [decider.decide(text, INTENT).choice for text in texts] == ["cancel", "book", "cancel"]


def test_scripted_decider_rejects_unknown_labels_and_running_out():
    with pytest.raises(DecisionError, match="not a label"):
        ScriptedDecider({"intent": ["refund"]}).decide("refund me", INTENT)
    with pytest.raises(DecisionError, match="no decision left"):
        ScriptedDecider({}).decide("hello", INTENT)


def test_questions_validate_their_labels():
    with pytest.raises(ValidationError):
        Question(id="q", instructions="?", criteria={"only": None})
    with pytest.raises(ValidationError):
        Question(id="q", instructions="?", criteria={"a": None, "b": None}, examples={"c": ("x",)})


def test_vega_decider_builds_without_loading_and_refuses_questions_it_was_not_fitted_on():
    decider = VegaDecider([INTENT])
    assert isinstance(decider, DecisionClient)
    other = INTENT.model_copy(update={"criteria": {"book": None, "other": None}})
    with pytest.raises(DecisionError, match="fitted on"):
        decider.decide("hello", other)


def test_vega_decides_one_call_at_a_time_and_each_caller_gets_its_own_answer():
    # Two users' turns decide at once. vegaml captures activations into shared state on the
    # model, so calls must never overlap, and nothing of one call may reach another's answer.
    class SharedStateModel:  # stands in for the loaded model: one activation buffer for all
        def __init__(self):
            self.inside, self.overlapped, self.activations = 0, False, None

        def decide(self, text, asked):
            self.inside += 1
            self.overlapped |= self.inside > 1
            self.activations = text
            time.sleep(0.001)  # let the other thread try to get in
            choice = "cancel" if "cancel" in self.activations.lower() else "book"
            self.inside -= 1
            return {"answers": {"intent": {"choice": choice, "confidence": 0.9}}}

    decider = VegaDecider([INTENT])
    decider._vega = model = SharedStateModel()  # loaded: no checkpoint needed
    barrier = threading.Barrier(2)
    answers: dict[str, list[str]] = {"asha": [], "ravi": []}

    def user(name, text):
        barrier.wait()
        for _ in range(50):
            answers[name].append(decider.decide(text, INTENT).choice)

    threads = [
        threading.Thread(target=user, args=("asha", "Book a plumber for Saturday")),
        threading.Thread(target=user, args=("ravi", "Cancel my electrician")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not model.overlapped
    assert answers == {"asha": ["book"] * 50, "ravi": ["cancel"] * 50}
