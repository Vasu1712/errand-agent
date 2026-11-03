"""The decision client interface: typed, calibrated choices for routing, never generated text.

A `Question` is a closed set of labels, with optional few-shot examples per label. A decision below
``vega_confidence_threshold`` is the caller's cue to ask the user instead of guessing. Decisions
only route the conversation; approval comes from application code, never from a decision.
"""

from collections.abc import Sequence
from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Question(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    instructions: str
    # label -> what it means; None when the label speaks for itself.
    criteria: dict[str, str | None]
    # label -> example messages, used to fit a head for this question.
    examples: dict[str, tuple[str, ...]] = {}

    @model_validator(mode="after")
    def _labels_agree(self) -> Self:
        if len(self.criteria) < 2:
            raise ValueError("a question needs at least two labels")
        unknown = set(self.examples) - set(self.criteria)
        if unknown:
            raise ValueError(f"examples given for labels the question lacks: {sorted(unknown)}")
        return self

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.criteria)


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: str
    choice: str
    confidence: float = Field(ge=0.0, le=1.0)
    probs: dict[str, float] = {}
    readout: str  # "engine" or "ttt" from Vega, "scripted" in tests
    abstain: bool = False  # Vega declined to commit; treated like low confidence
    latency_ms: float = 0.0

    def is_confident(self, threshold: float) -> bool:
        return not self.abstain and self.confidence >= threshold


class DecisionError(Exception):
    """No decision could be made for this question."""


class DecisionUnavailable(DecisionError):
    """The decision model itself isn't working (it failed to load, or crashed)."""


@runtime_checkable
class DecisionClient(Protocol):
    def decide(self, text: str, question: Question) -> Decision: ...

    def decide_many(self, text: str, questions: Sequence[Question]) -> dict[str, Decision]:
        """Several questions about one text, answered from a single read where possible."""
        ...
