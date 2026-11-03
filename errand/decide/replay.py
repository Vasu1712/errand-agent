"""Record a live decider's answers (Vega), and replay them offline.

Decisions are keyed by question and by a digest of the exact text asked about, so a replay gives
each question the answer Vega gave to that text when it was recorded.
"""

import hashlib
from collections import deque
from collections.abc import Iterable, Sequence

from pydantic import BaseModel

from errand.decide.base import Decision, DecisionClient, DecisionError, Question


class DecisionCall(BaseModel):
    question_id: str
    text_digest: str
    text: str  # the first 200 characters, for people reading the recording
    decision: Decision | None = None
    error: str | None = None


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class RecordingDecider:
    """Passes every question to `inner` and records the decision in `calls`."""

    def __init__(self, inner: DecisionClient) -> None:
        self.inner = inner
        self.calls: list[DecisionCall] = []

    def decide(self, text: str, question: Question) -> Decision:
        return self.decide_many(text, [question])[question.id]

    def decide_many(self, text: str, questions: Sequence[Question]) -> dict[str, Decision]:
        digest = text_digest(text)
        try:
            decisions = self.inner.decide_many(text, questions)
        except DecisionError as exc:
            for question in questions:
                self.calls.append(
                    DecisionCall(
                        question_id=question.id, text_digest=digest, text=text[:200], error=str(exc)
                    )
                )
            raise
        for question in questions:
            self.calls.append(
                DecisionCall(
                    question_id=question.id,
                    text_digest=digest,
                    text=text[:200],
                    decision=decisions[question.id],
                )
            )
        return decisions


class ReplayDecider:
    """Answers from recorded decisions for the same question and text, in recorded order.

    A question about a text the recording never saw raises `DecisionError`: the replayed run has
    diverged from the recorded one.
    """

    def __init__(self, calls: Iterable[DecisionCall]) -> None:
        self._answers: dict[tuple[str, str], deque[DecisionCall]] = {}
        for call in calls:
            self._answers.setdefault((call.question_id, call.text_digest), deque()).append(call)

    @property
    def remaining(self) -> int:
        return sum(len(queue) for queue in self._answers.values())

    def decide(self, text: str, question: Question) -> Decision:
        queue = self._answers.get((question.id, text_digest(text)))
        if not queue:
            raise DecisionError(
                f"the recording has no {question.id!r} decision left for: {text[:80]!r}"
            )
        call = queue.popleft()
        if call.decision is None:
            raise DecisionError(call.error or "recorded as failed")
        return call.decision

    def decide_many(self, text: str, questions: Sequence[Question]) -> dict[str, Decision]:
        return {question.id: self.decide(text, question) for question in questions}
