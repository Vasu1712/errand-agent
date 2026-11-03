"""A decider that follows a script: deterministic decisions for tests and offline evals."""

from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence

from errand.decide.base import Decision, DecisionError, Question

Outcome = str | tuple[str, float]
Rule = Callable[[str], Outcome]


class ScriptedDecider:
    """Answers each question from the script entry for its id.

    An entry is either a sequence of steps, used one per call, or a single rule (a callable from
    the text to an outcome) applied to every call. An outcome is a label, decided with
    `default_confidence`, or a ``(label, confidence)`` pair, and the label must be one of the
    question's. A step may also be an exception, raised for that call.
    """

    def __init__(
        self,
        script: Mapping[str, Iterable[Outcome | Rule] | Rule],
        *,
        default_confidence: float = 0.99,
    ) -> None:
        self._rules: dict[str, Rule] = {}
        self._steps: dict[str, deque[Outcome | Rule]] = {}
        for question_id, entry in script.items():
            if callable(entry):
                self._rules[question_id] = entry
            else:
                self._steps[question_id] = deque(entry)
        self._default_confidence = default_confidence
        self.calls: list[tuple[str, str]] = []  # (question id, text)

    @property
    def remaining(self) -> int:
        """Scripted steps not used yet (rules don't count)."""
        return sum(len(queue) for queue in self._steps.values())

    def decide(self, text: str, question: Question) -> Decision:
        self.calls.append((question.id, text))
        if question.id in self._rules:
            step: Outcome | Rule = self._rules[question.id]
        elif self._steps.get(question.id):
            step = self._steps[question.id].popleft()
        else:
            raise DecisionError(f"the script has no decision left for {question.id!r}")
        if isinstance(step, BaseException):
            raise step
        outcome = step(text) if callable(step) else step
        choice, confidence = (
            (outcome, self._default_confidence) if isinstance(outcome, str) else outcome
        )
        if choice not in question.criteria:
            raise DecisionError(f"scripted choice {choice!r} is not a label of {question.id!r}")
        return Decision(
            question_id=question.id,
            choice=choice,
            confidence=confidence,
            probs={choice: confidence},
            readout="scripted",
        )

    def decide_many(self, text: str, questions: Sequence[Question]) -> dict[str, Decision]:
        return {question.id: self.decide(text, question) for question in questions}
