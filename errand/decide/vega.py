"""The live decider: Vega (vegaml), following the paylane router pattern.

``vegaml.load(device="mps" if available else None)``, one ``fit`` per question on its few-shot
examples, a lock around ``decide()``, then the answer's ``choice`` and ``confidence``. Loading is
lazy and slow (it may download weights), so importing this module never touches torch or the
network.
"""

from collections.abc import Iterable, Sequence
from threading import Lock
from time import perf_counter
from typing import Any, Literal, Self

from errand.config import Settings
from errand.decide.base import Decision, DecisionError, DecisionUnavailable, Question

Device = Literal["auto", "mps", "cuda", "cpu"]


class VegaDecider:
    def __init__(
        self,
        questions: Iterable[Question],
        *,
        checkpoint: str | None = None,
        device: Device = "auto",
    ) -> None:
        self._questions = {question.id: question for question in questions}
        self._checkpoint = checkpoint
        self._device = device
        self._vega: Any = None
        self.fit_report: dict[str, Any] = {}
        # decide() captures activations into shared state on the model, so calls can't overlap.
        self._lock = Lock()

    @classmethod
    def from_settings(cls, questions: Iterable[Question], settings: Settings) -> Self:
        return cls(questions, checkpoint=settings.vega_model, device=settings.vega_device)

    def load(self) -> None:
        """Load the checkpoint and fit a head for each question with examples. Runs once when it
        works; a failure raises `DecisionUnavailable` and the next decision tries again."""
        with self._lock:
            if self._vega is not None:
                return
            try:
                self._vega = self._load()
            except Exception as exc:  # missing weights, no torch, out of memory, ...
                raise DecisionUnavailable(f"Vega could not be loaded: {exc}") from exc

    def _load(self) -> Any:
        import torch
        import vegaml

        device: str | None = self._device
        if device == "auto":
            # vegaml picks CUDA or CPU by itself; on Apple silicon the GPU is far faster.
            device = "mps" if torch.backends.mps.is_available() else None
        vega = vegaml.load(self._checkpoint, device=device)
        for question in self._questions.values():
            if question.examples:
                examples = [
                    (text, {question.id: label})
                    for label, texts in question.examples.items()
                    for text in texts
                ]
                self.fit_report.update(vega.fit(examples, _as_vega(question)))
        return vega

    def decide(self, text: str, question: Question) -> Decision:
        return self.decide_many(text, [question])[question.id]

    def decide_many(self, text: str, questions: Sequence[Question]) -> dict[str, Decision]:
        for question in questions:
            if self._questions.get(question.id) != question:
                # A head is fitted to one question's exact labels; any other needs a refit.
                raise DecisionError(
                    f"question {question.id!r} isn't the one this decider was fitted on"
                )
        self.load()
        asked: dict[str, dict[str, Any]] = {}
        for question in questions:
            asked.update(_as_vega(question))
        started = perf_counter()
        try:
            with self._lock:
                answers = self._vega.decide(text, asked)["answers"]
        except Exception as exc:
            raise DecisionUnavailable(f"Vega failed while deciding: {exc}") from exc
        latency_ms = (perf_counter() - started) * 1000
        return {
            question.id: Decision(
                question_id=question.id,
                choice=str(answers[question.id]["choice"]),
                confidence=float(answers[question.id]["confidence"]),
                probs={
                    str(label): float(p)
                    for label, p in answers[question.id].get("probs", {}).items()
                },
                readout=str(answers[question.id].get("readout", "engine")),
                abstain=bool(answers[question.id].get("abstain", False)),
                latency_ms=latency_ms,
            )
            for question in questions
        }


def _as_vega(question: Question) -> dict[str, dict[str, Any]]:
    return {
        question.id: {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": dict(question.criteria),
        }
    }
